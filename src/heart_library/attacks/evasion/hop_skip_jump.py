# MIT License
#
# Copyright (C) HEART Authors 2024
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
# persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
# Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""
This module extends ART's `HopSkipJump` attack to support HEART.
"""

from typing import List, Optional, Tuple, Union

import numpy as np
from art.attacks.evasion import HopSkipJump
from art.config import ART_NUMPY_DTYPE
from art.utils import (check_and_transform_label_format, get_labels_np_array,
                       to_categorical)
from tqdm.auto import tqdm


def softmax(x):  # pragma: no cover
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


class HeartHopSkipJump(HopSkipJump):
    """
    Extension of ART's implementation of a generic laser attack case which
    supports channel first images.
    """

    def __init__(
        self,
        classifier,
        batch_size: int = 64,
        targeted: bool = False,
        norm: Union[int, float, str] = 2,
        max_iter: int = 50,
        max_eval: int = 10000,
        init_eval: int = 100,
        init_size: int = 100,
        verbose: bool = True,
    ) -> None:
        """
        Create a HopSkipJump attack instance.

        :param classifier: A trained classifier.
        :param batch_size: The size of the batch used by the estimator during inference.
        :param targeted: Should the attack target one specific class.
        :param norm: Order of the norm. Possible values: "inf", np.inf or 2.
        :param max_iter: Maximum number of iterations.
        :param max_eval: Maximum number of evaluations for estimating gradient.
        :param init_eval: Initial number of evaluations for estimating gradient.
        :param init_size: Maximum number of trials for initial generation of adversarial examples.
        :param verbose: Show progress bars.
        """
        super().__init__(
            classifier=classifier,
            batch_size=batch_size,
            targeted=targeted,
            norm=norm,
            max_iter=max_iter,
            max_eval=max_eval,
            init_eval=init_eval,
            init_size=init_size,
            verbose=verbose,
        )

        self.total_queries: np.ndarray = np.array([])
        self.adv_query_idx: List = []
        self.perturbs: List = []
        self.perturbs_iter: List = []
        self.confs: List = []
        self.curr_idx: int = 0
        self.curr_val: np.ndarray = np.array([])

    def generate(self, x: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:  # pragma: no cover
        """
        Generate adversarial samples and return them in an array.

        :param x: An array with the original inputs to be attacked.
        :param y: Target values (class labels) one-hot-encoded of shape `(nb_samples, nb_classes)` or indices of shape
                  (nb_samples,).
        :param mask: An array with a mask broadcastable to input `x` defining where to apply adversarial perturbations.
                     Shape needs to be broadcastable to the shape of x and can also be of the same shape as `x`. Any
                     features for which the mask is zero will not be adversarially perturbed.
        :type mask: `np.ndarray`
        :param x_adv_init: Initial array to act as initial adversarial examples. Same shape as `x`.
        :type x_adv_init: `np.ndarray`
        :param resume: Allow users to continue their previous attack.
        :type resume: `bool`
        :return: An array holding the adversarial examples.
        """
        mask = kwargs.get("mask")
        self.total_queries = np.zeros(x.shape[0])
        self.adv_query_idx = [[] for i in range(x.shape[0])]
        self.perturbs = [[] for i in range(x.shape[0])]
        self.perturbs_iter = [[] for i in range(x.shape[0])]
        self.confs = [[] for i in range(x.shape[0])]
        self.curr_idx = 0
        self.curr_val = np.array([])

        if y is None:
            # Throw error if attack is targeted, but no targets are provided
            if self.targeted:  # pragma: no cover
                raise ValueError("Target labels `y` need to be provided for a targeted attack.")

            # Use model predictions as correct outputs
            y = get_labels_np_array(self.estimator.predict(x, batch_size=self.batch_size))  # type: ignore
            self.total_queries = self.total_queries + 1

        y = check_and_transform_label_format(y, nb_classes=self.estimator.nb_classes)
        if not isinstance(y, np.ndarray):
            raise ValueError("Targets not correctly formatted.")
        if self.estimator.nb_classes == 2 and y.shape[1] == 1:  # pragma: no cover
            raise ValueError(
                "This attack has not yet been tested for binary classification with a single output classifier."
            )

        # Check whether users need a stateful attack
        resume = kwargs.get("resume")

        if resume is not None and resume:
            start = self.curr_iter
        else:
            start = 0

        # Check the mask
        if mask is not None:
            if len(mask.shape) == len(x.shape):
                mask = mask.astype(ART_NUMPY_DTYPE)
            else:
                mask = np.array([mask.astype(ART_NUMPY_DTYPE)] * x.shape[0])
        else:
            mask = np.array([None] * x.shape[0])

        # Get clip_min and clip_max from the classifier or infer them from data
        if self.estimator.clip_values is not None:
            clip_min, clip_max = self.estimator.clip_values
        else:
            clip_min, clip_max = np.min(x), np.max(x)

        # Prediction from the original images
        preds = np.argmax(self.estimator.predict(x, batch_size=self.batch_size), axis=1)
        self.total_queries = self.total_queries + 1

        # Prediction from the initial adversarial examples if not None
        x_adv_init = kwargs.get("x_adv_init")

        if x_adv_init is not None:
            # Add mask param to the x_adv_init
            for i in range(x.shape[0]):
                if mask[i] is not None:
                    x_adv_init[i] = x_adv_init[i] * mask[i] + x[i] * (1 - mask[i])

            # Do prediction on the init
            init_pred_logits = self.estimator.predict(x_adv_init, batch_size=self.batch_size)
            init_preds = np.argmax(init_pred_logits, axis=1)
            self.total_queries = self.total_queries + 1
            for i in range(x_adv_init.shape[0]):
                dist = np.linalg.norm(x[i] - x_adv_init[i])
                self.perturbs[i] += [dist]
                # self.confs[i] += [np.max(softmax(init_pred_logits[i]))]

        else:
            init_preds = [None] * len(x)
            x_adv_init = [None] * len(x)

        # Assert that, if attack is targeted, y is provided
        if self.targeted and y is None:  # pragma: no cover
            raise ValueError("Target labels `y` need to be provided for a targeted attack.")

        # Some initial setups
        x_adv = x.astype(ART_NUMPY_DTYPE)
        self.original_image = x

        y = np.argmax(y, axis=1)

        # Generate the adversarial samples
        for ind, val in enumerate(tqdm(x_adv, desc="HopSkipJump", disable=not self.verbose)):
            self.curr_iter: int = start
            self.curr_idx = ind
            self.curr_val = val

            if self.targeted:
                x_adv[ind] = self._perturb(
                    x=val,
                    y=y[ind],  # type: ignore
                    y_p=preds[ind],
                    init_pred=init_preds[ind],
                    adv_init=x_adv_init[ind],
                    mask=mask[ind],
                    clip_min=clip_min,
                    clip_max=clip_max,
                )

            else:
                x_adv[ind] = self._perturb(
                    x=val,
                    y=-1,
                    y_p=preds[ind],
                    init_pred=init_preds[ind],
                    adv_init=x_adv_init[ind],
                    mask=mask[ind],
                    clip_min=clip_min,
                    clip_max=clip_max,
                )

        y = to_categorical(y, self.estimator.nb_classes)  # type: ignore

        return x_adv

    def _init_sample(
        self,
        x: np.ndarray,
        y: int,
        y_p: int,
        init_pred: int,
        adv_init: np.ndarray,
        mask: Optional[np.ndarray],
        clip_min: float,
        clip_max: float,
    ) -> Optional[Union[np.ndarray, Tuple[np.ndarray, int]]]:  # pragma: no cover
        """
        Find initial adversarial example for the attack.

        :param x: An array with 1 original input to be attacked.
        :param y: If `self.targeted` is true, then `y` represents the target label.
        :param y_p: The predicted label of x.
        :param init_pred: The predicted label of the initial image.
        :param adv_init: Initial array to act as an initial adversarial example.
        :param mask: An array with a mask to be applied to the adversarial perturbations. Shape needs to be
                     broadcastable to the shape of x. Any features for which the mask is zero will not be adversarially
                     perturbed.
        :param clip_min: Minimum value of an example.
        :param clip_max: Maximum value of an example.
        :return: An adversarial example.
        """
        nprd = np.random.RandomState()
        initial_sample = None

        if self.targeted:
            # Attack satisfied
            if y == y_p:
                return None

            # Attack unsatisfied yet and the initial image satisfied
            if adv_init is not None and init_pred == y:
                return adv_init.astype(ART_NUMPY_DTYPE), init_pred

            # Attack unsatisfied yet and the initial image unsatisfied
            for _ in range(self.init_size):
                random_img = nprd.uniform(clip_min, clip_max, size=x.shape).astype(x.dtype)

                if mask is not None:
                    random_img = random_img * mask + x * (1 - mask)

                pred_logits = self.estimator.predict(np.array([random_img]), batch_size=self.batch_size)
                random_class = np.argmax(
                    pred_logits,
                    axis=1,
                )[0]
                self.total_queries[self.curr_idx] += np.array([random_img]).shape[0]
                dist = np.linalg.norm(x - random_img)
                self.perturbs[self.curr_idx] += [dist]

                if random_class == y:
                    self.adv_query_idx[self.curr_idx].append(self.total_queries[self.curr_idx])
                    self.confs[self.curr_idx] += [np.max(softmax(pred_logits[0]))]

                    # Binary search to reduce the l2 distance to the original image
                    random_img = self._binary_search(
                        current_sample=random_img,
                        original_sample=x,
                        target=y,
                        norm=2,
                        clip_min=clip_min,
                        clip_max=clip_max,
                        threshold=0.001,
                    )
                    initial_sample = random_img, random_class

                    break
        else:
            # The initial image satisfied
            if adv_init is not None and init_pred != y_p:
                return adv_init.astype(ART_NUMPY_DTYPE), y_p

            # The initial image unsatisfied
            for _ in range(self.init_size):
                random_img = nprd.uniform(clip_min, clip_max, size=x.shape).astype(x.dtype)

                if mask is not None:
                    random_img = random_img * mask + x * (1 - mask)

                pred_logits = self.estimator.predict(np.array([random_img]), batch_size=self.batch_size)
                random_class = np.argmax(
                    pred_logits,
                    axis=1,
                )[0]
                self.total_queries[self.curr_idx] += np.array([random_img]).shape[0]
                dist = np.linalg.norm(x - random_img)
                self.perturbs[self.curr_idx] += [dist]

                if random_class != y_p:
                    self.adv_query_idx[self.curr_idx].append(self.total_queries[self.curr_idx])
                    self.confs[self.curr_idx] += [np.max(softmax(pred_logits[0]))]

                    # Binary search to reduce the l2 distance to the original image
                    random_img = self._binary_search(
                        current_sample=random_img,
                        original_sample=x,
                        target=y_p,
                        norm=2,
                        clip_min=clip_min,
                        clip_max=clip_max,
                        threshold=0.001,
                    )
                    initial_sample = random_img, y_p
                    break

        return initial_sample

    def _attack(
        self,
        initial_sample: np.ndarray,
        original_sample: np.ndarray,
        target: int,
        mask: Optional[np.ndarray],
        clip_min: float,
        clip_max: float,
    ) -> np.ndarray:  # pragma: no cover
        """
        Main function for the boundary attack.

        :param initial_sample: An initial adversarial example.
        :param original_sample: The original input.
        :param target: The target label.
        :param mask: An array with a mask to be applied to the adversarial perturbations. Shape needs to be
                     broadcastable to the shape of x. Any features for which the mask is zero will not be adversarially
                     perturbed.
        :param clip_min: Minimum value of an example.
        :param clip_max: Maximum value of an example.
        :return: an adversarial example.
        """
        # Set current perturbed image to the initial image
        current_sample = initial_sample
        # print(f'Image {self.curr_idx} starting eval at query idx {self.total_queries[self.curr_idx]}')
        # Main loop to wander around the boundary
        for _ in range(self.max_iter):
            # First compute delta
            delta = self._compute_delta(
                current_sample=current_sample,
                original_sample=original_sample,
                clip_min=clip_min,
                clip_max=clip_max,
            )

            # Then run binary search
            current_sample = self._binary_search(
                current_sample=current_sample,
                original_sample=original_sample,
                norm=self.norm,
                target=target,
                clip_min=clip_min,
                clip_max=clip_max,
            )

            # Next compute the number of evaluations and compute the update
            num_eval = min(int(self.init_eval * np.sqrt(self.curr_iter + 1)), self.max_eval)

            update = self._compute_update(
                current_sample=current_sample,
                num_eval=num_eval,
                delta=delta,
                target=target,
                mask=mask,
                clip_min=clip_min,
                clip_max=clip_max,
            )

            # Finally run step size search by first computing epsilon
            if self.norm == 2:
                dist = np.linalg.norm(original_sample - current_sample)
            else:
                dist = np.max(abs(original_sample - current_sample))

            epsilon = 2.0 * dist / np.sqrt(self.curr_iter + 1)
            success = False

            while not success:
                epsilon /= 2.0
                potential_sample = current_sample + epsilon * update
                success = self._adversarial_satisfactory(  # type: ignore
                    samples=potential_sample[None],
                    target=target,
                    clip_min=clip_min,
                    clip_max=clip_max,
                )

            # Update current sample
            current_sample = np.clip(potential_sample, clip_min, clip_max)

            # Calc perturbs for iter
            dist = np.linalg.norm(self.original_image[self.curr_idx] - current_sample)
            self.perturbs_iter[self.curr_idx] += [dist]

            # Update current iteration
            self.curr_iter += 1

            # If attack failed. return original sample
            if np.isnan(current_sample).any():  # pragma: no cover
                return original_sample

        return current_sample

    def _adversarial_satisfactory(
        self, samples: np.ndarray, target: int, clip_min: float, clip_max: float
    ) -> np.ndarray:  # pragma: no cover
        """
        Check whether an image is adversarial.

        :param samples: A batch of examples.
        :param target: The target label.
        :param clip_min: Minimum value of an example.
        :param clip_max: Maximum value of an example.
        :return: An array of 0/1.
        """
        samples = np.clip(samples, clip_min, clip_max)
        pred_logits = self.estimator.predict(samples, batch_size=self.batch_size)
        preds = np.argmax(pred_logits, axis=1)
        self.total_queries[self.curr_idx] += samples.shape[0]  # as num_eval might make > 1

        if self.targeted:
            result = preds == target
        else:
            result = preds != target

        for i, res in enumerate(result):
            if res:
                dist = np.linalg.norm(self.curr_val - samples[i])
                self.perturbs[self.curr_idx] += [dist]
                self.confs[self.curr_idx] += [np.max(softmax(pred_logits[i]))]
                self.adv_query_idx[self.curr_idx].append(self.total_queries[self.curr_idx] - len(result) + (i + 1))
        return result
