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
"""This module extends ART's `HopSkipJump` attack to support HEART."""

from typing import Any, Optional, Union

import numpy as np
from art.attacks.evasion import HopSkipJump
from art.config import ART_NUMPY_DTYPE
from art.utils import check_and_transform_label_format, get_labels_np_array, to_categorical
from numpy.typing import NDArray
from tqdm.auto import tqdm


def _softmax(x: NDArray[np.float32]) -> NDArray[np.float32]:
    """Compute softmax values for each sets of scores in x.

    Args:
        x (NDArray[np.float32]): Sets of scores.

    Returns:
        NDArray[np.float32]: Softmax values for all set of scores.
    """

    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


class HeartHopSkipJump(HopSkipJump):  # type: ignore
    """Extension of ART's implementation of a generic laser attack case which
    supports channel first images.

    Args:
        HopSkipJump (_type_): HopSkipJump object to be wrapped.

    Examples
    --------

    We can create a HeartHopSkipJump attack by defining the image data, model parameters, and attack specification:

    >>> from torchvision.models import resnet18, ResNet18_Weights
    >>> from heart_library.estimators.classification.pytorch import JaticPyTorchClassifier
    >>> import torch
    >>> from datasets import load_dataset
    >>> from heart_library.attacks.evasion import HeartHopSkipJump
    >>> from heart_library.attacks.attack import JaticAttack

    Define the JaticPyTorchClassifier inputs, in this case for image classification:

    >>> data = load_dataset("cifar10", split="test[0:10]")
    >>> model = resnet18(ResNet18_Weights)
    >>> loss_fn = torch.nn.CrossEntropyLoss(reduction="sum")
    >>> optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    >>> jptc = JaticPyTorchClassifier(
    ...     model=model,
    ...     loss=loss_fn,
    ...     optimizer=optimizer,
    ...     input_shape=(3, 32, 32),
    ...     nb_classes=10,
    ...     clip_values=(0, 255),
    ...     preprocessing=(0.0, 255),
    ... )

    Define the HeartHopSkipJump attack, wrap in HEART attack class and execute:

    >>> hsj_attack = HeartHopSkipJump(
    ...     classifier=jptc, targeted=True, verbose=True, max_iter=50, max_eval=10, init_eval=10
    ... )
    >>> attack = JaticAttack(hsj_attack, norm=2)

    Generate adversarial images:

    >>> x_adv, y, metadata = attack(data=data)
    >>> x_adv[0][0][0][0][0]
    158.0
    """

    def __init__(
        self,
        classifier: Any,  # noqa ANN401
        batch_size: int = 64,
        targeted: bool = False,
        norm: Union[float, str] = 2,
        max_iter: int = 50,
        max_eval: int = 10000,
        init_eval: int = 100,
        init_size: int = 100,
        verbose: bool = True,
    ) -> None:
        """Create a HopSkipJump attack instance.

        Args:
            classifier (Any): A trained classifier.
            batch_size (int, optional): The size of the batch used by the estimator during inference. Defaults to 64.
            targeted (bool, optional): Should the attack target one specific class. Defaults to False.
            norm (Union[int, float, str], optional): Order of the norm. Possible values: "inf", np.inf or 2.
                Defaults to 2.
            max_iter (int, optional): Maximum number of iterations. Defaults to 50.
            max_eval (int, optional): Maximum number of evaluations for estimating gradient. Defaults to 10000.
            init_eval (int, optional): Initial number of evaluations for estimating gradient. Defaults to 100.
            init_size (int, optional): Maximum number of trials for initial generation of adversarial examples.
                Defaults to 100.
            verbose (bool, optional): Show progress bars. Defaults to True.
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

        self.total_queries: NDArray[np.float32] = np.array([])
        self.adv_query_idx: list[list[np.int32]] = []
        self.perturbs: list[list[np.float32]] = []
        self.perturbs_iter: list[list[np.float32]] = []
        self.confs: list[list[np.float32]] = []
        self.curr_idx: int = 0
        self.curr_val: NDArray[np.float32] = np.array([])

    def generate(
        self,
        x: NDArray[np.float32],
        y: Optional[NDArray[np.float32]] = None,
        **kwargs: Any,  # noqa ANN401
    ) -> NDArray[np.float32]:
        """Generate adversarial samples and return them in an array.

        Args:
            x (NDArray[np.float32]): An array with the original inputs to be attacked.
            y (Optional[NDArray[np.float32]], optional): Target values (class labels) one-hot-encoded of shape
                `(nb_samples, nb_classes)` or indices of shape (nb_samples,). Defaults to None.
            mask (NDArray[np.float32]):
                An array with a mask broadcastable to input `x` defining where to apply adversarial perturbations.
                Shape needs to be broadcastable to the shape of x and can also be of the same shape as `x`.
                Any features for which the mask is zero will not be adversarially perturbed.
            x_adv_init (NDArray[np.float32]): Initial array to act as initial adversarial examples. Same shape as `x`.
            resume (bool): Allow users to continue their previous attack.
        Raises:
            ValueError: if target labels y are not provided.
            ValueError: if target labels y are not correctly provided as an np.ndarray.
            ValueError: if attack has not yet been tested for binary classification with a single output classifier..
            ValueError: if attack is targeted and target labels y are not provided.

        Returns:
            NDArray[np.float32]: An array holding the adversarial examples.
        """
        mask = kwargs.get("mask")
        self.total_queries = np.zeros(x.shape[0], dtype=np.float32)
        self.adv_query_idx = [[] for _ in range(x.shape[0])]
        self.perturbs = [[] for _ in range(x.shape[0])]
        self.perturbs_iter = [[] for _ in range(x.shape[0])]
        self.confs = [[] for _ in range(x.shape[0])]
        self.curr_idx = 0
        self.curr_val = np.array([])

        # Handle if y is None
        y = self.__check_y_none(x, y)
        y = check_and_transform_label_format(y, nb_classes=self.estimator.nb_classes)
        if not isinstance(y, np.ndarray):
            raise ValueError("Targets not correctly formatted.")
        if self.estimator.nb_classes == 2 and y.shape[1] == 1:  # pragma: no cover
            raise ValueError(
                "This attack has not yet been tested for binary classification with a single output classifier.",
            )

        # Check whether users need a stateful attack
        resume = kwargs.get("resume")

        start = self.curr_iter if resume is not None and resume else 0

        # Check the mask
        mask = self.__check_mask(x, mask, ART_NUMPY_DTYPE)

        # Get clip_min and clip_max from the classifier or infer them from data
        clip_min, clip_max = self.__clip_min_max(x)

        # Prediction from the original images
        preds = np.argmax(self.estimator.predict(x, batch_size=self.batch_size), axis=1)
        self.total_queries = self.total_queries + 1

        # Prediction from the initial adversarial examples if not None
        x_adv_init = kwargs.get("x_adv_init")

        # Add mask param to x_adv_init and perform prediction on x_adv_init.
        init_preds, x_adv_init = self.__pred_init(x, x_adv_init, mask)

        # Assert that, if attack is targeted, y is provided
        if self.targeted and y is None:  # pragma: no cover
            raise ValueError("Target labels `y` need to be provided for a targeted attack.")

        # Some initial setups
        x_adv = x.astype(ART_NUMPY_DTYPE)
        self.original_image = x

        y = np.argmax(y, axis=1)

        # Generate the adversarial samples
        if y is not None:
            x_adv, y = self.__gen_adv_samples(y, start, preds, init_preds, x_adv, x_adv_init, mask, clip_min, clip_max)

        y = to_categorical(y, self.estimator.nb_classes)  # type: ignore

        return x_adv

    def __check_y_none(self, x: NDArray[np.float32], y: Optional[NDArray[np.float32]]) -> NDArray[np.float32]:
        """Handle if y is None.

        Args:
            x (NDArray[np.float32]): An array with the original inputs to be attacked.
            y (NDArray[np.float32]): Target labels.

        Raises:
            ValueError: If target labels y are not provided for a targeted attack.

        Returns:
            NDArray[np.float32]: Use model predictions as correct outputs.
        """
        if y is None:
            # Throw error if attack is targeted, but no targets are provided
            if self.targeted:  # pragma: no cover
                raise ValueError("Target labels `y` need to be provided for a targeted attack.")

            # Use model predictions as correct outputs
            y = get_labels_np_array(self.estimator.predict(x, batch_size=self.batch_size))  # type: ignore
            self.total_queries = self.total_queries + 1
        return y

    def __check_mask(
        self,
        x: NDArray[np.float32],
        mask: Optional[Any],  # noqa ANN401
        art_numpy_dtype: NDArray[np.float32],
    ) -> NDArray[np.float32]:
        """Confirm mask format.

        Args:
            x (NDArray[np.float32]): An array with the original inputs to be attacked.
            mask (Optional[Any]):
                An array with a mask broadcastable to input `x` defining where to apply adversarial perturbations.
                Shape needs to be broadcastable to the shape of x and can also be of the same shape as `x`.
                Any features for which the mask is zero will not be adversarially perturbed.
            art_numpy_dtype (NDArray[np.float32]): np.float32

        Returns:
            NDArray[np.float32]: Mask as np.float32 type.
        """
        # Check the mask
        if mask is not None:
            if len(mask.shape) == len(x.shape):
                mask = mask.astype(art_numpy_dtype)
            else:
                mask = np.array([mask.astype(art_numpy_dtype)] * x.shape[0])
        else:
            mask = np.array([None] * x.shape[0])
        return mask

    def __clip_min_max(self, x: NDArray[np.float32]) -> tuple[float, float]:
        """Get clip_min and clip_max from the classifier or infer them from data.

        Args:
            x (NDArray[np.float32]): An array with the original inputs to be attacked.

        Returns:
            tuple[float, float]: Tuple of min and max clip values.
        """
        # Get clip_min and clip_max from the classifier or infer them from data
        clip_min: Any
        clip_max: Any
        if self.estimator.clip_values is not None:
            clip_min, clip_max = self.estimator.clip_values
        else:
            clip_min, clip_max = np.min(x), np.max(x)
        return clip_min, clip_max

    def __pred_init(
        self,
        x: NDArray[np.float32],
        x_adv_init: Optional[Any],  # noqa ANN401
        mask: NDArray[np.float32],
    ) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        """Add mask param to x_adv_init and perform prediction on x_adv_init.

        Args:
            x (NDArray[np.float32]): An array with the original inputs to be attacked.
            x_adv_init (Optional[Any]): Initial array to act as initial adversarial examples. Same shape as `x`.
            mask (NDArray[np.float32]):
                An array with a mask broadcastable to input `x` defining where to apply adversarial perturbations.
                Shape needs to be broadcastable to the shape of x and can also be of the same shape as `x`.
                Any features for which the mask is zero will not be adversarially perturbed.

        Returns:
            tuple[NDArray[np.float32], NDArray[np.float32]]: Initial adversarial examples.
        """
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
        return init_preds, x_adv_init

    def __gen_adv_samples(
        self,
        y: NDArray[np.float32],
        start: int,
        preds: NDArray[np.float32],
        init_preds: NDArray[np.float32],
        x_adv: NDArray[np.float32],
        x_adv_init: NDArray[np.float32],
        mask: NDArray[np.float32],
        clip_min: float,
        clip_max: float,
    ) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        """Generate adversarial samples.

        Args:
            y (NDArray[np.float32]): Target values (class labels) one-hot-encoded of shape `(nb_samples, nb_classes)`
                or indices of shape (nb_samples,).
            start (int): Current iteration.
            preds (NDArray[np.float32]): The max indices of the predicted labels of the intial image.
            init_preds (NDArray[np.float32]): The max indices of the predicted labels of the adversarial image.
            x_adv: Copy of x in format np.float32.
            x_adv_init (NDArray[np.float32]): Initial array to act as initial adversarial examples. Same shape as `x`.
            mask (NDArray[np.float32]):
                An array with a mask broadcastable to input `x` defining where to apply adversarial perturbations.
                Shape needs to be broadcastable to the shape of x and can also be of the same shape as `x`.
                Any features for which the mask is zero will not be adversarially perturbed.
            clip_min (float): Clip min value.
            clip_max (float): Clip max value.

        Returns:
            tuple[NDArray[np.float32], NDArray[np.float32]]: Adversarial samples and original targets.
        """
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
                    init_pred=init_preds[ind],  # type: ignore
                    adv_init=x_adv_init[ind],  # type: ignore
                    mask=mask[ind],
                    clip_min=clip_min,
                    clip_max=clip_max,
                )

            else:
                x_adv[ind] = self._perturb(
                    x=val,
                    y=-1,
                    y_p=preds[ind],
                    init_pred=init_preds[ind],  # type: ignore
                    adv_init=x_adv_init[ind],  # type: ignore
                    mask=mask[ind],
                    clip_min=clip_min,
                    clip_max=clip_max,
                )
        return x_adv, y

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
    ) -> Optional[Union[np.ndarray, tuple[np.ndarray, int]]]:  # pragma: no cover
        """Find initial adversarial example for the attack.

        Args:
            x (np.ndarray): An array with 1 original input to be attacked.
            y (int): If `self.targeted` is true, then `y` represents the target label.
            y_p (int): The predicted label of x.
            init_pred (int): The predicted label of the initial image.
            adv_init (np.ndarray): Initial array to act as an initial adversarial example.
            mask (Optional[np.ndarray]): An array with a mask to be applied to the adversarial perturbations.
                Shape needs to be broadcastable to the shape of x.
                Any features for which the mask is zero will not be adversarially perturbed.
            clip_min (float): Minimum value of an example.
            clip_max (float): Maximum value of an example.

        Returns:
            Optional[Union[np.ndarray, Tuple[np.ndarray, int]]]: An adversarial example.
        """
        nprd = np.random.RandomState()
        # initial_sample = None

        if self.targeted:
            initial_sample = self.__targeted_attack(x, y, y_p, init_pred, adv_init, mask, clip_min, clip_max, nprd)
        else:
            initial_sample = self.__non_targeted_attack(x, y_p, init_pred, adv_init, mask, clip_min, clip_max, nprd)
        return initial_sample

    def __targeted_attack(
        self,
        x: np.ndarray,
        y: int,
        y_p: int,
        init_pred: int,
        adv_init: np.ndarray,
        mask: Optional[np.ndarray],
        clip_min: float,
        clip_max: float,
        nprd: Any,  # noqa ANN401
    ) -> Optional[Union[np.ndarray, tuple[np.ndarray, int]]]:  # pragma: no cover
        """Find initial adversarial example if attack is targeted.

        Args:
            x (np.ndarray): An array with 1 original input to be attacked.
            y (int): If `self.targeted` is true, then `y` represents the target label.
            y_p (int): The predicted label of x.
            init_pred (int): The predicted label of the initial image.
            adv_init (np.ndarray): Initial array to act as an initial adversarial example.
            mask (Optional[np.ndarray]): An array with a mask to be applied to the adversarial perturbations.
                    Shape needs to be broadcastable to the shape of x.
                    Any features for which the mask is zero will not be adversarially perturbed.
            clip_min (float): Minimum value of an example.
            clip_max (float): Maximum value of an example.
            nprd (Any): Random state.

        Returns:
            Optional[Union[np.ndarray, Tuple[np.ndarray, int]]]: An adversarial example.
        """
        initial_sample = None
        # Attack satisfied
        if y == y_p:
            return None

        # Attack unsatisfied yet and the initial image satisfied
        if adv_init is not None and init_pred == y:
            return adv_init.astype(ART_NUMPY_DTYPE), init_pred

        # Attack unsatisfied yet and the initial image unsatisfied
        for _ in range(self.init_size):
            random_img = nprd.uniform(clip_min, clip_max, size=x.shape).astype(x.dtype)

            random_img = self.__rearrange_image(mask, random_img, x)

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
                self.confs[self.curr_idx] += [np.max(_softmax(pred_logits[0]))]

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
        return initial_sample

    def __rearrange_image(self, mask: Optional[np.ndarray], random_img: np.ndarray, x: np.ndarray) -> np.ndarray:
        """Rearrange random image based on mask and x.

        Args:
            mask (Optional[np.ndarray]): An array with a mask to be applied to the adversarial perturbations.
                    Shape needs to be broadcastable to the shape of x.
                    Any features for which the mask is zero will not be adversarially perturbed.
            random_img (np.ndarray): Random image.
            x (np.ndarray): An array with 1 original input to be attacked.

        Returns:
            np.ndarray: Rearranged image.
        """
        if mask is not None:
            random_img = random_img * mask + x * (1 - mask)
        return random_img

    def __non_targeted_attack(
        self,
        x: np.ndarray,
        y_p: int,
        init_pred: int,
        adv_init: np.ndarray,
        mask: Optional[np.ndarray],
        clip_min: float,
        clip_max: float,
        nprd: Any,  # noqa ANN401
    ) -> Optional[Union[np.ndarray, tuple[np.ndarray, int]]]:  # pragma: no cover
        """Find initial adversarial example if attack is not targeted.

        Args:
            x (np.ndarray): An array with 1 original input to be attacked.
            y_p (int): The predicted label of x.
            init_pred (int): The predicted label of the initial image.
            adv_init (np.ndarray): Initial array to act as an initial adversarial example.
            mask (Optional[np.ndarray]): An array with a mask to be applied to the adversarial perturbations.
                    Shape needs to be broadcastable to the shape of x.
                    Any features for which the mask is zero will not be adversarially perturbed.
            clip_min (float): Minimum value of an example.
            clip_max (float): Maximum value of an example.
            nprd (Any): Random state.

        Returns:
            Optional[Union[np.ndarray, Tuple[np.ndarray, int]]]: An adversarial example.
        """
        initial_sample = None
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
                self.confs[self.curr_idx] += [np.max(_softmax(pred_logits[0]))]

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
        """Main function for the boundary attack.

        Args:
            initial_sample (np.ndarray): An initial adversarial example.
            original_sample (np.ndarray): The original input.
            target (int): The target label.
            mask (Optional[np.ndarray]): An array with a mask to be applied to the adversarial perturbations.
                Shape needs to be broadcastable to the shape of x.
                Any features for which the mask is zero will not be adversarially perturbed.
            clip_min (float): Minimum value of an example.
            clip_max (float): Maximum value of an example.

        Returns:
            np.ndarray: an adversarial example.
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
            current_sample = np.clip(potential_sample, clip_min, clip_max)  # type: ignore

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
        self,
        samples: np.ndarray,
        target: int,
        clip_min: float,
        clip_max: float,
    ) -> np.ndarray:  # pragma: no cover
        """Check whether an image is adversarial.

        Args:
            samples (np.ndarray): A batch of examples.
            target (int): The target label.
            clip_min (float): Minimum value of an example.
            clip_max (float): Maximum value of an example.

        Returns:
            np.ndarray: An array of 0/1.
        """
        samples = np.clip(samples, clip_min, clip_max)
        pred_logits = self.estimator.predict(samples, batch_size=self.batch_size)
        preds = np.argmax(pred_logits, axis=1)
        self.total_queries[self.curr_idx] += samples.shape[0]  # as num_eval might make > 1

        result = preds == target if self.targeted else preds != target

        for i, res in enumerate(result):
            if res:
                dist = np.linalg.norm(self.curr_val - samples[i])
                self.perturbs[self.curr_idx] += [dist]
                self.confs[self.curr_idx] += [np.max(_softmax(pred_logits[i]))]
                self.adv_query_idx[self.curr_idx].append(self.total_queries[self.curr_idx] - len(result) + (i + 1))
        return result
