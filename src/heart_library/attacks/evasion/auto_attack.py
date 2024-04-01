# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2023
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
This module facilitates a Parallel version of AutoAttack.
"""
import logging
from copy import deepcopy
from typing import TYPE_CHECKING, Optional, Tuple, Union

import multiprocess
import numpy as np
from art.attacks import EvasionAttack
from art.attacks.evasion.auto_attack import AutoAttack
from art.config import ART_NUMPY_DTYPE
from art.utils import check_and_transform_label_format, get_labels_np_array

if TYPE_CHECKING:
    from art.utils import CLASSIFIER_TYPE

logger = logging.getLogger(__name__)


def run_attack(
    x: np.ndarray,
    y: np.ndarray,
    sample_is_robust: np.ndarray,
    attack: EvasionAttack,
    estimator: "CLASSIFIER_TYPE",
    norm: Union[int, float, str] = np.inf,
    eps: float = 0.3,
    gpu: bool = False,
    queue=None,
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run attack.

    :param x: An array of the original inputs.
    :param y: An array of the labels.
    :param sample_is_robust: Store the initial robustness of examples.
    :param attack: Evasion attack to run.
    :return: An array holding the adversarial examples.
    """
    # Attack only correctly classified samples
    if gpu:
        ident = multiprocess.current_process().ident
        gpu_id = queue.get()
        logger.info("%s: starting process on GPU %s", ident, gpu_id)
        estimator.set_device(f"cuda:{gpu_id}")
        attack.set_estimator(estimator)

    x_robust = x[sample_is_robust]
    y_robust = y[sample_is_robust]

    # Generate adversarial examples
    attack.verbose = False
    x_robust_adv = attack.generate(x=x_robust, y=y_robust, **kwargs)
    y_pred_robust_adv = estimator.predict(x_robust_adv)

    # Check and update successful examples
    rel_acc = 1e-4
    order = np.inf if norm == "inf" else norm
    assert isinstance(order, (int, float))
    norm_is_smaller_eps = (1 - rel_acc) * np.linalg.norm(
        (x_robust_adv - x_robust).reshape((x_robust_adv.shape[0], -1)), axis=1, ord=order
    ) <= eps

    if attack.targeted:
        samples_misclassified = np.argmax(y_pred_robust_adv, axis=1) == np.argmax(y_robust, axis=1)
    elif not attack.targeted:
        samples_misclassified = np.argmax(y_pred_robust_adv, axis=1) != np.argmax(y_robust, axis=1)
    else:  # pragma: no cover
        raise ValueError

    sample_is_not_robust = np.logical_and(samples_misclassified, norm_is_smaller_eps)

    x_robust[sample_is_not_robust] = x_robust_adv[sample_is_not_robust]
    x[sample_is_robust] = x_robust

    sample_is_robust[sample_is_robust] = np.invert(sample_is_not_robust)

    if gpu:
        logger.info("%s: Putting GPU %s back into queue.", ident, gpu_id)
        queue.put(gpu_id)

    return x, sample_is_robust


class JaticAutoAttack(AutoAttack):
    """
    JATIC defined extension of ART core AutoAttack
    """

    # Identify samples yet to have attack metadata identified
    SAMPLE_DEFAULT = -1
    # Identify samples misclassified therefore no attack metadata required
    SAMPLE_MISCLASSIFIED = -2

    def __init__(self, parallel: bool = False, gpu: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.parallel = parallel

        if gpu:
            from numba import cuda  # pylint: disable=C0415

            try:
                num_devices = len(cuda.gpus)
                logger.info("Found %s GPU devices.", num_devices)
                self.gpu = True
            except cuda.cudadrv.error.CudaSupportError as error:
                logger.debug("Cuda driver error: %s. Reverting to CPU.", error)
                self.gpu = False
        else:
            self.gpu = False

        if parallel:
            self.queue = multiprocess.Manager().Queue()
        self.best_attacks: np.ndarray = np.array([])

    def generate(self, x: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        """
        Generate adversarial samples and return them in an array.

        :param x: An array with the original inputs.
        :param y: Target values (class labels) one-hot-encoded of shape `(nb_samples, nb_classes)` or indices of shape
                  (nb_samples,). Only provide this parameter if you'd like to use true labels when crafting adversarial
                  samples. Otherwise, model predictions are used as labels to avoid the "label leaking" effect
                  (explained in this paper: https://arxiv.org/abs/1611.01236). Default is `None`.
        :param mask: An array with a mask broadcastable to input `x` defining where to apply adversarial perturbations.
                     Shape needs to be broadcastable to the shape of x and can also be of the same shape as `x`. Any
                     features for which the mask is zero will not be adversarially perturbed.
        :type mask: `np.ndarray`
        :return: An array holding the adversarial examples.
        """

        x_adv = x.astype(ART_NUMPY_DTYPE)
        if y is not None:
            y = check_and_transform_label_format(y, nb_classes=self.estimator_orig.nb_classes)

        if y is None:
            y = get_labels_np_array(self.estimator.predict(x, batch_size=self.batch_size))

        # Determine correctly predicted samples
        y_pred = self.estimator_orig.predict(x.astype(ART_NUMPY_DTYPE))
        sample_is_robust = np.argmax(y_pred, axis=1) == np.argmax(y, axis=1)

        # Set slots for images which have yet to be filled as SAMPLE_DEFAULT
        self.best_attacks = np.array([self.SAMPLE_DEFAULT] * len(x))
        # Set samples that are misclassified and do not need to be filled as SAMPLE_MISCLASSIFIED
        self.best_attacks[np.logical_not(sample_is_robust)] = self.SAMPLE_MISCLASSIFIED

        self.args = []
        # Untargeted attacks
        for attack in self.attacks:
            # Stop if all samples are misclassified
            if np.sum(sample_is_robust) == 0:
                break

            if attack.targeted:
                attack.set_params(targeted=False)

            if self.parallel:
                self.args.append(
                    (
                        deepcopy(x_adv),
                        deepcopy(y),
                        deepcopy(sample_is_robust),
                        deepcopy(attack),
                        deepcopy(self.estimator_orig),
                        deepcopy(self.norm),
                        deepcopy(self.eps),
                        deepcopy(self.gpu),
                        self.queue,
                    )
                )
            else:
                x_adv, sample_is_robust = run_attack(
                    x=x_adv,
                    y=y,
                    sample_is_robust=sample_is_robust,
                    attack=attack,
                    estimator=self.estimator,
                    norm=self.norm,
                    eps=self.eps,
                    gpu=self.gpu,
                    **kwargs,
                )
                # create a mask which identifies images which this attack was effective on
                # not including originally misclassified images
                atk_mask = np.logical_and(
                    np.array([i == self.SAMPLE_DEFAULT for i in self.best_attacks]), np.logical_not(sample_is_robust)
                )
                # update attack at image index with index of attack that was successful
                self.best_attacks[atk_mask] = self.attacks.index(attack)

        # Targeted attacks
        if self.targeted:
            # Labels for targeted attacks
            y_t = np.array([range(y.shape[1])] * y.shape[0])
            y_idx = np.argmax(y, axis=1)
            y_idx = np.expand_dims(y_idx, 1)
            y_t = y_t[y_t != y_idx]
            targeted_labels = np.reshape(y_t, (y.shape[0], -1))

            for attack in self.attacks:
                try:
                    attack.set_params(targeted=True)

                    for i in range(self.estimator.nb_classes - 1):
                        # Stop if all samples are misclassified
                        if np.sum(sample_is_robust) == 0:
                            break

                        target = check_and_transform_label_format(
                            targeted_labels[:, i], nb_classes=self.estimator.nb_classes
                        )
                        if self.parallel:
                            self.args.append(
                                (
                                    deepcopy(x_adv),
                                    deepcopy(target),
                                    deepcopy(sample_is_robust),
                                    deepcopy(attack),
                                    deepcopy(self.estimator_orig),
                                    deepcopy(self.norm),
                                    deepcopy(self.eps),
                                    deepcopy(self.gpu),
                                    self.queue,
                                )
                            )
                        else:
                            x_adv, sample_is_robust = run_attack(
                                x=x_adv,
                                y=target,
                                sample_is_robust=sample_is_robust,
                                attack=attack,
                                estimator=self.estimator,
                                norm=self.norm,
                                eps=self.eps,
                                gpu=self.gpu,
                                **kwargs,
                            )
                            # create a mask which identifies images which this attack was effective on
                            # not including originally misclassified images
                            atk_mask = np.logical_and(
                                np.array([i == self.SAMPLE_DEFAULT for i in self.best_attacks]),
                                np.logical_not(sample_is_robust),
                            )
                            # update attack at image index with index of attack that was successful
                            self.best_attacks[atk_mask] = self.attacks.index(attack)
                except ValueError as error:
                    logger.warning("Error completing attack: %s}", str(error))

        if self.parallel:
            if self.gpu:
                from numba import cuda  # pylint: disable=C0415

                devices = cuda.gpus
                for device in devices:
                    logger.info("Putting device %s into the queue.", device.id)
                    self.queue.put(device.id)
            with multiprocess.get_context("spawn").Pool() as pool:
                # Results come back in the order that they were issued
                results = pool.starmap(run_attack, self.args)
            perturbations = []
            is_robust = []
            for img_idx in range(len(x)):
                perturbations.append(np.array([np.linalg.norm(x[img_idx] - i[0][img_idx]) for i in results]))
                is_robust.append([i[1][img_idx] for i in results])
            best_attacks = np.argmin(np.where(np.invert(np.array(is_robust)), np.array(perturbations), np.inf), axis=1)
            x_adv = np.concatenate([results[best_attacks[img]][0][[img]] for img in range(len(x))])
            self.best_attacks = best_attacks
        return x_adv

    def __repr__(self) -> str:
        best_attack_meta = "\n".join(
            [
                f"image {i+1}: {str(self.args[idx][3])}" if idx != 0 else f"image {i+1}: n/a"
                for i, idx in enumerate(self.best_attacks)
            ]
        )
        auto_attack_meta = (
            f"AutoAttack(targeted={self.targeted}, parallel={self.parallel}, num_attacks={len(self.args)})"
        )
        return f"{auto_attack_meta}\nBestAttacks:\n{best_attack_meta}"
