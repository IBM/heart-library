# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (HEART) Authors 2024
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
"""This module implements a JATIC compatible ART Attack."""

import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Optional, Union

import numpy as np
from art.attacks import EvasionAttack
from maite.protocols import ArrayLike
from numpy.typing import NDArray

from heart_library.estimators.object_detection.pytorch import JaticPyTorchObjectDetectionOutput
from heart_library.utils import process_inputs_for_art


class JaticEvasionAttackOutput:
    """
    Dataclass output JaticEvasionAttackOutput

    Examples
    --------

    We can create a JaticAttack using ProjectedGradientDescentPyTorch and generate a JaticEvasionAttackOutput:

    >>> from art.attacks.evasion import ProjectedGradientDescentPyTorch
    >>> from heart_library.attacks.attack import JaticAttack
    >>> import torchvision
    >>> from torchvision.models import resnet18, ResNet18_Weights
    >>> import torch
    >>> import os
    >>> import numpy as np
    >>> from heart_library.estimators.classification.pytorch import JaticPyTorchClassifier

    Define the JaticPyTorchClassifier:

    >>> model = resnet18(ResNet18_Weights)
    >>> loss_fn = torch.nn.CrossEntropyLoss(reduction="sum")
    >>> optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    >>> classifier = JaticPyTorchClassifier(
    ...     model=model,
    ...     loss=loss_fn,
    ...     optimizer=optimizer,
    ...     input_shape=(3, 32, 32),
    ...     nb_classes=10,
    ...     clip_values=(0, 255),
    ...     channels_first=False,
    ...     preprocessing=(0.0, 255),
    ... )

    Prepare the data, execute the attack, and generate the output:

    >>> data = torchvision.datasets.CIFAR10("../data", train=False, download=False)
    >>> data = torch.utils.data.Subset(data, list(range(10)))

    >>> predictions = classifier(data)

    >>> attack = JaticAttack(
    ...     ProjectedGradientDescentPyTorch(
    ...         estimator=classifier, norm=np.inf, eps=8, eps_step=2, max_iter=5, targeted=False
    ...     ),
    ...     norm=2,
    ... )

    >>> x_test_adv, _, meta = attack(data=data)
    """

    def __init__(
        self,
        images: list[NDArray[np.float32]],
        targets: NDArray[np.float32],
        metadata: list[dict[str, Any]],
    ) -> None:  # pyright: ignore
        """JaticEvasionAttackOutput initialization.

        Args:
            images (List[NDArray[np.float32]]): Array representation of images.
            targets (NDArray[np.float32]): Targets.
            metadata (List[dict[str, Any]]): Metadata.
        """
        self.images = images
        self.targets = targets
        self.metadata = metadata

    def __len__(self) -> int:
        """Returns image count.

        Returns:
            int: Image count.
        """
        return len(self.images)

    def __getitem__(self, ind: int) -> tuple[NDArray[np.float32], NDArray[np.float32], dict[str, Any]]:
        """Returns images, targets, metadata.

        Args:
            ind (int): Index of image, target, metadata combination to be returned.

        Returns:
            Tuple[NDArray[np.float32], NDArray[np.float32], dict[str, Any]]:
                 Specified image, target, metadata combination.
        """
        return self.images[ind], self.targets[ind], self.metadata[ind]


@dataclass
class JaticPoisonAttackOutput:
    """Dataclass output JaticPoisonAttackOutput"""

    poisoning_examples: NDArray[np.float32]
    poisoning_labels: NDArray[np.float32]


class JaticAttack:
    """Wrapper for JATIC compatible attacks"""

    metadata: dict[str, Any]

    def __init__(self, attack: Any, norm: int = 0, id: Optional[str] = None) -> None:  # noqa ANN401
        """JaticAttack initialization.

        Args:
            attack (Any): Attack.
            norm (int, optional): The norm of the adversarial perturbation. Possible values: “inf”, np.inf, 1 or 2.
                 Defaults to 0.
        """
        self._attack = attack
        self._norm = norm
        self.metadata = {"id": id if id is not None else str(uuid.uuid4())}

    def reset_patch(self, patch: Union[ArrayLike, float]) -> None:
        """
        Reset the adversarial patch.

        :param patch: ArrayLike or float - the patch value to use for resetting the patch
        """
        # check attack is a patch attack
        if not hasattr(self._attack, "apply_patch"):
            raise ValueError("Cannot apply 'reset_patch' on attack that is not an adversarial patch attack.")
        if isinstance(patch, (np.ndarray, float)):
            self._attack.reset_patch(np.asarray(patch))
        else:
            raise ValueError(f"The patch must be of type ArrayLike or float and not of type {type(patch)}.")

    def __call__(
        self,
        data: Union[
            tuple[Sequence[NDArray[np.float32]], Sequence[NDArray[np.float32]], Sequence[dict[str, Any]]],
            NDArray[np.float32],
        ],
        **kwargs: Any,  # noqa ANN401
    ) -> tuple[
        Sequence[NDArray[np.float32]],
        Union[Sequence[NDArray[np.float32]], Sequence[JaticPyTorchObjectDetectionOutput], Optional[Any]],
        Sequence[dict[str, Any]],
    ]:
        """Convert input data to ART supported types, run the specified attack,
             and add benign predictions to metadata as needed.

        Args:
            data (Union[ Tuple[Sequence[NDArray[np.float32]], Sequence[NDArray[np.float32]], Sequence[dict[str, Any]]],
                 NDArray[np.float32], ]): Images, targets, metadata.

        Returns:
            Tuple[ Sequence[NDArray[np.float32]], Union[Sequence[NDArray[np.float32]],
                 Sequence[JaticPyTorchObjectDetectionOutput], Optional[Any]], Sequence[dict[str, Any]], ]:
                  JaticAttack output.
        """

        attack_output: tuple[
            Sequence[NDArray],
            Union[Sequence[NDArray], Sequence[JaticPyTorchObjectDetectionOutput], Optional[Any]],
            Sequence[dict[str, Any]],
        ] = ([], [], [])

        # convert to ART supported data types
        # assume data is a batch of data and returns a modified version of that batch
        x, y, meta = process_inputs_for_art(data)
        is_object_detection = False
        if y is not None and not isinstance(y, np.ndarray):
            # check if y is a list of dicts, as per object detection
            y, is_object_detection = self.__check_for_object_detection(y, is_object_detection)

        # run the attack
        if isinstance(self._attack, EvasionAttack):
            adv_output = self._attack.generate(x, y, **kwargs)  # pyright: ignore[reportArgumentType]

            # check if adversarial patch attack
            # requires extra step of applying the patch
            if hasattr(self._attack, "apply_patch"):
                scale = kwargs.get("scale", 1.0)
                image_mask = kwargs.get("mask", np.array([]))
                patch, mask, adv_images = self.__check_for_d_patches(adv_output, x, scale, image_mask)

                meta = self.__check_meta(meta, patch, mask, adv_images)

                attack_output = self.__output_for_evasion(is_object_detection, y, adv_images, meta, attack_output)

            # not a patch attack
            else:
                # convert to JATIC supported data types
                attack_output = self.__output_not_evasion(is_object_detection, y, adv_output, meta, attack_output)

            # check if should calculate the difference between benign
            # and adversarial images to store in meta
            attack_output = self.__calc_diff_meta(attack_output, x)

        return attack_output

    def get_attack(self) -> Any:  # noqa ANN401
        """Get the attack type which is being wrapped.

        Returns:
            Any: Attack.
        """
        return self._attack

    def __check_for_object_detection(
        self,
        y: Union[NDArray[np.float32], list[dict[str, Any]]],
        is_object_detection: bool,
    ) -> tuple[Union[NDArray[np.float32], list[dict[str, Any]]], bool]:
        """Check to see if this case is for object detection based on y.

        Args:
            y (Union[NDArray[np.float32], list[dict[str, Any]]]): Targets.
            is_object_detection (bool): Object detection flag.

        Returns:
            tuple[Union[NDArray[np.float32], list[dict[str, Any]]], bool]: Updated y and is_object_detection.
        """
        # check if y is a list of dicts, as per object detection
        if y is not None and not all(isinstance(i, dict) for i in y):
            y = np.array(y)
        else:
            is_object_detection = True
        return y, is_object_detection

    def __check_for_d_patches(
        self,
        adv_output: NDArray[np.float32],
        x: NDArray[np.float32],
        scale: np.float32,
        image_mask: Optional[NDArray[np.float32]] = None,
    ) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
        """Generate patch, mask, adv_images based on type of EvasionAttack.

        Args:
            adv_output (NDArray[np.float32]): Adversarial samples.
            x (NDArray[np.float32]): Input images.
            scale (np.float32): scaling factor for applied patch
            image_mask (NDArray[np.float32]): mask for patch applied to image

        Returns:
            tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]: Patch, mask, adversarial images.
        """
        from art.attacks.evasion import DPatch, RobustDPatch

        if image_mask is None:
            image_mask = np.array([])

        if isinstance(self._attack, (RobustDPatch, DPatch)):
            patch = adv_output
            mask = np.array([])
            adv_images = self._attack.apply_patch(x=x)
        else:
            patch, mask = adv_output
            if len(image_mask) > 0:
                adv_images = self._attack.apply_patch(x=x, scale=scale, mask=image_mask)
            else:
                adv_images = self._attack.apply_patch(x=x, scale=scale)
        return patch, mask, adv_images

    def __check_meta(
        self,
        meta: list[dict[str, Any]],
        patch: NDArray[np.float32],
        mask: NDArray[np.float32],
        adv_images: NDArray[np.float32],
    ) -> list[dict[str, Any]]:
        """Update metadata to include adversarial samples.

        Args:
            meta (list[dict[str, Any]]): Input metadata.
            patch (NDArray[np.float32]): Patch from adversarial samples.
            mask (NDArray[np.float32]): Mask from adversarial samples.
            adv_images (NDArray[np.float32]): Patched images.

        Returns:
            list[dict[str, Any]]: Updated metadata.
        """
        new_meta = []

        if meta is not None and None not in meta and len(meta) > 0:
            for item in meta:
                new_item = {**item, "patch": patch, "mask": mask}
                new_meta.append(new_item)
        else:
            new_meta.extend([{"patch": patch, "mask": mask} for _ in range(len(adv_images))])
        return new_meta

    def __output_for_evasion(
        self,
        is_object_detection: bool,
        y: Optional[Union[NDArray[np.float32], list[dict[str, Any]]]],
        adv_images: NDArray[np.float32],
        meta: list[dict[str, Any]],
        attack_output: tuple[
            Sequence[NDArray[np.float32]],
            Union[Sequence[NDArray[np.float32]], Sequence[JaticPyTorchObjectDetectionOutput], Optional[Any]],
            Sequence[dict[str, Any]],
        ],
    ) -> tuple[
        Sequence[NDArray[np.float32]],
        Union[Sequence[NDArray[np.float32]], Sequence[JaticPyTorchObjectDetectionOutput], Optional[Any]],
        Sequence[dict[str, Any]],
    ]:
        """Generate attack output for evasion attacks.

        Args:
            is_object_detection (bool): Object detection flag.
            y (Union[NDArray[np.float32], list[dict[str, Any]]]): Targets.
            adv_images (NDArray[np.float32]): Patched images.
            meta (list[dict[str, Any]]): Input metadata.
            attack_output (tuple[
                Sequence[NDArray[np.float32]],
                Union[Sequence[NDArray[np.float32]], Sequence[JaticPyTorchObjectDetectionOutput], Optional[Any]],
                Sequence[dict[str, Any]],
                ]): Initialized attack output.

        Returns:
            tuple[
                Sequence[NDArray[np.float32]],
                Union[Sequence[NDArray[np.float32]], Sequence[JaticPyTorchObjectDetectionOutput], Optional[Any]],
                Sequence[dict[str, Any]],
            ]: Attack output includes adversarial images, targets, metadata.
        """
        if is_object_detection:
            targets = [JaticPyTorchObjectDetectionOutput(detection) for detection in y] if y is not None else []
            attack_output = (list(adv_images), targets, meta)
        else:
            attack_output = (list(adv_images), y, meta)
        return attack_output

    def __output_not_evasion(
        self,
        is_object_detection: bool,
        y: Optional[Union[NDArray[np.float32], list[dict[str, Any]]]],
        adv_output: NDArray[np.float32],
        meta: list[dict[str, Any]],
        attack_output: tuple[
            Sequence[NDArray[np.float32]],
            Union[Sequence[NDArray[np.float32]], Sequence[JaticPyTorchObjectDetectionOutput], Optional[Any]],
            Sequence[dict[str, Any]],
        ],
    ) -> tuple[
        Sequence[NDArray[np.float32]],
        Union[Sequence[NDArray[np.float32]], Sequence[JaticPyTorchObjectDetectionOutput], Optional[Any]],
        Sequence[dict[str, Any]],
    ]:
        """Generate attack output for non-evasion attacks.

        Args:
            is_object_detection (bool): Object detection flag.
            y (Union[NDArray[np.float32], list[dict[str, Any]]]): Targets.
            adv_output (NDArray[np.float32]): Adversarial samples.
            meta (list[dict[str, Any]]): Input metadata.
            attack_output (tuple[
                Sequence[NDArray[np.float32]],
                Union[Sequence[NDArray[np.float32]], Sequence[JaticPyTorchObjectDetectionOutput], Optional[Any]],
                Sequence[dict[str, Any]],
                ]): Initialized attack output.

        Returns:
            tuple[
            Sequence[NDArray[np.float32]],
            Union[Sequence[NDArray[np.float32]], Sequence[JaticPyTorchObjectDetectionOutput], Optional[Any]],
            Sequence[dict[str, Any]],
            ]: Attack output includes adversarial images, targets, metadata.
        """
        # convert to JATIC supported data types
        if is_object_detection:
            targets = [JaticPyTorchObjectDetectionOutput(detection) for detection in y] if y is not None else []
            attack_output = (list(adv_output), targets, meta)
        else:
            attack_output = (list(adv_output), y, meta)
        return attack_output

    def __calc_diff_meta(
        self,
        attack_output: tuple[
            Sequence[NDArray[np.float32]],
            Union[Sequence[NDArray[np.float32]], Sequence[JaticPyTorchObjectDetectionOutput], Optional[Any]],
            Sequence[dict[str, Any]],
        ],
        x: NDArray[np.float32],
    ) -> tuple[
        Sequence[NDArray[np.float32]],
        Union[Sequence[NDArray[np.float32]], Sequence[JaticPyTorchObjectDetectionOutput], Optional[Any]],
        Sequence[dict[str, Any]],
    ]:
        """Check if should calculate the difference between benign and adversarial images to store in meta.

        Args:
            attack_output (tuple[
                Sequence[NDArray[np.float32]],
                Union[Sequence[NDArray[np.float32]], Sequence[JaticPyTorchObjectDetectionOutput], Optional[Any]],
                Sequence[dict[str, Any]],
                ]): Attack output includes adversarial images, targets, metadata.
            x (NDArray[np.float32]): Input images.

        Returns:
            tuple[
                Sequence[NDArray[np.float32]],
                Union[Sequence[NDArray[np.float32]], Sequence[JaticPyTorchObjectDetectionOutput], Optional[Any]],
                Sequence[dict[str, Any]],
                ]: Only metatdata is updated in attack output to reflect delta.
        """
        # check if should calculate the difference between benign
        # and adversarial images to store in meta
        if self._norm > 0:
            diff = np.linalg.norm((np.asarray(attack_output[0]) - x).reshape(len(x), -1), ord=self._norm, axis=1)
            meta = attack_output[2]
            if meta is not None and None not in meta and len(meta) > 0:
                for i, item in enumerate(meta):
                    item.update({"delta": diff[i]})
            else:
                meta = []
                for item in diff:
                    meta.append({"delta": item})
                attack_output = (attack_output[0], attack_output[1], meta)
        return attack_output
