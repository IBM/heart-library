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
"""
This module implements a JATIC compatible ART Attack.
"""
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from art.attacks import Attack, EvasionAttack
from maite.protocols import ArrayLike

from heart_library.estimators.object_detection.pytorch_detr import \
    JaticPyTorchObjectDetectionOutput
from heart_library.utils import process_inputs_for_art


class JaticEvasionAttackOutput:
    """
    Dataclass output JaticEvasionAttackOutput
    """

    def __init__(self, images: List[np.ndarray], targets: np.ndarray, metadata: List[Dict[str, Any]]):
        self.images = images
        self.targets = targets
        self.metadata = metadata

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, ind: int) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        return self.images[ind], self.targets[ind], self.metadata[ind]


@dataclass
class JaticPoisonAttackOutput:
    """
    Dataclass output JaticEvasionAttackOutput
    """

    poisoning_examples: ArrayLike
    poisoning_labels: ArrayLike


class JaticAttack:  # pylint: disable=R0901
    """
    Wrapper for JATIC compatible attacks
    """

    def __init__(self, attack: Attack, norm: int = 0):
        self._attack = attack
        self._norm = norm

    def __call__(
        self, data: Union[Tuple[ArrayLike, ArrayLike, Sequence[dict[str, Any]]], ArrayLike], **kwargs
    ) -> Tuple[
        ArrayLike,
        Union[ArrayLike, Sequence[JaticPyTorchObjectDetectionOutput], Optional[Any]],
        Sequence[dict[str, Any]],
    ]:
        """
        Run the attack.. depending on the attack, call different func.. depending on the attack, different output
        .. metrics?
        .. include an argument which would add "benign predictions" to the metadata.. to be used in metrics
        .. where both benign and adversarial images needed.. also store the original image in metadata?
        """

        attack_output: Tuple[
            ArrayLike,
            Union[ArrayLike, Sequence[JaticPyTorchObjectDetectionOutput], Optional[Any]],
            Sequence[dict[str, Any]],
        ]

        # convert to ART supported data types
        # assume data is a batch of data and returns a modified version of that batch
        x, y, meta = process_inputs_for_art(data)
        is_object_detection = False
        if y is not None and not isinstance(y, np.ndarray):
            # check if y is a list of dicts, as per object detection
            if not all(isinstance(i, dict) for i in y):
                y = np.array(y)
            else:
                is_object_detection = True

        # run the attack
        if isinstance(self._attack, EvasionAttack):

            adv_output = self._attack.generate(x, y, **kwargs)

            # check if adversarial patch attack
            # requries extra step of applying the patch
            if hasattr(self._attack, "apply_patch"):
                from art.attacks.evasion import (  # pylint: disable=C0415
                    DPatch, RobustDPatch)

                if isinstance(self._attack, (RobustDPatch, DPatch)):
                    patch = adv_output
                    mask = np.array([])
                    adv_images = self._attack.apply_patch(x=x)
                else:
                    patch, mask = adv_output
                    adv_images = self._attack.apply_patch(x=x, scale=1)

                if meta is not None and None not in meta and len(meta) > 0:
                    for item in meta:
                        item.update({"patch": patch, "mask": mask})
                else:
                    meta = []
                    for _ in range(len(adv_images)):
                        meta.append({"patch": patch, "mask": mask})

                if is_object_detection:
                    if y is not None:
                        targets = [JaticPyTorchObjectDetectionOutput(detection) for detection in y]
                    else:
                        targets = []
                    attack_output = (np.asarray(list(adv_images)), targets, meta)
                else:
                    attack_output = np.asarray(list(adv_images)), y, meta

            # not a patch attack
            else:
                # convert to JATIC supported data types
                if is_object_detection:
                    if y is not None:
                        targets = [JaticPyTorchObjectDetectionOutput(detection) for detection in y]
                    else:
                        targets = []
                    attack_output = (np.asarray(list(adv_output)), targets, meta)
                else:
                    attack_output = np.asarray(list(adv_output)), y, meta

            # check if should calculate the difference between benign
            # and adversarial images to store in meta
            if self._norm > 0:
                diff = np.linalg.norm((attack_output[0] - x).reshape(len(x), -1), ord=self._norm, axis=1)
                meta = attack_output[2]
                if meta is not None and None not in meta and len(meta) > 0:
                    for i, item in enumerate(meta):
                        item.update({"delta": diff[i]})
                else:
                    meta = []
                    for item in diff:
                        meta.append({"delta": item})
                    attack_output = (attack_output[0], attack_output[1], meta)

        if attack_output is None:
            raise ValueError("Attack error, output is None.")

        return attack_output
