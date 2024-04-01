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
This module implements a JATIC compatible ART Attack.
"""
from dataclasses import dataclass
from typing import Union

import numpy as np
from art.attacks import Attack, EvasionAttack, PoisoningAttack
from maite.protocols import SupportsArray

from heart_library.utils import process_inputs_for_art


@dataclass
class JaticEvasionAttackOutput:
    """
    Dataclass output JaticEvasionAttackOutput
    """

    adversarial_examples: SupportsArray
    adversarial_patch: SupportsArray


@dataclass
class JaticPoisonAttackOutput:
    """
    Dataclass output JaticEvasionAttackOutput
    """

    poisoning_examples: SupportsArray
    poisoning_labels: SupportsArray


class JaticAttack:  # pylint: disable=R0901
    """
    Wrapper for JATIC compatible attacks
    """

    def __init__(self, attack: Attack):
        self.attack = attack

    def run_attack(self, data: SupportsArray, **kwargs) -> Union[JaticEvasionAttackOutput, JaticPoisonAttackOutput]:
        """
        Run the attack.. depending on the attack, call different func.. depending on the attack, different output
        .. metrics?
        """

        # convert to ART supported data types
        x, y = process_inputs_for_art(data, self.attack.estimator.device)

        if y is not None and not isinstance(y, np.ndarray):
            # check if y is a list of dicts, as per object detection
            if not all(isinstance(i, dict) for i in y):
                y = np.array(y)

        # run the attack
        if isinstance(self.attack, EvasionAttack):
            results = self.attack.generate(x, y, **kwargs)

            if hasattr(self.attack, "apply_patch"):
                patch = results
                adv_images = self.attack.apply_patch(x=x, scale=0.4)
                return JaticEvasionAttackOutput(adversarial_examples=adv_images, adversarial_patch=patch)
            # convert to JATIC supported data types
            return JaticEvasionAttackOutput(adversarial_examples=results, adversarial_patch=np.array([]))

        if isinstance(self.attack, PoisoningAttack):
            results = self.attack.poison(x, y, **kwargs)
            # convert to JATIC supported data types
            return JaticPoisonAttackOutput(**results)

        raise Exception("Invalid attack.")
