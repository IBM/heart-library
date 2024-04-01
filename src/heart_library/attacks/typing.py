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
ART attack Protocols
"""
from typing import Protocol, Union, runtime_checkable

from maite.protocols import SupportsArray


@runtime_checkable
class HasEvasionAttackResult(Protocol):
    """
    Scores are predictions for either an image or detection box.

    Attributes
    ----------
    adversarial_examples : SupportsArray
        Adversarial sample computed for an evasion attack.

    Examples
    --------

    """

    adversarial_examples: SupportsArray


@runtime_checkable
class HasPoisonAttackResult(Protocol):
    """
    Scores are predictions for either an image or detection box.

    Attributes
    ----------
    poisoning_examples : SupportsArray
        Poisoned samples computed for a poisoning attack.

    poisoning_labels : SupportsArray
        Poisoned labels computed for a poisoning attack.

    Examples
    --------

    """

    poisoning_examples: SupportsArray
    poisoning_labels: SupportsArray


@runtime_checkable
class Attack(Protocol):
    """
    An attack that takes in images and optionally labels and returns attack result e.g. adversarial image

    Methods
    -------
    run_attack(data: SupportsArray, ) -> Union[HasLogits, HasProbs, HasScores]
        Run inference on the data.

    Examples
    --------

    """

    def run_attack(self, data: SupportsArray, **kwargs) -> Union[HasEvasionAttackResult, HasPoisonAttackResult]:
        """Returns the labels for the model."""
        ...
