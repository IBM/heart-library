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
This module implements a JATIC compatible ART PyTorchClassifier.
"""
from dataclasses import dataclass
from typing import Any, Sequence

from art.estimators.classification.pytorch import PyTorchClassifier
from maite.errors import InvalidArgument
from maite.protocols import HasLogits, SupportsArray

from heart_library.utils import process_inputs_for_art

META_NOT_SPECIFIED: str = "Not specified"


@dataclass
class JaticPytorchClassifierOutput:
    """
    Dataclass output of ART JATIC PyTorch Classifier
    """

    logits: SupportsArray


@dataclass
class HeartClassifierMetadata:
    """
    HEART metadata dataclass for PyTorch Classifier
    """

    model_name: str
    provider: str
    task: str = "Image Classification"


class JaticPyTorchClassifier(PyTorchClassifier):  # pylint: disable=R0901
    """
    JATIC compatible extension of ART core PyTorchClassifier
    """

    metadata: HeartClassifierMetadata

    def __init__(
        self,
        labels: Sequence[str],
        model_name: str = META_NOT_SPECIFIED,
        provider: str = META_NOT_SPECIFIED,
        **kwargs: Any
    ):
        super().__init__(**kwargs)
        if labels is None:
            raise InvalidArgument("No labels were provided.")
        self._labels: Sequence[str] = labels
        self.metadata = HeartClassifierMetadata(model_name, provider)

    def __call__(self, data: SupportsArray) -> HasLogits:

        # convert to ART supported type
        images, _ = process_inputs_for_art(data, self._device)

        # make prediction
        output = self.predict(images)

        # convert back to JATIC supported type
        return JaticPytorchClassifierOutput(output)

    def get_labels(self) -> Sequence[str]:
        """Get labels."""
        if self._labels is None:
            raise InvalidArgument("No labels were provided.")
        return self._labels
