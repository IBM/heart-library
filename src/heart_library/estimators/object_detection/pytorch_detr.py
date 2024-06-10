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
This module implements a JATIC compatible ART PyTorchDETR.
"""
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch
from art.estimators.object_detection.pytorch_detection_transformer import \
    PyTorchDetectionTransformer
from maite.errors import InvalidArgument
from maite.protocols import ArrayLike, HasDetectionPredictions

from heart_library.utils import process_inputs_for_art


@dataclass
class JaticPyTorchObjectDetectionOutput:
    """
    Dataclass output of HEART PyTorch Object detector
    """

    scores: ArrayLike
    labels: ArrayLike
    boxes: ArrayLike


@dataclass
class HeartObjectDetectionMetadata:
    """
    Metadata dataclass of HEART Object Detector
    """

    model_name: str
    provider: str
    task: str = "Object Detection"


class JaticPyTorchDETR(PyTorchDetectionTransformer):  # pylint: disable=R0901
    """
    JATIC compatible extension of ART core PyTorchDETR
    """

    metadata: HeartObjectDetectionMetadata

    def __init__(self, labels: Sequence[str], model_path: str = "detr_resnet50", **kwargs: Any):

        model = torch.hub.load("facebookresearch/detr", model_path, pretrained=True)
        super().__init__(model=model, **kwargs)

        if labels is None:
            raise InvalidArgument("No labels were provided.")
        self._labels: Sequence[str] = labels
        self.metadata = HeartObjectDetectionMetadata(model_name="detr_resnet50", provider="facebook")

    def __call__(self, data: ArrayLike) -> HasDetectionPredictions:

        # convert to ART supported type
        images, _ = process_inputs_for_art(data, self._device)

        # make prediction
        output = self.predict(images)

        # convert back to JATIC supported type
        return JaticPyTorchObjectDetectionOutput(
            scores=np.concatenate([np.expand_dims(det["scores"], 0) for det in output]),
            boxes=np.concatenate([np.expand_dims(det["boxes"], 0) for det in output]),
            labels=np.concatenate([np.expand_dims(det["labels"], 0) for det in output]),
        )

    def get_labels(self) -> Sequence[str]:
        """Get labels."""
        if self._labels is None:
            raise InvalidArgument("No labels were provided.")
        return self._labels
