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
import sys
from typing import Any, Dict, Sequence

import numpy as np
import torch
from art.estimators.object_detection.pytorch_detection_transformer import \
    PyTorchDetectionTransformer
from maite.protocols import ArrayLike

from heart_library.utils import process_inputs_for_art


class JaticPyTorchObjectDetectionOutput:
    """
    Object Detection Output
    """

    def __init__(self, detection: Dict[str, np.ndarray]):
        """
        param: Dict[str, np.ndarray] - detection data
        """
        self._boxes = detection["boxes"]
        self._labels = detection["labels"]
        self._scores = detection["scores"]

    @property
    def boxes(self) -> np.ndarray:
        """
        Return detection bounding boxes
        """
        return self._boxes

    @boxes.setter
    def boxes(self, value):
        """
        Update detection bounding boxes
        """
        self._boxes = value

    @property
    def labels(self) -> np.ndarray:
        """
        Return detection labels
        """
        return self._labels

    @labels.setter
    def labels(self, value):
        """
        Update detection labels
        """
        self._labels = value

    @property
    def scores(self) -> np.ndarray:
        """
        Return detection scores
        """
        return self._scores

    @scores.setter
    def scores(self, value):
        """
        Update detection scores
        """
        self._scores = value


class JaticPyTorchDETR(PyTorchDetectionTransformer):  # pylint: disable=R0901
    """
    JATIC compatible extension of ART core PyTorchDETR
    """

    def __init__(self, model_path: str = "detr_resnet50", **kwargs: Any):

        # Issue: https://github.com/pytorch/hub/issues/243
        if "models" in sys.modules:
            sys.modules.pop("models")
        model = torch.hub.load("facebookresearch/detr", model_path, pretrained=True)

        super().__init__(model=model, **kwargs)

    def __call__(self, data: Sequence[ArrayLike]) -> Sequence[JaticPyTorchObjectDetectionOutput]:

        # convert to ART supported type
        images, _, _ = process_inputs_for_art(data)

        # make prediction
        output = self.predict(images)

        # convert back to JATIC supported type
        return [JaticPyTorchObjectDetectionOutput(detection) for detection in output]
