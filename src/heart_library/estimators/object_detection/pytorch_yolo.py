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
This module implements a JATIC compatible ART PyTorchYolo.
"""
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch
import yolov5
from art.estimators.object_detection.pytorch_yolo import PyTorchYolo
from maite.errors import InvalidArgument
from maite.protocols import HasDetectionPredictions, SupportsArray
from yolov5.utils.loss import ComputeLoss

from heart_library.utils import process_inputs_for_art


@dataclass
class JaticPyTorchObjectDetectionOutput:
    """
    Dataclass output of HEART PyTorch Classifier
    """

    scores: SupportsArray
    labels: SupportsArray
    boxes: SupportsArray


@dataclass
class HeartObjectDetectionMetadata:
    """
    Metadata dataclass of HEART YOLO Object Detector
    """

    model_name: str
    provider: str
    task: str = "Object Detection"


class Yolo(torch.nn.Module):
    """
    Wrapper for YOLO object detection models
    """

    def __init__(self, model, device):
        super().__init__()
        self.model = model
        self.model.hyp = {
            "box": 0.05,
            "obj": 1.0,
            "cls": 0.5,
            "anchor_t": 4.0,
            "cls_pw": 1.0,
            "obj_pw": 1.0,
            "fl_gamma": 0.0,
        }
        self.model.model.model.to(device)  # Required when using GPU

        self.compute_loss = ComputeLoss(self.model.model.model)

    def forward(self, x, targets=None):
        """
        Modified forward to facilitate computation of loss dict
        """
        if self.training:
            outputs = self.model.model.model(x)
            loss, loss_items = self.compute_loss(outputs, targets)
            loss_components_dict = {"loss_total": loss}
            loss_components_dict["loss_box"] = loss_items[0]
            loss_components_dict["loss_obj"] = loss_items[1]
            loss_components_dict["loss_cls"] = loss_items[2]
            return loss_components_dict
        return self.model(x)


class JaticPyTorchYolo(PyTorchYolo):  # pylint: disable=R0901
    """
    JATIC compatible extension of ART core PyTorchYolo
    """

    metadata: HeartObjectDetectionMetadata

    def __init__(self, labels: Sequence[str], model_path: str = "yolov5s.pt", **kwargs: Any):
        model = None
        try:
            model = yolov5.load(model_path)
        except Exception as load_exception:
            raise Exception("Yolov5 model was not successful loaded.") from load_exception

        model = Yolo(model, kwargs["device_type"])
        super().__init__(model=model, **kwargs)

        if labels is None:
            raise InvalidArgument("No labels were provided.")
        self._labels: Sequence[str] = labels
        self.metadata = HeartObjectDetectionMetadata(model_name="YOLOv5", provider="yolov5")

    def __call__(self, data: SupportsArray) -> HasDetectionPredictions:

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
