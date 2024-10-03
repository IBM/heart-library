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
This module implements a JATIC compatible ART Object Detector.
"""
import sys
from typing import Any, Dict, List, Sequence, Union

import numpy as np
import torch
from art.estimators.object_detection import (PyTorchDetectionTransformer,
                                             PyTorchFasterRCNN,
                                             PyTorchObjectDetector,
                                             PyTorchYolo)
from maite.protocols import ArrayLike
from torchvision.models import detection as fasterrcnn

from heart_library.utils import process_inputs_for_art

if sys.version_info >= (3, 10):
    import yolov5
    from yolov5.utils.loss import ComputeLoss

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


COCO_YOLO_LABELS = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]
COCO_FASTER_RCNN_LABELS = COCO_DETR_LABELS = [
    "background",
    "person",
    "bicycle",
    "car",
    "motorbike",
    "aeroplane",
    "bus",
    "train",
    "truck",
    "boat",
    "trafficlight",
    "firehydrant",
    "streetsign",
    "stopsign",
    "parkingmeter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "hat",
    "backpack",
    "umbrella",
    "shoe",
    "eyeglasses",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sportsball",
    "kite",
    "baseballbat",
    "baseballglove",
    "skateboard",
    "surfboard",
    "tennisracket",
    "bottle",
    "plate",
    "wineglass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hotdog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "sofa",
    "pottedplant",
    "bed",
    "mirror",
    "diningtable",
    "window",
    "desk",
    "toilet",
    "door",
    "tvmonitor",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cellphone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "blender",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddybear",
    "hairdrier",
    "toothbrush",
    "hairbrush",
]
SUPPORTED_DETECTORS: dict = {
    "yolov5s": "YOLOv5 model. Ref: https://github.com/ultralytics/yolov5",
    "yolov5n": "YOLOv5 model. Ref: https://github.com/ultralytics/yolov5",
    "yolov5m": "YOLOv5 model. Ref: https://github.com/ultralytics/yolov5",
    "yolov5l": "YOLOv5 model. Ref: https://github.com/ultralytics/yolov5",
    "yolov5x": "YOLOv5 model. Ref: https://github.com/ultralytics/yolov5",
    "yolov5n6": "YOLOv5 model. Ref: https://github.com/ultralytics/yolov5",
    "yolov5s6": "YOLOv5 model. Ref: https://github.com/ultralytics/yolov5",
    "yolov5m6": "YOLOv5 model. Ref: https://github.com/ultralytics/yolov5",
    "yolov5l6": "YOLOv5 model. Ref: https://github.com/ultralytics/yolov5",
    "yolov5x6": "YOLOv5 model. Ref: https://github.com/ultralytics/yolov5",
    "fasterrcnn_resnet50_fpn": "Faster R-CNN model. Ref: \
https://pytorch.org/vision/master/models/generated/torchvision.models\
.detection.fasterrcnn_resnet50_fpn.html#\
torchvision.models.detection.fasterrcnn_resnet50_fpn",
    "fasterrcnn_resnet50_fpn_v2": "Faster R-CNN model. Ref: \
https://pytorch.org/vision/master/models/generated/torchvision.models.\
detection.fasterrcnn_resnet50_fpn_v2.html#\
torchvision.models.detection.fasterrcnn_resnet50_fpn_v2",
    "fasterrcnn_mobilenet_v3_large_fpn": "Faster R-CNN model. Ref: \
https://pytorch.org/vision/master/models/generated/torchvision.models.\
detection.fasterrcnn_mobilenet_v3_large_fpn.html#\
torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn",
    "fasterrcnn_mobilenet_v3_large_320_fpn": "Faster R-CNN model. Ref: \
https://pytorch.org/vision/master/models/generated/torchvision.models\
.detection.fasterrcnn_mobilenet_v3_large_320_fpn.html#\
torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn",
    "detr_resnet50": "Detection Transformer. Ref: https://github.com/facebookresearch/detr",
    "detr_resnet101": "Detection Transformer. Ref: https://github.com/facebookresearch/detr",
    "detr_resnet50_dc5": "Detection Transformer. Ref: https://github.com/facebookresearch/detr",
    "detr_resnet101_dc5": "Detection Transformer. Ref: https://github.com/facebookresearch/detr",
}


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


class JaticPyTorchObjectDetector(PyTorchObjectDetector):  # pylint: disable=R0901
    """
    JATIC compatible extension of ART core PyTorchObjectDetector
    """

    def __init__(self, model: Union["torch.nn.Module", str] = "", model_type: str = "", **kwargs: Any):
        """
        :param model: Union[torch.nn.Module, str] - a loaded model or path to model
        :param type: str - one of supported_detectors e.g. yolov5.
        """
        self.model_type = model_type

        if "models" in sys.modules:
            sys.modules.pop("models")

        if isinstance(model, torch.nn.Module):
            super().__init__(model=model, **kwargs)
            if "yolo" in model_type:
                self._detector = PyTorchYolo(model, **kwargs)
            elif "detr" in model_type:
                self._detector = PyTorchDetectionTransformer(model, **kwargs)
            elif "fasterrcnn" in model_type:
                self._detector = PyTorchFasterRCNN(model, **kwargs)
            else:
                raise ValueError(f"Model type {model_type} is not supported. Try one of {SUPPORTED_DETECTORS}.")

        elif isinstance(model, str):

            if "device_type" in kwargs:
                device_type = kwargs["device_type"]
            else:
                if not torch.cuda.is_available():
                    device_type = torch.device("cpu")
                else:  # pragma: no cover
                    cuda_idx = torch.cuda.current_device()
                    device_type = torch.device(f"cuda:{cuda_idx}")

            if model_type == "":
                raise ValueError(
                    f"To use a local model, please specify param: type, \
                        with one of the supported models: {SUPPORTED_DETECTORS}"
                )

            # YOLO
            if "yolo" in model_type:
                if sys.version_info >= (3, 10):
                    try:
                        if model == "":
                            loaded_model = Yolo(yolov5.load(f"{model_type}.pt"), device_type)
                        else:  # pragma: no cover
                            loaded_model = Yolo(yolov5.load(model), device_type)
                    except Exception as load_exception:
                        raise Exception("Yolov5 model was not successful loaded.") from load_exception
                    self._detector = PyTorchYolo(loaded_model, **kwargs)
                else:
                    raise ValueError("yolov5 models require python versions 3.10 and above.")

            # DETR
            elif "detr" in model_type:
                if model == "":
                    loaded_model = torch.hub.load("facebookresearch/detr", model_type, pretrained=True)
                else:  # pragma: no cover
                    checkpoint = torch.load(model, map_location=device_type)
                    loaded_model = torch.hub.load("facebookresearch/detr", model_type, pretrained=False)
                    loaded_model.load_state_dict(checkpoint["model"])
                self._detector = PyTorchDetectionTransformer(loaded_model, **kwargs)

            # Faster-RCNN
            elif "fasterrcnn" in model_type:
                if model == "":
                    frcnn_detector = getattr(fasterrcnn, model_type)
                    loaded_model = frcnn_detector(
                        pretrained=True, progress=True, num_classes=91, pretrained_backbone=True
                    )
                else:  # pragma: no cover
                    checkpoint = torch.load(model, map_location=device_type)
                    n_classes = checkpoint["roi_heads.box_predictor.cls_score.weight"].shape[0]
                    frcnn_detector = getattr(fasterrcnn, model_type)
                    loaded_model = frcnn_detector(
                        pretrained=False, progress=True, num_classes=n_classes, pretrained_backbone=True
                    )
                    loaded_model.load_state_dict(checkpoint)
                self._detector = PyTorchFasterRCNN(loaded_model, **kwargs)
            else:
                raise ValueError(f"Model type {model_type} is not supported. Try one of {SUPPORTED_DETECTORS}.")

    def __getattr__(self, attr):
        return getattr(self._detector, attr)

    def __call__(self, data: ArrayLike) -> Sequence[JaticPyTorchObjectDetectionOutput]:
        # convert to ART supported type
        images, _, _ = process_inputs_for_art(data)

        # make prediction
        output = self._detector.predict(images)

        # convert back to JATIC supported type
        return [JaticPyTorchObjectDetectionOutput(detection) for detection in output]

    def _translate_labels(self, labels: List[Dict[str, "torch.Tensor"]]) -> Any:
        """Route to method of instantiated detector"""
        return self._detector._translate_labels(labels)  # pylint: disable=W0212

    def _translate_predictions(self, predictions: Any) -> List[Dict[str, np.ndarray]]:
        """Route to method of instantiated detector"""
        return self._detector._translate_predictions(predictions)  # pylint: disable=W0212
