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
"""This module implements a JATIC compatible ART Object Detector."""

import sys
import uuid
from collections.abc import Sequence
from typing import Any, Optional, Union

import numpy as np
import torch
from art.estimators.object_detection import (
    PyTorchDetectionTransformer,
    PyTorchFasterRCNN,
    PyTorchObjectDetector,
    PyTorchYolo,
)
from maite.protocols import ArrayLike
from numpy.typing import NDArray
from torchvision.models import detection as fasterrcnn

from heart_library.utils import process_inputs_for_art

if sys.version_info >= (3, 10):
    import yolov5
    from yolov5.utils.loss import ComputeLoss

    class Yolo(torch.nn.Module):
        """Wrapper for YOLO object detection models

        Args:
            torch (torch.nn.Module): YOLO object detection models.
        """

        def __init__(self, model: Any, device: str) -> None:  # noqa ANN401
            """Yolo initialization.

            Args:
                model (Any): Object detection model.
                device (str): The desired device of the parameters and buffers in this module.
            """
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

            self._compute_loss: Any = ComputeLoss(self.model.model.model)

        def forward(
            self,
            x: torch.Tensor,
            targets: Optional[torch.Tensor] = None,
        ) -> Union[torch.Tensor, dict[str, torch.Tensor]]:
            """Modified forward to facilitate computation of loss dict

            Args:
                x (torch.Tensor): Input tensor.
                targets (Optional[torch.Tensor], optional): Target values. Defaults to None.

            Returns:
                Union[torch.Tensor, dict[str, torch.Tensor]]: Output tensor(s) produced by the network.
            """
            if self.training:
                outputs = self.model.model.model(x)
                loss, loss_items = self._compute_loss(outputs, targets)
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
SUPPORTED_DETECTORS: dict[str, str] = {
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
    """Object Detection Output"""

    def __init__(self, detection: dict[str, NDArray[np.float32]]) -> None:
        """JaticPyTorchObjectDetectionOutput initialization.
        Args:
            detection (dict[str, NDArray[np.float32]]): Detection data."""
        self._boxes = detection["boxes"]
        self._labels = detection["labels"]
        self._scores = detection["scores"]

    @property
    def boxes(self) -> NDArray[np.float32]:
        """Return detection bounding boxes

        Returns:
            NDArray[np.float32]:  The boxes in [y1, x1, y2, x2] format, with 0 <= x1 < x2 <= W and 0 <= y1 < y2 <= H.
        """
        return self._boxes

    @boxes.setter
    def boxes(self, value: NDArray[np.float32]) -> None:
        """Update detection bounding boxes

        Args:
            value (NDArray[np.float32]):
                The boxes in [y1, x1, y2, x2] format, with 0 <= x1 < x2 <= W and 0 <= y1 < y2 <= H.
        """
        self._boxes = value

    @property
    def labels(self) -> NDArray[np.float32]:
        """Return detection labels

        Returns:
            NDArray[np.float32]: The labels for each image.
        """
        return self._labels

    @labels.setter
    def labels(self, value: NDArray[np.float32]) -> None:
        """Update detection labels

        Args:
            value (NDArray[np.float32]): The labels for each image.
        """
        self._labels = value

    @property
    def scores(self) -> NDArray[np.float32]:
        """Return detection scores

        Returns:
            NDArray[np.float32]: The scores or each prediction.
        """
        return self._scores

    @scores.setter
    def scores(self, value: NDArray[np.float32]) -> None:
        """Update detection scores

        Args:
            value (NDArray[np.float32]): The scores or each prediction.
        """
        self._scores = value


class JaticPyTorchObjectDetector(PyTorchObjectDetector):
    """JATIC compatible extension of ART core PyTorchObjectDetector

    Args:
        PyTorchObjectDetector (PyTorchObjectDetector): ART core PyTorchObjectDetector.

    Examples
    --------

    We can create a JaticPyTorchObjectDetector and pass in sample data for detection:

    >>> from heart_library.estimators.object_detection import JaticPyTorchObjectDetector
    >>> import torch
    >>> import numpy
    >>> from datasets import load_dataset
    >>> from torchvision.transforms import transforms

    Define the JaticPyTorchObjectDetector, in this case passing in a resnet model:

    >>> MEAN = [0.485, 0.456, 0.406]
    >>> STD = [0.229, 0.224, 0.225]
    >>> preprocessing = (MEAN, STD)

    >>> detector = JaticPyTorchObjectDetector(
    ...     model_type="detr_resnet50_dc5",
    ...     input_shape=(3, 800, 800),
    ...     clip_values=(0, 1),
    ...     attack_losses=(
    ...         "loss_ce",
    ...         "loss_bbox",
    ...         "loss_giou",
    ...     ),
    ...     device_type="cpu",
    ...     optimizer=torch.nn.CrossEntropyLoss(),
    ...     preprocessing=preprocessing,
    ... )

    Prepare images for detection:

    >>> data = load_dataset("guydada/quickstart-coco", split="train[20:25]")
    >>> preprocess = transforms.Compose([transforms.Resize(800), transforms.CenterCrop(800), transforms.ToTensor()])

    >>> data = data.map(lambda x: {"image": preprocess(x["image"]), "label": None})

    Execute object detection and return JaticPyTorchObjectDetectionOutput:

    >>> detections = detector(data)
    """

    metadata: dict[str, Any]

    def __init__(
        self,
        model: Union["torch.nn.Module", str] = "",
        model_type: str = "",
        metadata_id: Optional[str] = None,
        **kwargs: Any,  # noqa ANN401
    ) -> None:
        """JaticPyTorchObjectDetector initialization.

        Args:
            model (Union[torch.nn.Module, str], optional): a loaded model or path to model. Defaults to "".
            model_type (str, optional): one of supported_detectors e.g. yolov5. Defaults to "".

        Raises:
            ValueError: if model type is not one of "yolo", "detr", or "fastercnn".
            ValueError: if model type is custom specified.
            Exception: if Yolov5 model was not successfully loaded.
            ValueError: if python version < 3.10
            ValueError: if model type is not one of "yolo", "detr", or "fastercnn".
        """
        self.metadata = {"id": metadata_id if metadata_id is not None else str(uuid.uuid4())}
        self.model_type = model_type

        if "models" in sys.modules:
            sys.modules.pop("models")

        if isinstance(model, torch.nn.Module):
            super().__init__(model=model, **kwargs)
            # Define object detection framework for torch.nn.module based on provided model_type.
            self.__handle_torch_nn(model_type, model, **kwargs)

        elif isinstance(model, str):
            device: Any = None
            # Initialize device type and handle missing model type.
            device = self.__handle_device_model_type(device, model_type, kwargs)

            loaded_model: Any = None
            # Load model frameworks based on str input.
            self.__handle_model_str_inputs(model, model_type, device, loaded_model, **kwargs)

    def __handle_model_str_inputs(
        self,
        model: str,
        model_type: str,
        device: Any,  # noqa ANN401
        loaded_model: Any,  # noqa ANN401
        **kwargs: Any,  # noqa ANN401
    ) -> None:
        """Load model frameworks based on str input.

        Args:
            model (str): A loaded model or path to model.
            model_type (str): Pytorch framework.
            device (Any): Torch device type to be used.
            loaded_model (Any): Empty initialization to be wrapped.

        Raises:
            ValueError: If model type is not one of yolo, detr, fasterrcnn.
        """
        # YOLO
        if "yolo" in model_type:
            self.__handle_yolo_model(model, model_type, device, loaded_model, **kwargs)
        # DETR
        elif "detr" in model_type:
            self.__handle_detr_model(model, model_type, device, loaded_model, **kwargs)

        # Faster-RCNN
        elif "fasterrcnn" in model_type:
            self.__handle_fasterrcnn_model(model, model_type, device, loaded_model, **kwargs)
        else:
            raise ValueError(f"Model type {model_type} is not supported. Try one of {SUPPORTED_DETECTORS}.")

    def __handle_torch_nn(self, model_type: str, model: torch.nn.Module, **kwargs: Any) -> None:  # noqa ANN401
        """Define object detection framework for torch.nn.module based on provided model_type.

        Args:
            model_type (str): Pytorch framework.
            model (torch.nn.module): A loaded model or path to model.

        Raises:
            ValueError: If model type is not one of yolo, detr, or fastercnn.
        """
        if "yolo" in model_type:
            self._detector = PyTorchYolo(model, **kwargs)
        elif "detr" in model_type:
            self._detector = PyTorchDetectionTransformer(model, **kwargs)
        else:
            self._detector = PyTorchObjectDetector(model, **kwargs)

    def __handle_device_model_type(
        self,
        device: Any,  # noqa ANN401
        model_type: str,
        kwargs: Any,  # noqa ANN401
    ) -> Union[torch.device, str]:
        """Initialize device type and handle missing model type.

        Args:
            device (Any): Empty initialization.
            model_type (str): Pytorch framework.

        Raises:
            ValueError: If model type is not specified.

        Returns:
            Union[torch.device, str]: Torch device type to be used.
        """
        if "device_type" in kwargs:
            device = kwargs["device_type"]
        else:
            if not torch.cuda.is_available():
                device = torch.device("cpu")
            else:  # pragma: no cover
                cuda_idx = torch.cuda.current_device()
                device = torch.device(f"cuda:{cuda_idx}")

        if model_type == "":
            raise ValueError(
                f"To use a local model, please specify param: type, \
                    with one of the supported models: {SUPPORTED_DETECTORS}",
            )
        return device

    def __handle_yolo_model(
        self,
        model: str,
        model_type: str,
        device: str,
        loaded_model: Any,  # noqa ANN401
        **kwargs: Any,  # noqa ANN401
    ) -> None:
        """Load yolo model.

        Args:
            model (str): A loaded model or path to model.
            model_type (str): Pytorch framework.
            device (str): Torch device type to be used.
            loaded_model (Any): Empty initialization to be wrapped.

        Raises:
            Exception: If Yolov5 model was not successfully loaded.
            ValueError: If python version < 3.10.
        """
        if sys.version_info >= (3, 10):
            try:
                if model == "":
                    loaded_model = Yolo(yolov5.load(f"{model_type}.pt"), device)
                else:  # pragma: no cover
                    loaded_model = Yolo(yolov5.load(model), device)
            except (NotImplementedError, FileNotFoundError) as e:
                raise Exception("Yolov5 model was not successfully loaded.") from e
            self._detector = PyTorchYolo(loaded_model, **kwargs)
        else:
            raise ValueError("yolov5 models require python versions 3.10 and above.")

    def __handle_detr_model(
        self,
        model: str,
        model_type: str,
        device: Union[torch.device, str],
        loaded_model: Any,  # noqa ANN401
        **kwargs: Any,  # noqa ANN401
    ) -> None:
        """Load detr model.

        Args:
            model (str): A loaded model or path to model.
            model_type (str): Pytorch framework.
            device (Union[torch.device, str]): Torch device type to be used.
            loaded_model (Any): Empty initialization to be wrapped.
        """
        if model == "":
            loaded_model = torch.hub.load("facebookresearch/detr", model_type, pretrained=True)
        else:  # pragma: no cover
            checkpoint = torch.load(model, map_location=device)
            loaded_model = torch.hub.load("facebookresearch/detr", model_type, pretrained=False)
            loaded_model.load_state_dict(checkpoint["model"])
        self._detector = PyTorchDetectionTransformer(loaded_model, **kwargs)

    def __handle_fasterrcnn_model(
        self,
        model: str,
        model_type: str,
        device: Union[torch.device, str],
        loaded_model: Any,  # noqa ANN401
        **kwargs: Any,  # noqa ANN401
    ) -> None:
        """Load Faster-RCNN model.

        Args:
            model (str): A loaded model or path to model.
            model_type (str): Pytorch framework.
            device (Union[torch.device, str]): Torch device type to be used.
            loaded_model (Any): Empty initialization to be wrapped.
        """
        if model == "":
            frcnn_detector = getattr(fasterrcnn, model_type)
            loaded_model = frcnn_detector(
                pretrained=True,
                progress=True,
                num_classes=91,
                pretrained_backbone=True,
            )
        else:  # pragma: no cover
            checkpoint = torch.load(model, map_location=device)
            n_classes = checkpoint["roi_heads.box_predictor.cls_score.weight"].shape[0]
            frcnn_detector = getattr(fasterrcnn, model_type)
            loaded_model = frcnn_detector(
                pretrained=False,
                progress=True,
                num_classes=n_classes,
                pretrained_backbone=True,
            )
            loaded_model.load_state_dict(checkpoint)
        self._detector = PyTorchFasterRCNN(loaded_model, **kwargs)

    def __getattr__(self, attr: str) -> Any:  # noqa ANN401
        """Return value of detector attribute.

        Args:
            attr (str): string that contain's detector's name.

        Returns:
            Any: Model framework.
        """
        return getattr(self._detector, attr)

    def __call__(self, data: Sequence[ArrayLike]) -> Sequence[JaticPyTorchObjectDetectionOutput]:
        """Convert JATIC supported data to ART supported data and perform prediction.

        Args:
            data (Any): JATIC supported data.

        Returns:
            Sequence[JaticPyTorchObjectDetectionOutput]: Predictions in JATIC supported type.
        """
        # convert to ART supported type
        images, _, _ = process_inputs_for_art(data)

        # make prediction
        output = self._detector.predict(images)

        # convert back to JATIC supported type
        return [JaticPyTorchObjectDetectionOutput(detection) for detection in output]

    def _translate_labels(self, labels: list[dict[str, "torch.Tensor"]]) -> Any:  # noqa ANN401
        """Route to method of instantiated detector.

        Args:
            labels (List[dict[str, "torch.Tensor"]]): Object detection labels in format x1y1x2y2 (torchvision).

        Returns:
            Any: Object detection labels in format x1y1x2y2 (torchvision).
        """
        return self._detector._translate_labels(labels)  # noqa SLF001

    def _translate_predictions(self, predictions: Any) -> list[dict[str, np.ndarray]]:  # noqa ANN401
        """Route to method of instantiated detector.

        Args:
            predictions (Any): Object detection predictions in format x1y1x2y2 (torchvision).

        Returns:
            List[dict[str, np.ndarray]]: Object detection predictions in format x1y1x2y2 (torchvision).
        """
        return self._detector._translate_predictions(predictions)  # noqa SLF001
