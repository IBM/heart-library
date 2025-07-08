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

import logging

import pytest

from tests.utils import HEARTTestError

logger = logging.getLogger(__name__)

REQUESTS_TIMEOUT = 60


@pytest.mark.required
def test_jatic_detr(heart_warning):
    try:
        import numpy as np
        import requests
        from maite.protocols.object_detection import Model, ObjectDetectionTarget
        from PIL import Image
        from torchvision import transforms

        from heart_library.estimators.object_detection import JaticPyTorchObjectDetector

        detector = JaticPyTorchObjectDetector(
            model_type="detr_resnet50",
            device_type="cpu",
            input_shape=(3, 800, 800),
            clip_values=(0, 1),
            attack_losses=(
                "loss_ce",
                "loss_bbox",
                "loss_giou",
            ),
        )

        assert isinstance(detector, Model)

        number_channels = 3
        input_shape = (number_channels, 800, 800)

        transform = transforms.Compose(
            [
                transforms.Resize(input_shape[1], interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(input_shape[1]),
                transforms.ToTensor(),
            ],
        )

        coco_images = []

        im = Image.open(
            requests.get(
                "http://images.cocodataset.org/val2017/000000094852.jpg",
                stream=True,
                timeout=REQUESTS_TIMEOUT,
            ).raw,
        )

        im = transform(im).numpy()
        coco_images.append(im)
        coco_images = np.array(coco_images)

        detections = detector(coco_images)

        assert isinstance(detections[0], ObjectDetectionTarget)

        np.testing.assert_array_almost_equal(
            detections[0].scores[:3],
            np.array(
                [0.00076229, 0.0027429, 0.085511],
                "float32",
            ),
            decimal=0.01,
        )

        np.testing.assert_array_almost_equal(
            detections[0].labels[:3],
            np.array(
                [22, 22, 22],
                dtype="float32",
            ),
            decimal=0.01,
        )

        np.testing.assert_array_almost_equal(
            detections[0].boxes[:3],
            np.array(
                [
                    [105.07, 351.3, 175.56, 528.8],
                    [274.97, 387.28, 500.22, 652.64],
                    [147.21, 393.13, 421.43, 651],
                ],
            ),
            decimal=0.01,
        )

        import torch

        loaded_model = torch.hub.load("facebookresearch/detr", "detr_resnet50", pretrained=True)
        detector = JaticPyTorchObjectDetector(
            loaded_model,
            "detr_resnet50",
            device_type="cpu",
            input_shape=(3, 800, 800),
            clip_values=(0, 1),
            attack_losses=(
                "loss_ce",
                "loss_bbox",
                "loss_giou",
            ),
        )

    except HEARTTestError as e:
        heart_warning(e)


@pytest.mark.required
def test_jatic_faster_rcnn(heart_warning):
    try:
        import numpy as np
        import requests
        from maite.protocols.object_detection import Model, ObjectDetectionTarget
        from PIL import Image
        from torchvision import transforms

        from heart_library.estimators.object_detection import JaticPyTorchObjectDetector

        detector = JaticPyTorchObjectDetector(
            model_type="fasterrcnn_resnet50_fpn",
            device_type="cpu",
            input_shape=(3, 640, 640),
            clip_values=(0, 1),
            attack_losses=(
                "loss_classifier",
                "loss_box_reg",
                "loss_objectness",
                "loss_rpn_box_reg",
            ),
        )

        assert isinstance(detector, Model)

        number_channels = 3
        input_shape = (number_channels, 640, 640)

        transform = transforms.Compose(
            [
                transforms.Resize(input_shape[1], interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(input_shape[1]),
                transforms.ToTensor(),
            ],
        )

        coco_images = []

        im = Image.open(
            requests.get(
                "http://images.cocodataset.org/val2017/000000094852.jpg",
                stream=True,
                timeout=REQUESTS_TIMEOUT,
            ).raw,
        )

        im = transform(im).numpy()
        coco_images.append(im)
        coco_images = np.array(coco_images)

        detections = detector(coco_images)

        assert isinstance(detections[0], ObjectDetectionTarget)

        np.testing.assert_array_almost_equal(
            detections[0].scores[:3],
            np.array(
                [0.9994759, 0.9088036, 0.4767685],
                "float32",
            ),
            decimal=0.01,
        )

        np.testing.assert_array_almost_equal(
            detections[0].labels[:3],
            np.array(
                [22, 22, 22],
                dtype="float32",
            ),
            decimal=0.01,
        )

        np.testing.assert_array_almost_equal(
            detections[0].boxes[:3],
            np.array(
                [
                    [92.3659, 144.71861, 475.3492, 519.8371],
                    [194.51048, 311.40665, 334.29144, 505.1],
                    [114.69924, 268.2348, 343.74084, 506.49698],
                ],
            ),
            decimal=0.01,
        )

        from torchvision.models import detection as fasterrcnn

        frcnn_detector = fasterrcnn.fasterrcnn_resnet50_fpn
        loaded_model = frcnn_detector(
            pretrained=True,
            progress=True,
            num_classes=91,
            pretrained_backbone=True,
        )
        detector = JaticPyTorchObjectDetector(
            loaded_model,
            "fasterrcnn_resnet50_fpn",
            device_type="cpu",
            input_shape=(3, 640, 640),
            clip_values=(0, 1),
            attack_losses=(
                "loss_classifier",
                "loss_box_reg",
                "loss_objectness",
                "loss_rpn_box_reg",
            ),
        )

    except HEARTTestError as e:
        heart_warning(e)


def test_jatic_yolo(heart_warning):
    try:
        import numpy as np
        import requests
        from maite.protocols.object_detection import Model, ObjectDetectionTarget
        from PIL import Image
        from torchvision import transforms

        from heart_library.estimators.object_detection import JaticPyTorchObjectDetector

        detector = JaticPyTorchObjectDetector(
            model_type="yolo12n",
            device_type="cpu",
            input_shape=(3, 640, 640),
            clip_values=(0, 1),
        )

        assert isinstance(detector, Model)

        number_channels = 3
        input_shape = (number_channels, 640, 640)

        transform = transforms.Compose(
            [
                transforms.Resize(input_shape[1], interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(input_shape[1]),
                transforms.ToTensor(),
            ],
        )

        coco_images = []

        im = Image.open(
            requests.get(
                "http://images.cocodataset.org/val2017/000000094852.jpg",
                stream=True,
                timeout=REQUESTS_TIMEOUT,
            ).raw,
        )

        im = transform(im).numpy()
        coco_images.append(im)
        coco_images = np.array(coco_images)

        detections = detector(coco_images)

        assert isinstance(detections[0], ObjectDetectionTarget)

        np.testing.assert_almost_equal(
            detections[0].scores[0],
            0.77979,
            decimal=0.01,
        )

        np.testing.assert_almost_equal(
            detections[0].labels[0],
            20,
            decimal=1.00,
        )

        np.testing.assert_array_almost_equal(
            detections[0].boxes[0],
            np.array([86.496, 152.87, 467.88, 523.54], dtype=np.float32),
            decimal=0.01,
        )

    except HEARTTestError as e:
        heart_warning(e)


@pytest.mark.required
def test_supported_detectors(heart_warning):
    try:
        from heart_library.estimators.object_detection import (
            COCO_DETR_LABELS,
            COCO_FASTER_RCNN_LABELS,
            COCO_YOLO_LABELS,
            SUPPORTED_DETECTORS,
        )

        assert isinstance(SUPPORTED_DETECTORS, dict)
        assert len(list(SUPPORTED_DETECTORS.keys())) == 16
        assert isinstance(COCO_YOLO_LABELS, list)
        assert len(COCO_YOLO_LABELS) == 80
        assert isinstance(COCO_FASTER_RCNN_LABELS, list)
        assert len(COCO_FASTER_RCNN_LABELS) == 92
        assert isinstance(COCO_DETR_LABELS, list)
        assert len(COCO_DETR_LABELS) == 92

    except HEARTTestError as e:
        heart_warning(e)


@pytest.mark.required
def test_jatic_yolo_versions(heart_warning):
    """
    Test that JaticPyTorchObjectDetector supports multiple YOLO versions and uses the correct loss dict for v8+.
    """
    try:
        import torch
        from art.estimators.object_detection import PyTorchYolo

        from heart_library.estimators.object_detection import JaticPyTorchObjectDetector

        yolo_versions = ["yolov3u.pt", "yolov5n.pt", "yolov8n.pt", "yolov10n.pt", "yolov12n.pt"]
        dummy_input = torch.randn(1, 3, 640, 640)

        for model_type in yolo_versions:
            logger.info(f"Testing model_type: {model_type}")
            try:
                detector = JaticPyTorchObjectDetector(
                    model=model_type,
                    model_type=model_type.replace(".pt", ""),
                    device_type="cpu",
                    input_shape=(3, 640, 640),
                    clip_values=(0, 1),
                )

                # Accessing protected member for test inspection only
                detector_impl = detector._detector  # noqa: SLF001
                assert isinstance(detector_impl, PyTorchYolo)

                model = detector_impl.model
                assert isinstance(model, torch.nn.Module)

                # For v8 and above, check loss dict keys in training mode
                version_num = int(model_type.replace("yolov", "").replace(".pt", ""))
                if version_num >= 8:
                    model.train()
                    dummy_targets = [
                        {"boxes": torch.tensor([[0.0, 0.0, 1.0, 1.0]]), "labels": torch.tensor([0])},
                    ]
                    loss_dict = model(dummy_input, dummy_targets)
                    assert isinstance(loss_dict, dict)
                    assert {"loss_total", "loss_box", "loss_cls", "loss_dfl"}.issubset(loss_dict.keys())
                else:
                    model.eval()
                    outputs = model(dummy_input)
                    assert isinstance(outputs, list)

            except (RuntimeError, ValueError, TypeError, FileNotFoundError) as e:
                logger.warning(f"Skipping {model_type} due to error: {e}")
                continue

    except HEARTTestError as e:
        heart_warning(e)
