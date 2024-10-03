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

from tests.utils import HEARTTestException
import pytest
import sys


logger = logging.getLogger(__name__)


def test_jatic_detr(heart_warning):
    try:
        from maite.protocols.object_detection import Model, ObjectDetectionTarget
        from heart_library.estimators.object_detection import JaticPyTorchObjectDetector
        from torchvision import transforms
        from PIL import Image
        import requests
        import numpy as np

        detector = JaticPyTorchObjectDetector(model_type="detr_resnet50",
                                              device_type='cpu',
                                            input_shape=(3, 800, 800),
                                            clip_values=(0, 1), 
                                            attack_losses=("loss_ce",
                                                "loss_bbox",
                                                "loss_giou",),)

        assert isinstance(detector, Model)
        
        NUMBER_CHANNELS = 3
        INPUT_SHAPE = (NUMBER_CHANNELS, 800, 800)

        transform = transforms.Compose([
                transforms.Resize(INPUT_SHAPE[1], interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(INPUT_SHAPE[1]),
                transforms.ToTensor()
            ])

        coco_images = []
        im = Image.open(requests.get("http://images.cocodataset.org/val2017/000000094852.jpg",
                                     stream=True).raw)
        im = transform(im).numpy()
        coco_images.append(im)
        coco_images = np.array(coco_images)
        
        detections = detector(coco_images)
        
        assert isinstance(detections[0], ObjectDetectionTarget)
        
        np.testing.assert_array_almost_equal(detections[0].scores[:3], np.array([ 0.00076229,   0.0027429,    0.085511], 'float32'), decimal=0.01)
        np.testing.assert_array_almost_equal(detections[0].labels[:3], np.array([22, 22, 22], dtype='float32'), decimal=0.01)
        np.testing.assert_array_almost_equal(detections[0].boxes[:3], np.array([[     105.07,       351.3,      175.56,       528.8],
                                                                                [     274.97,      387.28,      500.22,      652.64],
                                                                                [     147.21,      393.13,      421.43,         651]]), decimal=0.01)
        
        import torch
        loaded_model = torch.hub.load("facebookresearch/detr", "detr_resnet50", pretrained=True)
        detector = JaticPyTorchObjectDetector(loaded_model, "detr_resnet50", device_type='cpu',
                                            input_shape=(3, 800, 800),
                                            clip_values=(0, 1), 
                                            attack_losses=("loss_ce",
                                                "loss_bbox",
                                                "loss_giou",),)
        
    except HEARTTestException as e:
        heart_warning(e)
        

def test_jatic_faster_rcnn(heart_warning):
    try:
        from maite.protocols.object_detection import Model, ObjectDetectionTarget
        from heart_library.estimators.object_detection import JaticPyTorchObjectDetector
        from torchvision import transforms
        from PIL import Image
        import requests
        import numpy as np
        

        detector = JaticPyTorchObjectDetector(model_type="fasterrcnn_resnet50_fpn",
                                              device_type='cpu',
                                                input_shape=(3, 640, 640),
                                                clip_values=(0, 1), 
                                                attack_losses=("loss_classifier",
                                                                "loss_box_reg",
                                                                "loss_objectness",
                                                                "loss_rpn_box_reg",),)

        assert isinstance(detector, Model)
        
        NUMBER_CHANNELS = 3
        INPUT_SHAPE = (NUMBER_CHANNELS, 640, 640)

        transform = transforms.Compose([
                transforms.Resize(INPUT_SHAPE[1], interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(INPUT_SHAPE[1]),
                transforms.ToTensor()
            ])

        coco_images = []
        im = Image.open(requests.get("http://images.cocodataset.org/val2017/000000094852.jpg",
                                     stream=True).raw)
        im = transform(im).numpy()
        coco_images.append(im)
        coco_images = np.array(coco_images)
        
        detections = detector(coco_images)
        
        assert isinstance(detections[0], ObjectDetectionTarget)
        
        np.testing.assert_array_almost_equal(detections[0].scores[:3], np.array([0.9994759, 0.9088036, 0.4767685], 'float32'), decimal=0.01)
        np.testing.assert_array_almost_equal(detections[0].labels[:3], np.array([22, 22, 22], dtype='float32'), decimal=0.01)
        np.testing.assert_array_almost_equal(detections[0].boxes[:3], np.array([[ 92.3659,  144.71861, 475.3492,  519.8371 ],
                                                                                [194.51048, 311.40665, 334.29144, 505.1    ],
                                                                                [114.69924, 268.2348,  343.74084, 506.49698]]), decimal=0.01)
        
        from heart_library.estimators.object_detection.pytorch import fasterrcnn
        
        frcnn_detector = getattr(fasterrcnn, "fasterrcnn_resnet50_fpn")
        loaded_model = frcnn_detector(
            pretrained=True, progress=True, num_classes=91, pretrained_backbone=True
        )
        detector = JaticPyTorchObjectDetector(loaded_model, "fasterrcnn_resnet50_fpn", device_type='cpu',
                                                input_shape=(3, 640, 640),
                                                clip_values=(0, 1), 
                                                attack_losses=("loss_classifier",
                                                                "loss_box_reg",
                                                                "loss_objectness",
                                                                "loss_rpn_box_reg",))
        
    except HEARTTestException as e:
        heart_warning(e)
        
@pytest.mark.skipif(sys.version_info < (3, 10), reason="YOLOv5 requires python version 3.10 or above.")
def test_jatic_yolo(heart_warning):
    try:
        from maite.protocols.object_detection import Model, ObjectDetectionTarget
        from heart_library.estimators.object_detection import JaticPyTorchObjectDetector
        from torchvision import transforms
        from PIL import Image
        import requests
        import numpy as np

        detector = JaticPyTorchObjectDetector(model_type="yolov5s",
                                              device_type='cpu',
                                                input_shape=(3, 640, 640),
                                                clip_values=(0, 1),)

        assert isinstance(detector, Model)
        
        NUMBER_CHANNELS = 3
        INPUT_SHAPE = (NUMBER_CHANNELS, 640, 640)

        transform = transforms.Compose([
                transforms.Resize(INPUT_SHAPE[1], interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(INPUT_SHAPE[1]),
                transforms.ToTensor()
            ])

        coco_images = []
        im = Image.open(requests.get("http://images.cocodataset.org/val2017/000000094852.jpg",
                                     stream=True).raw)
        im = transform(im).numpy()
        coco_images.append(im)
        coco_images = np.array(coco_images)
        
        detections = detector(coco_images)
        
        assert isinstance(detections[0], ObjectDetectionTarget)
        
        np.testing.assert_array_almost_equal(detections[0].scores[:3], np.array([9.9953e-07,  8.4781e-08,  3.9297e-08], 'float32'), decimal=0.01)
        np.testing.assert_array_almost_equal(detections[0].labels[:3], np.array([14, 14, 14], dtype='float32'), decimal=0.01)
        np.testing.assert_array_almost_equal(detections[0].boxes[:3], np.array([[ 0, 0, 6.9441, 8.0846],
                                                                                [ 0.074183, 0, 22.789, 7.5572],
                                                                                [ 5.2672, 0, 31.972, 7.5443]]), decimal=0.01)
        
        from heart_library.estimators.object_detection.pytorch import Yolo, yolov5
        
        loaded_model = Yolo(yolov5.load(f"yolov5s.pt"), "cpu")
        detector = JaticPyTorchObjectDetector(loaded_model, "yolov5s", device_type='cpu',
                                                input_shape=(3, 640, 640),
                                                clip_values=(0, 1),)
        
        
    except HEARTTestException as e:
        heart_warning(e)


def test_supported_detectors(heart_warning):
    try:
        from heart_library.estimators.object_detection import (SUPPORTED_DETECTORS,
                                                               COCO_YOLO_LABELS,
                                                               COCO_FASTER_RCNN_LABELS,
                                                               COCO_DETR_LABELS)
        
        assert isinstance(SUPPORTED_DETECTORS, dict)
        assert len(list(SUPPORTED_DETECTORS.keys())) == 18
        assert isinstance(COCO_YOLO_LABELS, list)
        assert len(COCO_YOLO_LABELS) == 80
        assert isinstance(COCO_FASTER_RCNN_LABELS, list)
        assert len(COCO_FASTER_RCNN_LABELS) == 92
        assert isinstance(COCO_DETR_LABELS, list)
        assert len(COCO_DETR_LABELS) == 92
        
    except HEARTTestException as e:
        heart_warning(e)
