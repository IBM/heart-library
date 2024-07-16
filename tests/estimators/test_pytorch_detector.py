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


logger = logging.getLogger(__name__)


def test_jatic_support_detection(heart_warning):
    try:
        from maite.protocols.object_detection import Model, ObjectDetectionTarget
        from heart_library.estimators.object_detection import JaticPyTorchDETR
        from torchvision import transforms
        from PIL import Image
        import requests
        import numpy as np
        
        coco_labels = [
            'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
            'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
            'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
            'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack',
            'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
            'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
            'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
            'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
            'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
            'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
            'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
            'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
            'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
            'toothbrush'
        ]


        detector = JaticPyTorchDETR(device_type='cpu',
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
        
        print('boxes', detections[0].boxes[:3])
        print('labels', detections[0].labels[:3])
        print('scores', detections[0].scores[:3])
        
        np.testing.assert_array_almost_equal(detections[0].scores[:3], np.array([ 0.00076229,   0.0027429,    0.085511], 'float32'), decimal=0.01)
        np.testing.assert_array_almost_equal(detections[0].labels[:3], np.array([22, 22, 22], dtype='float32'), decimal=0.01)
        np.testing.assert_array_almost_equal(detections[0].boxes[:3], np.array([[     105.07,       351.3,      175.56,       528.8],
                                                                                [     274.97,      387.28,      500.22,      652.64],
                                                                                [     147.21,      393.13,      421.43,         651]]), decimal=0.01)
        
    except HEARTTestException as e:
        heart_warning(e)
        

def test_jatic_faster_rcnn(heart_warning):
    try:
        from maite.protocols.object_detection import Model, ObjectDetectionTarget
        from heart_library.estimators.object_detection import JaticPyTorchFasterRCNN
        from torchvision import transforms
        from PIL import Image
        import requests
        import numpy as np
        

        detector = JaticPyTorchFasterRCNN(device_type='cpu',
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
        
    except HEARTTestException as e:
        heart_warning(e)
