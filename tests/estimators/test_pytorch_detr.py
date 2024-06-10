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

import logging

import pytest
from tests.utils import HEARTTestException
from art.utils import load_dataset


logger = logging.getLogger(__name__)


@pytest.mark.skip_framework("keras", "non_dl_frameworks", "mxnet", "kerastf", "tensorflow1", "tensorflow2v1")
def test_jatic_support_detection(art_warning):
    try:
        from maite.protocols import ObjectDetector, HasDetectionPredictions, ModelMetadata
        from heart_library.estimators.object_detection import JaticPyTorchDETR
        
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
                                "loss_giou",),
                            labels=coco_labels)

        assert isinstance(detector, ObjectDetector)
        assert isinstance(detector.metadata, ModelMetadata)

        import numpy as np
        
        (x_train, _), (_, _), _, _ = load_dataset('cifar10')

        img = x_train[[1]].transpose(0, 3, 1, 2).astype('float32')
        
        output = detector(img)
        
        assert isinstance(output, HasDetectionPredictions)
        
        np.testing.assert_array_almost_equal(output.scores[0][:5], np.array([ 0.166942,   0.16694196, 0.16694193, 0.16694197, 0.16694193], 'float32'), decimal=0.01)
        np.testing.assert_array_almost_equal(output.labels[0][:5], np.array([61,61,61,61,61], dtype='float32'), decimal=0.01)
        np.testing.assert_array_almost_equal(output.boxes[0][:5], np.array([[123.02651,  105.02686,  243.54053,  293.68375 ],
                                                                            [123.026566, 105.02692,  243.54054,  293.68378 ],
                                                                            [123.02651,  105.02684,  243.54053,  293.68378 ],
                                                                            [123.02647,  105.026886, 243.54053,  293.6838  ],
                                                                            [123.026566, 105.02694,  243.5406,   293.68378 ]]), decimal=0.01)
    except HEARTTestException as e:
        art_warning(e)
