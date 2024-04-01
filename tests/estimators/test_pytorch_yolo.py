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
        from heart_library.estimators.object_detection import JaticPyTorchYolo
        
        coco_labels = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
        'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
        'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
        'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 
        'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 
        'teddy bear', 'hair drier', 'toothbrush']


        detector = JaticPyTorchYolo(device_type='cpu',
                            input_shape=(3, 640, 640),
                            clip_values=(0, 255), 
                            attack_losses=("loss_total", "loss_cls",
                                        "loss_box",
                                        "loss_obj"),
                            labels=coco_labels)

        assert isinstance(detector, ObjectDetector)
        assert isinstance(detector.metadata, ModelMetadata)

        import numpy as np
        
        (x_train, _), (_, _), _, _ = load_dataset('cifar10')

        img = x_train[[1]].transpose(0, 3, 1, 2).astype('float32')
        
        output = detector(img)
        
        assert isinstance(output, HasDetectionPredictions)
        
        np.testing.assert_array_almost_equal(output.scores[0][:5], np.array([ 0.00052563, 0.00295, 0.0022102, 0.00044677, 0.0016044], 'float32'), decimal=0.01)
        np.testing.assert_array_almost_equal(output.labels[0][:5], np.array([0,0,0,0,0], dtype='float32'), decimal=0.01)
        np.testing.assert_array_almost_equal(output.boxes[0][:5], np.array([[    0.48038,           0,      10.373,      14.822],
                                                                            [          0,           0,      30.509,       20.61],
                                                                            [     1.0478,           0,     32.493,      19.561],
                                                                            [     21.357,           0,      32.436,      13.564],
                                                                            [    0.52886,           0,      16.289,      29.805]]), decimal=0.01)
    except HEARTTestException as e:
        art_warning(e)
