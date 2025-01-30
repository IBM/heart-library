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

from tests.utils import HEARTTestException

logger = logging.getLogger(__name__)


@pytest.mark.required
def test_jatic_supported_obj_det_bb_patch_attack(heart_warning):
    try:
        from heart_library.estimators.object_detection import JaticPyTorchObjectDetector
        from maite.protocols.object_detection import Augmentation
        from art.attacks.evasion import RobustDPatch
        from heart_library.attacks.attack import JaticAttack
        from torchvision import transforms
        from PIL import Image
        import requests
        import numpy as np
        from copy import deepcopy
        from typing import Dict, Tuple, Any
        
        detector = detector = JaticPyTorchObjectDetector(model_type="fasterrcnn_resnet50_fpn",
                                              device_type='cpu',
                                                input_shape=(3, 640, 640),
                                                clip_values=(0, 255),
                                                attack_losses=("loss_classifier",
                                                                "loss_box_reg",
                                                                "loss_objectness",
                                                                "loss_rpn_box_reg",))

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
        coco_images = np.array(coco_images)*255
        detections = detector(coco_images)

        class TargetedImageDataset:
            def __init__(self, images, groundtruth, target_label, threshold=0.5):
                self.images = images
                self.groundtruth = groundtruth
                self.target_label = target_label
                self.threshold = threshold
                
            def __len__(self)->int:
                return len(self.images)
            
            def __getitem__(self, ind: int) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
                image = self.images[ind]
                targeted_detection = self.groundtruth[ind]
                targeted_detection.boxes = targeted_detection.boxes[targeted_detection.scores>self.threshold]
                targeted_detection.scores = np.asarray([1.0]*len(targeted_detection.boxes))
                targeted_detection.labels = [self.target_label]*len(targeted_detection.boxes)
                return (image, targeted_detection, {})
            
        targeted_data = TargetedImageDataset(coco_images, deepcopy(detections), 14, 0.8)

        evasion_attack = RobustDPatch(
            detector,
            patch_shape=(3, 300, 300),
            patch_location=(0, 0),
            crop_range=[0,0],
            brightness_range=[1.0, 1.0],
            rotation_weights=[1, 0, 0, 0],
            sample_size=1,
            learning_rate=1.99,
            max_iter=5,
            verbose=True,
            targeted=True
        )
        attack = JaticAttack(evasion_attack, norm=2)
        
        assert isinstance(attack, Augmentation)

        adv_images, _, _ = attack(targeted_data)
        
        adv_detections = detector(adv_images)
        
        np.testing.assert_raises(AssertionError, np.testing.assert_array_equal, adv_images[0][[0]], coco_images[[0]])
        np.testing.assert_raises(AssertionError, np.testing.assert_array_equal, adv_detections[0].boxes, detections[0].boxes)
        np.testing.assert_raises(AssertionError, np.testing.assert_array_equal, adv_detections[0].scores, detections[0].scores)
        np.testing.assert_raises(AssertionError, np.testing.assert_array_equal, adv_detections[0].labels, detections[0].labels)

    except HEARTTestException as e:
        heart_warning(e)
