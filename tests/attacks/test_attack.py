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
from tests.utils import HEARTTestException, get_cifar10_image_classifier_pt
from art.utils import load_dataset


logger = logging.getLogger(__name__)


@pytest.mark.skip_framework("keras", "non_dl_frameworks", "mxnet", "kerastf", "tensorflow1", "tensorflow2v1")
def test_jatic_supported_attack(art_warning):
    try:
        from art.attacks.evasion import ProjectedGradientDescentPyTorch
        from heart_library.attacks.attack import JaticAttack
        from heart_library.attacks.typing import Attack, HasEvasionAttackResult
        from maite.protocols import ImageClassifier

        jptc = get_cifar10_image_classifier_pt()
        pgd_attack = ProjectedGradientDescentPyTorch(estimator=jptc, targeted=True)
        attack = JaticAttack(pgd_attack)

        assert isinstance(jptc, ImageClassifier)
        assert isinstance(attack, Attack)

        import numpy as np

        (x_train, _), (_, _), _, _ = load_dataset('cifar10')

        img = x_train[[0]].transpose(0, 3, 1, 2).astype('float32')

        data = {'image': img[0], 'label': 4}

        x_adv = attack.run_attack(data=data)
        assert isinstance(x_adv, HasEvasionAttackResult)
        x_adv = x_adv.adversarial_examples

        assert np.argmax(jptc(x_adv[0]).logits) == 4
        assert np.argmax(jptc(img).logits) != np.argmax(jptc(x_adv[0]).logits)

    except HEARTTestException as e:
        art_warning(e)
        
@pytest.mark.skip_framework("keras", "non_dl_frameworks", "mxnet", "kerastf", "tensorflow1", "tensorflow2v1")
def test_jatic_supported_black_box_attack(art_warning):
    try:
        from art.attacks.evasion import HopSkipJump
        from heart_library.attacks.attack import JaticAttack
        from heart_library.attacks.typing import Attack, HasEvasionAttackResult
        from maite.protocols import ImageClassifier

        jptc = get_cifar10_image_classifier_pt()
        pgd_attack = HopSkipJump(classifier=jptc)
        attack = JaticAttack(pgd_attack)

        assert isinstance(jptc, ImageClassifier)
        assert isinstance(attack, Attack)

        import numpy as np

        (x_train, _), (_, _), _, _ = load_dataset('cifar10')

        img = x_train[[0]].transpose(0, 3, 1, 2).astype('float32')

        data = {'image': img[0], 'label': 4}

        x_adv = attack.run_attack(data=data)
        assert isinstance(x_adv, HasEvasionAttackResult)
        x_adv = x_adv.adversarial_examples

        assert np.argmax(jptc(x_adv[0]).logits) == 3
        assert np.argmax(jptc(img).logits) != np.argmax(jptc(x_adv[0]).logits)

    except HEARTTestException as e:
        art_warning(e)
        
def test_jatic_supported_patch_attack(art_warning):
    try:
        from art.attacks.evasion import AdversarialPatchPyTorch
        from heart_library.attacks.attack import JaticAttack
        from heart_library.attacks.typing import Attack, HasEvasionAttackResult
        from maite.protocols import ImageClassifier

        jptc = get_cifar10_image_classifier_pt()
        
        batch_size = 16
        scale_min = 0.3
        scale_max = 1.0
        rotation_max = 0
        learning_rate = 5000.
        max_iter = 2000
        patch_shape = (3, 14, 14)
        patch_location = (18,18)
        
        patch_attack = AdversarialPatchPyTorch(estimator=jptc, rotation_max=rotation_max, patch_location=patch_location,
                            scale_min=scale_min, scale_max=scale_max, patch_type='square',
                            learning_rate=learning_rate, max_iter=max_iter, batch_size=batch_size,
                            patch_shape=patch_shape, verbose=False, targeted=True)
        attack = JaticAttack(patch_attack)

        assert isinstance(jptc, ImageClassifier)
        assert isinstance(attack, Attack)

        import numpy as np

        (x_train, _), (_, _), _, _ = load_dataset('cifar10')

        img = x_train[[0]].transpose(0, 3, 1, 2).astype('float32')

        data = {'image': img[0], 'label': 3}

        attack_output = attack.run_attack(data=data)
        assert isinstance(attack_output, HasEvasionAttackResult)
        x_adv = attack_output.adversarial_examples
        patch, _ = attack_output.adversarial_patch

        assert patch.shape == patch_shape
        assert np.argmax(jptc(x_adv[0]).logits) == 3

    except HEARTTestException as e:
        art_warning(e)

def test_jatic_supported_obj_det_patch_attack(art_warning):
    try:
        from art.attacks.evasion import AdversarialPatchPyTorch
        from heart_library.attacks.attack import JaticAttack
        from heart_library.estimators.object_detection import JaticPyTorchYolo
        from heart_library.attacks.typing import Attack, HasEvasionAttackResult
        from maite.protocols import ObjectDetector
        from torchvision import transforms
        from PIL import Image
        import requests

        labels = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
        detector = JaticPyTorchYolo(device_type='cpu',
                            input_shape=(3, 32, 32),
                            clip_values=(0, 1), 
                            attack_losses=("loss_total", "loss_cls",
                                        "loss_box",
                                        "loss_obj"),
                            labels=labels)
        
        rotation_max=0.0
        scale_min=0.5
        scale_max=1.0
        distortion_scale_max=0.0
        learning_rate=1.99
        max_iter=5
        batch_size=16
        patch_shape=(3, 300, 300)
        patch_location=(100,100)
        patch_type="circle"
        optimizer="Adam"

        patch_attack = AdversarialPatchPyTorch(detector, rotation_max=rotation_max, 
                            scale_min=scale_min, scale_max=scale_max, optimizer=optimizer, distortion_scale_max=distortion_scale_max,
                            learning_rate=learning_rate, max_iter=max_iter, batch_size=batch_size, patch_location=patch_location,
                            patch_shape=patch_shape, patch_type=patch_type, verbose=True, targeted=True)
        attack = JaticAttack(patch_attack)
            
        assert isinstance(detector, ObjectDetector)
        assert isinstance(attack, Attack)

        import numpy as np

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
        
        output = detector(coco_images)

        dets = [{'boxes': output.boxes[i],
            'scores': output.scores[i],
            'labels': output.labels[i]} for i in range(len(coco_images))]

        data = {'image': coco_images[[0]], 'label': dets[-1:]}
        attack_output = attack.run_attack(data=data)
        
        adv_output = detector(attack_output.adversarial_examples)
        isinstance(adv_output, HasEvasionAttackResult)
        
        np.testing.assert_array_almost_equal(adv_output.boxes[0][:3], np.array([[0, 0, 8.3302, 23.068],
                                                                                [0, 0, 28.769, 22.031],
                                                                                [0.53017, 0, 30.525, 26.07]], dtype='float32'), 0.001)
        np.testing.assert_array_almost_equal(adv_output.scores[0][:3], np.array([0.0045606, 1.0218e-05, 1.558e-05], dtype='float32'), 0.001)
        np.testing.assert_array_almost_equal(adv_output.labels[0][:3], np.array([39, 9, 9], dtype='float32'), 0.001)

    except HEARTTestException as e:
        art_warning(e)
