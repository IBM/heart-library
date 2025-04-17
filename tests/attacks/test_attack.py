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
from art.utils import load_dataset

from tests.utils import HEARTTestError, get_cifar10_image_classifier_pt

logger = logging.getLogger(__name__)

REQUESTS_TIMEOUT = 60


@pytest.mark.required
def test_jatic_supported_attack(heart_warning):
    try:
        import maite.protocols.image_classification as ic
        from art.attacks.evasion import ProjectedGradientDescentPyTorch

        from heart_library.attacks.attack import JaticAttack

        jptc = get_cifar10_image_classifier_pt()
        pgd_attack = ProjectedGradientDescentPyTorch(estimator=jptc, targeted=True)
        attack = JaticAttack(pgd_attack)

        assert isinstance(jptc, ic.Model)
        assert isinstance(attack, ic.Augmentation)

        import numpy as np

        (x_train, _), (_, _), _, _ = load_dataset("cifar10")

        img = x_train[[0]].transpose(0, 3, 1, 2).astype("float32")

        data = {"images": img[:1], "labels": [4]}

        x_adv, _, _ = attack(data=data)

        assert np.argmax(jptc(x_adv[0])) == 4
        assert np.argmax(jptc(img)) != np.argmax(jptc(x_adv[0]))

    except HEARTTestError as e:
        heart_warning(e)


@pytest.mark.required
def test_jatic_supported_black_box_attack(heart_warning):
    try:
        import maite.protocols.image_classification as ic
        from art.attacks.evasion import HopSkipJump

        from heart_library.attacks.attack import JaticAttack

        jptc = get_cifar10_image_classifier_pt()
        pgd_attack = HopSkipJump(classifier=jptc)
        attack = JaticAttack(pgd_attack)

        assert isinstance(jptc, ic.Model)
        assert isinstance(attack, ic.Augmentation)

        import numpy as np

        (x_train, _), (_, _), _, _ = load_dataset("cifar10")

        img = x_train[[0]].transpose(0, 3, 1, 2).astype("float32")

        data = {"images": img[:1], "labels": [4]}

        x_adv, _, _ = attack(data=data)

        assert np.argmax(jptc(x_adv[0])) == 3
        assert np.argmax(jptc(img)) != np.argmax(jptc(x_adv[0]))

    except HEARTTestError as e:
        heart_warning(e)


@pytest.mark.required
def test_jatic_supported_patch_attack(heart_warning):
    try:
        import maite.protocols.image_classification as ic
        from art.attacks.evasion import AdversarialPatchPyTorch

        from heart_library.attacks.attack import JaticAttack

        jptc = get_cifar10_image_classifier_pt()

        batch_size = 16
        scale_min = 0.3
        scale_max = 1.0
        rotation_max = 0
        learning_rate = 5000.0
        max_iter = 2000
        patch_shape = (3, 14, 14)
        patch_location = (18, 18)

        patch_attack = AdversarialPatchPyTorch(
            estimator=jptc,
            rotation_max=rotation_max,
            patch_location=patch_location,
            scale_min=scale_min,
            scale_max=scale_max,
            patch_type="square",
            learning_rate=learning_rate,
            max_iter=max_iter,
            batch_size=batch_size,
            patch_shape=patch_shape,
            verbose=False,
            targeted=True,
        )
        attack = JaticAttack(patch_attack)

        assert isinstance(jptc, ic.Model)
        assert isinstance(attack, ic.Augmentation)

        import numpy as np

        (x_train, _), (_, _), _, _ = load_dataset("cifar10")

        img = x_train[[0]].transpose(0, 3, 1, 2).astype("float32")

        data = {"images": img[:1], "labels": [3], "metadata": [{}]}

        x_adv, _, metadata = attack(data=data)

        patch = metadata[0]["patch"]

        assert patch.shape == patch_shape
        assert np.argmax(jptc(x_adv[0])) == 3

    except HEARTTestError as e:
        heart_warning(e)


@pytest.mark.required
def test_jatic_supported_obj_det_patch_attack(heart_warning):
    try:
        import requests
        from art.attacks.evasion import ProjectedGradientDescent
        from maite.protocols.object_detection import Model, ObjectDetectionTarget
        from PIL import Image
        from torchvision import transforms

        from heart_library.attacks.attack import JaticAttack
        from heart_library.estimators.object_detection import JaticPyTorchObjectDetector

        detector = JaticPyTorchObjectDetector(
            model_type="detr_resnet50",
            device_type="cpu",
            input_shape=(3, 800, 800),
            clip_values=(0, 1),
            attack_losses=("loss_ce", "loss_bbox", "loss_giou"),
        )

        evasion_attack = ProjectedGradientDescent(estimator=detector, max_iter=2)
        attack = JaticAttack(evasion_attack)

        assert isinstance(detector, Model)

        import numpy as np

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

        attack_output, _, _ = attack(coco_images)

        detections = detector(coco_images)
        adv_output = detector(attack_output)

        isinstance(adv_output, ObjectDetectionTarget)

        np.testing.assert_raises(
            AssertionError,
            np.testing.assert_array_equal,
            adv_output[0].boxes,
            detections[0].boxes,
        )

        np.testing.assert_raises(
            AssertionError,
            np.testing.assert_array_equal,
            adv_output[0].scores,
            detections[0].scores,
        )

        np.testing.assert_raises(
            AssertionError,
            np.testing.assert_array_equal,
            adv_output[0].labels,
            detections[0].labels,
        )

    except HEARTTestError as e:
        heart_warning(e)
