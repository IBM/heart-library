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
from art.utils import load_dataset

from tests.utils import HEARTTestError, get_cifar10_image_classifier_pt

logger = logging.getLogger(__name__)


@pytest.mark.required
def test_laser_attack(heart_warning):
    try:
        import maite.protocols.image_classification as ic
        import numpy as np

        from heart_library.attacks.attack import JaticAttack
        from heart_library.attacks.evasion import HeartLaserBeamAttack
        from heart_library.utils import process_inputs_for_art

        jptc = get_cifar10_image_classifier_pt()

        laser_attack = HeartLaserBeamAttack(
            estimator=jptc,
            iterations=5,
            random_initializations=10,
            max_laser_beam=(580, 3.14, 32, 32),
        )
        attack = JaticAttack(laser_attack)

        assert isinstance(jptc, ic.Model)
        assert isinstance(attack, ic.Augmentation)

        (x_train, _), (_, _), _, _ = load_dataset("cifar10")

        img = x_train[[0]].transpose(0, 3, 1, 2).astype("float32")

        data = {"images": img[:1], "labels": [3]}

        x_adv, _, _ = attack(data=data)
        x_adv = np.stack(x_adv)

        np.testing.assert_array_equal(x_adv[[0]].shape, img[[0]].shape)

        x, y, _ = process_inputs_for_art(data)
        x_adv = laser_attack.generate(x, y)
        assert isinstance(x_adv, np.ndarray)

        with pytest.raises(ValueError, match="Unrecognized input dimension. Only tensors NHWC are acceptable."):
            laser_attack.generate(x.ravel())

    except HEARTTestError as e:
        heart_warning(e)
