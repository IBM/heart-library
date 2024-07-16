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
from tests.utils import HEARTTestException, get_cifar10_image_classifier_pt
from art.utils import load_dataset


logger = logging.getLogger(__name__)


def test_laser_attack(heart_warning):
    try:
        from heart_library.attacks.evasion import HeartLaserBeamAttack
        from heart_library.attacks.attack import JaticAttack
        import maite.protocols.image_classification as ic
        import numpy as np

        jptc = get_cifar10_image_classifier_pt()
        
        laser_attack = HeartLaserBeamAttack(estimator=jptc, iterations=5, max_laser_beam=(580, 3.14, 32, 32))
        attack = JaticAttack(laser_attack)

        assert isinstance(jptc, ic.Model)
        assert isinstance(attack, ic.Augmentation)

        import numpy as np

        (x_train, _), (_, _), _, _ = load_dataset('cifar10')

        img = x_train[[0]].transpose(0, 3, 1, 2).astype('float32')

        data = {'images': img[:1], 'labels': [3]}

        x_adv, _, _ = attack(data=data)

        np.testing.assert_raises(AssertionError, np.testing.assert_array_equal, x_adv[[0]], img[[0]])
        np.testing.assert_raises(AssertionError, np.testing.assert_array_equal, jptc(x_adv[[0]]), jptc(img[[0]]))

    except HEARTTestException as e:
        heart_warning(e)
