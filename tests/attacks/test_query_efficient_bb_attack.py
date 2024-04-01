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


@pytest.mark.skip_framework("keras", "non_dl_frameworks", "mxnet", "kerastf", "tensorflow1", "tensorflow2v1")
def test_query_efficient_bb_attack(art_warning):
    try:
        from heart_library.attacks.attack import JaticAttack
        from heart_library.attacks.evasion import HeartQueryEfficientBlackBoxAttack
        from heart_library.attacks.typing import Attack, HasEvasionAttackResult
        from maite.protocols import ImageClassifier
        import numpy as np

        ptc = get_cifar10_image_classifier_pt(from_logits=True)

        query_attack = HeartQueryEfficientBlackBoxAttack(estimator=ptc, eps=0.2)
        attack = JaticAttack(query_attack)

        assert isinstance(ptc, ImageClassifier)
        assert isinstance(attack, Attack)

        (x_train, _), (_, _), _, _ = load_dataset('cifar10')

        img = x_train[[0]].transpose(0, 3, 1, 2).astype('float32')

        data = {'image': img[0], 'label': 3}

        attack_output = attack.run_attack(data=data)
        assert isinstance(attack_output, HasEvasionAttackResult)
        x_adv = attack_output.adversarial_examples

        assert np.argmax(ptc(x_adv[0]).logits) == 6

    except HEARTTestException as e:
        art_warning(e)
