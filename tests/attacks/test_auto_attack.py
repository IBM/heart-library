# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (HEART) Authors 2024
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

from tests.utils import HEARTTestException, get_cifar10_image_classifier_pt
from art.utils import load_dataset


logger = logging.getLogger(__name__)


def test_parallel_auto_attack(heart_warning):
    try:
        from art.attacks.evasion.auto_attack import AutoAttack
        from art.attacks.evasion.projected_gradient_descent.projected_gradient_descent_pytorch import (
            ProjectedGradientDescentPyTorch,
        )
        import numpy as np

        labels = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
        ptc = get_cifar10_image_classifier_pt(from_logits=True, is_jatic=False)

        (x_train, y_train), (_, _), _, _ = load_dataset("cifar10")
        x_train = x_train[:10].transpose(0, 3, 1, 2).astype("float32")
        y_train = y_train[:10]

        attacks = []
        attacks.append(
            ProjectedGradientDescentPyTorch(
                estimator=ptc, norm=np.inf, eps=0.1, max_iter=10, targeted=False, batch_size=32, verbose=False
            )
        )
        jatic_attack_noparallel = AutoAttack(estimator=ptc, attacks=attacks, targeted=True, parallel=False)
        jatic_attack_parallel = AutoAttack(estimator=ptc, attacks=attacks, targeted=True, parallel=True)
        core_attack = AutoAttack(estimator=ptc, attacks=attacks, targeted=True)

        no_parallel_adv = jatic_attack_noparallel.generate(x=x_train, y=y_train)
        parallel_adv = jatic_attack_parallel.generate(x=x_train, y=y_train)
        core_adv = core_attack.generate(x=x_train, y=y_train)

        no_parallel_adv_preds = [labels[np.argmax(ptc.predict(i))] for i in no_parallel_adv]
        parallel_adv_preds = [labels[np.argmax(ptc.predict(i))] for i in parallel_adv]
        core_adv_preds = [labels[np.argmax(ptc.predict(i))] for i in core_adv]

        assert core_adv_preds != parallel_adv_preds
        assert core_adv_preds == no_parallel_adv_preds

        assert (
            repr(jatic_attack_parallel)
            == "AutoAttack(targeted=True, parallel=True, num_attacks=10)\nBestAttacks:\nimage 1: "
            + "ProjectedGradientDescentPyTorch(norm=inf, eps=0.1, eps_step=0.1, targeted=True, num_random_init=0,"
            + " batch_size=32, minimal=False, summary_writer=None, decay=None, max_iter=10, random_eps=False, verbose=False, )\nimage"
            + " 2: ProjectedGradientDescentPyTorch(norm=inf, eps=0.1, eps_step=0.1, targeted=True, num_random_init=0,"
            + " batch_size=32, minimal=False, summary_writer=None, decay=None, max_iter=10, random_eps=False, verbose=False, )\nimage"
            + " 3: ProjectedGradientDescentPyTorch(norm=inf, eps=0.1, eps_step=0.1, targeted=True, num_random_init=0, "
            + "batch_size=32, minimal=False, summary_writer=None, decay=None, max_iter=10, random_eps=False, verbose=False, )\nimage 4"
            + ": n/a\nimage 5: n/a\nimage 6: n/a\nimage 7: n/a\nimage 8: ProjectedGradientDescentPyTorch(norm=inf, eps=0.1,"
            + " eps_step=0.1, targeted=True, num_random_init=0, batch_size=32, minimal=False, summary_writer=None, decay=None, max_iter=10,"
            + " random_eps=False, verbose=False, )\nimage 9: ProjectedGradientDescentPyTorch(norm=inf, eps=0.1, eps_step=0.1,"
            + " targeted=True, num_random_init=0, batch_size=32, minimal=False, summary_writer=None, decay=None, max_iter=10,"
            + " random_eps=False, verbose=False, )\nimage 10: n/a"
        )

    except HEARTTestException as e:
        heart_warning(e)
