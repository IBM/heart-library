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

import pytest
from art.utils import load_dataset

from tests.utils import HEARTTestError, get_cifar10_image_classifier_pt

logger = logging.getLogger(__name__)


@pytest.mark.required
def test_parallel_auto_attack(heart_warning):
    try:
        import numpy as np
        from art.attacks.evasion.auto_attack import AutoAttack
        from art.attacks.evasion.projected_gradient_descent.projected_gradient_descent_pytorch import (
            ProjectedGradientDescentPyTorch,
        )
        from psutil import cpu_count

        labels = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
        ptc = get_cifar10_image_classifier_pt(from_logits=True, is_jatic=False)
        parallel_pool_size = cpu_count(logical=False) - 1  # set parallel pool to max physical cpus - 1

        (x_train, y_train), (_, _), _, _ = load_dataset("cifar10")
        x_train = x_train[:10].transpose(0, 3, 1, 2).astype("float32")
        y_train = y_train[:10]

        attacks = []
        attacks.append(
            ProjectedGradientDescentPyTorch(
                estimator=ptc,
                norm=np.inf,
                eps=0.1,
                max_iter=10,
                targeted=False,
                batch_size=32,
                verbose=False,
            ),
        )
        jatic_attack_noparallel = AutoAttack(
            estimator=ptc,
            attacks=attacks,
            targeted=True,
        )

        jatic_attack_parallel = AutoAttack(
            estimator=ptc,
            attacks=attacks,
            targeted=True,
            parallel_pool_size=parallel_pool_size,
        )

        core_attack = AutoAttack(
            estimator=ptc,
            attacks=attacks,
            targeted=True,
        )

        # GitLab HEART Issue #319
        jatic_attack_parallel.estimator_orig._optimizer = None  # noqa SLF001

        no_parallel_adv = jatic_attack_noparallel.generate(x=x_train, y=y_train)
        parallel_adv = jatic_attack_parallel.generate(x=x_train, y=y_train)
        core_adv = core_attack.generate(x=x_train, y=y_train)

        no_parallel_adv_preds = [labels[np.argmax(ptc.predict(i))] for i in no_parallel_adv]
        parallel_adv_preds = [labels[np.argmax(ptc.predict(i))] for i in parallel_adv]
        core_adv_preds = [labels[np.argmax(ptc.predict(i))] for i in core_adv]

        assert core_adv_preds != parallel_adv_preds
        assert core_adv_preds == no_parallel_adv_preds

        expected_parallel_output = (
            "AutoAttack("
            "targeted=True, "
            f"parallel_pool_size={parallel_pool_size}, "
            "num_attacks=10"
            ")\n"
            "BestAttacks:\n"
            "image 1: ProjectedGradientDescentPyTorch("
            "norm=inf, "
            "eps=0.1, "
            "eps_step=0.1, "
            "targeted=True, "
            "num_random_init=0, "
            "batch_size=32, "
            "minimal=False, "
            "summary_writer=None, "
            "decay=None, "
            "max_iter=10, "
            "random_eps=False, "
            "verbose=False, "
            ")\n"
            "image 2: ProjectedGradientDescentPyTorch("
            "norm=inf, "
            "eps=0.1, "
            "eps_step=0.1, "
            "targeted=True, "
            "num_random_init=0, "
            "batch_size=32, "
            "minimal=False, "
            "summary_writer=None, "
            "decay=None, max_iter=10, "
            "random_eps=False, "
            "verbose=False, "
            ")\n"
            "image 3: ProjectedGradientDescentPyTorch("
            "norm=inf, "
            "eps=0.1, "
            "eps_step=0.1, "
            "targeted=True, "
            "num_random_init=0, "
            "batch_size=32, "
            "minimal=False, "
            "summary_writer=None, "
            "decay=None, "
            "max_iter=10, "
            "random_eps=False, "
            "verbose=False, "
            ")\n"
            "image 4: n/a\n"
            "image 5: n/a\n"
            "image 6: n/a\n"
            "image 7: n/a\n"
            "image 8: ProjectedGradientDescentPyTorch("
            "norm=inf, "
            "eps=0.1, "
            "eps_step=0.1, "
            "targeted=True, "
            "num_random_init=0, "
            "batch_size=32, "
            "minimal=False, "
            "summary_writer=None, "
            "decay=None, "
            "max_iter=10, "
            "random_eps=False, "
            "verbose=False, "
            ")\n"
            "image 9: ProjectedGradientDescentPyTorch("
            "norm=inf, "
            "eps=0.1, "
            "eps_step=0.1, "
            "targeted=True, "
            "num_random_init=0, "
            "batch_size=32, "
            "minimal=False, "
            "summary_writer=None, "
            "decay=None, "
            "max_iter=10, "
            "random_eps=False, "
            "verbose=False, "
            ")\n"
            "image 10: n/a"
        )

        assert repr(jatic_attack_parallel) == expected_parallel_output

    except HEARTTestError as e:
        heart_warning(e)
