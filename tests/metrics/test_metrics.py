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
def test_accuracy_perturbation_metric(art_warning):
    try:
        from art.attacks.evasion import ProjectedGradientDescentPyTorch
        from heart_library.attacks.attack import JaticAttack
        from heart_library.attacks.typing import Attack, HasEvasionAttackResult
        from heart_library.metrics.metrics import AccuracyPerturbationMetric
        from maite.protocols import ImageClassifier, Metric
        import numpy as np

        jptc = get_cifar10_image_classifier_pt()
        pgd_attack = ProjectedGradientDescentPyTorch(estimator=jptc)
        attack = JaticAttack(pgd_attack)

        (x_train, y_train), (_, _), _, _ = load_dataset("cifar10")

        x_train = x_train.transpose(0, 3, 1, 2).astype("float32")

        x_adv = attack.run_attack(data=x_train[[0]])
        x_adv = x_adv.adversarial_examples

        metric = AccuracyPerturbationMetric()
        metric.update(jptc, jptc.device, x_train[[0]], x_adv[[0]], y_train[[0]], acc_type='adversarial')
        clean_accuracy, adversarial_accuracy, perturbation_added = metric.compute()

        assert isinstance(metric, Metric)
        assert clean_accuracy == 1.0
        assert adversarial_accuracy == 0.0
        assert perturbation_added == pytest.approx(13.853668, 0.1)
        
        
        pgd_attack = ProjectedGradientDescentPyTorch(estimator=jptc, max_iter=1)
        attack = JaticAttack(pgd_attack)

        x_adv = attack.run_attack(data=x_train[:10])
        x_adv = x_adv.adversarial_examples
        metric.update(jptc, jptc.device, x_train[:10], x_adv, y_train[:10], acc_type='robust')
        clean_accuracy, robust_accuracy, perturbation_added = metric.compute()

        assert isinstance(metric, Metric)
        assert clean_accuracy == 0.5
        assert robust_accuracy == 0.3
        assert perturbation_added == pytest.approx(5.334152, 0.1)

    except HEARTTestException as e:
        art_warning(e)

def test_robustness_bias_metric(art_warning):
    try:
        from art.attacks.evasion import ProjectedGradientDescentPyTorch
        from heart_library.attacks.attack import JaticAttack
        from heart_library.metrics.metrics import RobustnessBiasMetric
        from maite.protocols import Metric
        import numpy as np

        jptc = get_cifar10_image_classifier_pt()
        pgd_attack = ProjectedGradientDescentPyTorch(estimator=jptc)
        attack = JaticAttack(pgd_attack)

        (x_train, y_train), (_, _), _, _ = load_dataset("cifar10")

        x_train = x_train.transpose(0, 3, 1, 2).astype("float32")

        x_adv = attack.run_attack(data=x_train[[0]])
        x_adv = x_adv.adversarial_examples

        metric = RobustnessBiasMetric()
        metric.update(jptc, jptc.device, x_train[[0]], x_adv[[0]], y_train[[0]])
        series = metric.compute()

        assert isinstance(metric, Metric)
        np.testing.assert_almost_equal(series[np.argmax(y_train[0])][:5], 
                                       np.array([[0.0, 1.0],
                                                [0.15003705265546086, 1.0],
                                                [0.3000741053109217, 1.0],
                                                [0.45011115796638257, 1.0],
                                                [0.6001482106218434, 1.0]]), 0.01)
    except HEARTTestException as e:
        art_warning(e)

def test_bb_quality_metric(art_warning):
    try:
        from heart_library.attacks.evasion import HeartHopSkipJump
        from heart_library.attacks.attack import JaticAttack
        from heart_library.metrics.metrics import BlackBoxAttackQualityMetric
        from maite.protocols import Metric
        import numpy as np

        jptc = get_cifar10_image_classifier_pt()
        bb_attack = HeartHopSkipJump(classifier=jptc, max_iter=5, max_eval=10, init_eval=2, init_size=10, )
        attack = JaticAttack(bb_attack)

        (x_train, y_train), (_, _), _, _ = load_dataset("cifar10")

        x_train = x_train.transpose(0, 3, 1, 2).astype("float32")

        x_adv = attack.run_attack(data=x_train[[0]])
        x_adv = x_adv.adversarial_examples

        metric = BlackBoxAttackQualityMetric()
        metric.update(attack)
        metrics = metric.compute()
        
        assert isinstance(metric, Metric)
        assert "total_queries" in metrics
        assert metrics["total_queries"][0] == pytest.approx(100, 10)
        
        assert "adv_queries" in metrics
        assert metrics["adv_queries"][0] == pytest.approx(40, 10)
        
        assert "benign_queries" in metrics
        assert metrics["benign_queries"][0] == pytest.approx(60, 10)
        
        assert "adv_query_idx" in metrics
        assert "adv_perturb_total" in metrics
        assert "adv_confs_total" in metrics
        
    except HEARTTestException as e:
        art_warning(e)

