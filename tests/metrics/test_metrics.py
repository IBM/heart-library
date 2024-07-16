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
from tests.utils import HEARTTestException, get_cifar10_image_classifier_pt
from art.utils import load_dataset


logger = logging.getLogger(__name__)


def test_accuracy_perturbation_metric(heart_warning):
    try:
        from art.attacks.evasion import ProjectedGradientDescentPyTorch
        from heart_library.attacks.attack import JaticAttack
        from heart_library.metrics.metrics import AccuracyPerturbationMetric
        from maite.protocols.image_classification import Metric
        import numpy as np

        jptc = get_cifar10_image_classifier_pt()
        pgd_attack = ProjectedGradientDescentPyTorch(estimator=jptc)
        attack = JaticAttack(pgd_attack, norm=2)

        (x_train, y_train), (_, _), _, _ = load_dataset("cifar10")

        x_train = x_train.transpose(0, 3, 1, 2).astype("float32")

        x_adv, _, metadata = attack(x_train[[0]])

        metric = AccuracyPerturbationMetric(jptc(x_train[[0]]), metadata, accuracy_type="adversarial")
        metric.update(jptc(x_adv), np.argmax(y_train[[0]], axis=1))
        acc_pert = metric.compute()

        assert isinstance(metric, Metric)
        assert acc_pert["clean_accuracy"] == 1.0
        assert acc_pert["adversarial_accuracy"] == 0.0
        assert acc_pert["mean_delta"] == pytest.approx(13.853668, 0.1)
        
        
        pgd_attack = ProjectedGradientDescentPyTorch(estimator=jptc, max_iter=1)
        attack = JaticAttack(pgd_attack, norm=2)

        x_adv, _, metadata = attack(x_train[:10])
        metric = AccuracyPerturbationMetric(jptc(x_train[:10]), metadata, accuracy_type="robust")
        metric.update(jptc(x_adv), np.argmax(y_train[:10], axis=1))
        acc_pert = metric.compute()

        assert isinstance(metric, Metric)
        assert acc_pert["clean_accuracy"] == 0.5
        assert acc_pert["robust_accuracy"] == 0.3
        assert acc_pert["mean_delta"] == pytest.approx(5.334152, 0.1)

    except HEARTTestException as e:
        heart_warning(e)

def test_robustness_bias_metric(heart_warning):
    try:
        from art.attacks.evasion import ProjectedGradientDescentPyTorch
        from heart_library.attacks.attack import JaticAttack
        from heart_library.metrics.metrics import RobustnessBiasMetric
        from maite.protocols.image_classification import Metric
        import numpy as np

        jptc = get_cifar10_image_classifier_pt()
        pgd_attack = ProjectedGradientDescentPyTorch(estimator=jptc)
        attack = JaticAttack(pgd_attack, norm=2)

        (x_train, y_train), (_, _), _, _ = load_dataset("cifar10")

        x_train = x_train.transpose(0, 3, 1, 2).astype("float32")

        x_adv, _, metadata = attack(x_train[[0]])
        metric = RobustnessBiasMetric(metadata, list(range(10)))
        metric.update(jptc(x_adv), jptc(x_train[[0]]))
        series = metric.compute()

        assert isinstance(metric, Metric)
        
        np.testing.assert_almost_equal(series[np.argmax(y_train[0])][:5], 
                                       np.array([[0.0, 1.0],
                                                [0.15003705265546086, 1.0],
                                                [0.3000741053109217, 1.0],
                                                [0.45011115796638257, 1.0],
                                                [0.6001482106218434, 1.0]]), 0.01)
    except HEARTTestException as e:
        heart_warning(e)

def test_bb_quality_metric(heart_warning):
    try:
        from heart_library.attacks.evasion import HeartHopSkipJump
        from heart_library.attacks.attack import JaticAttack
        from heart_library.metrics.metrics import BlackBoxAttackQualityMetric
        from maite.protocols.image_classification import Metric as ICMetric
        from maite.protocols.object_detection import Metric as ODMetric

        jptc = get_cifar10_image_classifier_pt()
        bb_attack = HeartHopSkipJump(classifier=jptc, max_iter=5, max_eval=10, init_eval=2, init_size=10, )
        attack = JaticAttack(bb_attack)

        (x_train, _), (_, _), _, _ = load_dataset("cifar10")

        x_train = x_train.transpose(0, 3, 1, 2).astype("float32")

        _, _, _ = attack(data=x_train[[0]])

        metric = BlackBoxAttackQualityMetric(attack)
        metric.update()
        metrics = metric.compute()
        
        assert isinstance(metric, (ICMetric, ODMetric))
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
        heart_warning(e)

