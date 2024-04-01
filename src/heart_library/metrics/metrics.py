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
"""
Module implementing varying metrics for assessing model robustness. These fall mainly under two categories:
attack-dependent and attack-independent.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import logging
from typing import Any, Dict, Sequence, Tuple

import numpy as np
import numpy.linalg as la
from maite.protocols import (ArrayLike, HasLogits, HasProbs, HasScores,
                             ImageClassifier, SupportsArray)
from numpy.typing import NDArray

from heart_library.utils import process_inputs_for_art

logger = logging.getLogger(__name__)


class RobustnessBiasMetric:
    """
    A metric which describes Robustness Bias for features
    of datasets. Currently supports only classification tasks.
    """
    def __init__(self):
        self._state = {}

    def reset(self):
        """
        Reset the metric to default values.
        """
        self._state = {}

    def update(
        self,
        classifier: ImageClassifier,
        device: str,
        data: SupportsArray,
        attack_out: NDArray,
        labels: NDArray = np.array(None),
        norm_type: int = 2,
        interval: int = 100,
    ):
        """
        Updates the metric value. Takes as input:
        :param classifier: The image classifier
        :param device: The device on which to compute the metric
        :param data: The clean sample data
        :param attack_out: The adversarial sample data
        :param labels: The classification labels
        :param norm_type: The norm to use when calculating distance
        """

        y: Any = np.array(None)
        if labels.all() is None:
            # labels not explicitly provided, assume they are in data
            x, y = process_inputs_for_art(data, device)
        else:
            x, _ = process_inputs_for_art(data, device)
            if isinstance(y, int):
                y = [labels]
            else:
                y = labels

        if isinstance(x, Sequence):
            assert x[0].shape == attack_out.shape
        else:
            assert x.shape == attack_out.shape

        if isinstance(y, np.ndarray) and len(y.shape) > 1:
            y = np.argmax(y, axis=1)

        errors = np.linalg.norm((x - attack_out).reshape(x.shape[0], -1), ord=norm_type, axis=1)

        orig_result = classifier(x)
        attack_result = classifier(attack_out)

        success: NDArray
        if isinstance(orig_result, HasLogits) and isinstance(attack_result, HasLogits):
            success = (np.argmax(orig_result.logits, axis=1) != np.argmax(attack_result.logits, axis=1)).astype(int)
        elif isinstance(orig_result, HasProbs) and isinstance(attack_result, HasProbs):
            success = (np.argmax(orig_result.probs, axis=1) != np.argmax(attack_result.probs, axis=1)).astype(int)
        elif isinstance(orig_result, HasScores) and isinstance(attack_result, HasScores):
            success = (np.argmax(orig_result.scores, axis=1) != np.argmax(attack_result.scores, axis=1)).astype(int)
        else:
            raise ValueError

        taus = np.linspace(0, max(errors) + 1, interval)
        series: Dict = {}
        for tau in taus:
            for label in range(len(classifier.get_labels())):
                idxs_of_label = np.argwhere(np.array(y) == label).ravel()
                idxs_of_label_success = np.argwhere(success[idxs_of_label] == 1).ravel()
                idxs_of_label = idxs_of_label[idxs_of_label_success]
                errors_of_label = errors[idxs_of_label]
                idx_error_greater_tau = np.argwhere(errors_of_label > tau).ravel()
                if len(errors_of_label) != 0:
                    proportion = len(idx_error_greater_tau) / len(errors_of_label)
                    if label in series:
                        series[label].append([tau, proportion])
                    else:
                        series[label] = [[tau, proportion]]

        self._state = series

    def compute(self) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
        """
        Returns the computed metric
        in Tuple (clean_accuracy, robust_accuracy, average_perturbation)
        """
        return self._state

    def to(self, device: Any):  # pylint: disable=C0103,W0613
        """Unused protocol"""
        return self


class AccuracyPerturbationMetric:
    """
    A metric for easily calculating the clean and robust accuracy
    as well as the perturbation between clean and adversarial input.
    """

    def __init__(self):
        self._state = (np.array([0.0]), np.array([0.0]), np.array([0.0]))

    def reset(self):
        """
        Reset the metric to default values.
        """
        self._state = (np.array([0.0]), np.array([0.0]), np.array([0.0]))

    def update(
        self,
        classifier: ImageClassifier,
        device: str,
        data: SupportsArray,
        attack_out: NDArray,
        labels: NDArray = np.array(None),
        norm_type: int = 2,
        acc_type: str = "robust",
    ):
        """
        Updates the metric value. Takes as input:
        :param classifier: The image classifier
        :param device: The device on which to compute the metric
        :param data: The clean sample data
        :param attack_out: The adversarial sample data
        :param labels: The classification labels
        :param norm_type: The norm to use when calculating distance
        :param acc_type: The type of accuracy to calculate. Choice of "adversarial" or "robust".
            - Robust accuracy is the accuracy of the model on all samples
            - Adversarial accuracy is the accuracy of the model only samples which were
              correctly predicted in the non-adversarial scenario
        """
        y: Any = np.array(None)
        if labels.all() is None:
            # labels not explicitly provided, assume they are in data
            x, y = process_inputs_for_art(data, device)
        else:
            x, _ = process_inputs_for_art(data, device)
            if isinstance(y, int):
                y = [labels]
            else:
                y = labels

        if isinstance(x, Sequence):
            assert x[0].shape == attack_out.shape
        else:
            assert x.shape == attack_out.shape

        orig_result = classifier(x)
        attack_result = classifier(attack_out)

        if isinstance(orig_result, HasLogits) and isinstance(attack_result, HasLogits):
            y_orig = np.argmax(orig_result.logits, axis=1)
            y_pred = np.argmax(attack_result.logits, axis=1)
        elif isinstance(orig_result, HasProbs) and isinstance(attack_result, HasProbs):
            y_orig = np.argmax(orig_result.probs, axis=1)
            y_pred = np.argmax(attack_result.probs, axis=1)
        elif isinstance(orig_result, HasScores) and isinstance(attack_result, HasScores):
            y_orig = np.argmax(orig_result.scores, axis=1)
            y_pred = np.argmax(attack_result.scores, axis=1)
        else:
            raise ValueError

        if isinstance(y, np.ndarray) and len(y.shape) > 1:
            y = np.argmax(y, axis=1)

        y_corr = y_orig == y

        clean_acc = np.sum(y_corr) / len(y_orig)
        if acc_type == "adversarial":
            attack_acc = np.sum((y_pred == y_orig) & y_corr) / np.sum(y_corr)
        elif acc_type == "robust":
            attack_acc = np.mean(y_pred == y)

        idxs = y_pred != y
        avg_perts = 0.0
        perts_norm = la.norm((attack_out - x).reshape(x.shape[0], -1), ord=norm_type, axis=1)
        perts_norm = perts_norm[idxs]
        avg_perts = np.mean(perts_norm)

        self._state = (clean_acc, attack_acc, avg_perts)

    def compute(self) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
        """
        Returns the computed metric
        in Tuple (clean_accuracy, robust_accuracy, average_perturbation)
        """
        return self._state

    def to(self, device: Any):  # pylint: disable=C0103,W0613
        """Unused protocol"""
        return self
