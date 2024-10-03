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
"""
Module implementing varying metrics for assessing model robustness. These fall mainly under two categories:
attack-dependent and attack-independent.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import logging
from typing import Any, Dict, Sequence, Union

import numpy as np

from heart_library.attacks.attack import JaticAttack
from heart_library.estimators.object_detection import \
    JaticPyTorchObjectDetectionOutput

logger = logging.getLogger(__name__)


class HeartMAPMetric:
    """
    Facilitating support for Torchmetric's MAP metric.
    """

    def __init__(self, **kwargs):
        """
        :param **kwargs: arguments passed to Torchmetric's MAP metric
        """
        from torchmetrics.detection.mean_ap import \
            MeanAveragePrecision  # pylint: disable=C0415

        self.metric = MeanAveragePrecision(**kwargs)

    def reset(self) -> None:
        """
        Clear contents of current metric's cache of predictions and targets.
        """
        return self.metric.reset()

    def update(
        self,
        preds_batch: Sequence[JaticPyTorchObjectDetectionOutput],
        targets_batch: Sequence[JaticPyTorchObjectDetectionOutput],
    ) -> None:
        """
        Add predictions and targets to metric's cache for later calculation.
        :param preds_batch: predictions in ObjectDetectionTarget format.
        :param targets_batch: groundtruth targets in ObjectDetectionTarget format.
        """
        import torch  # pylint: disable=C0415

        # Torchmetrics mAP expects list of dicts with one dict per image; each dict with:
        # - boxes: Tensor w/shape (num_boxes, 4)
        # - scores: Tensor w/shape (num_boxes)
        # - labels: Tensor w/shape (num_boxes)
        # iterate over images in batch
        for preds, targets in zip(preds_batch, targets_batch):
            # put predictions and labels in dictionaries
            # tensor bridge to PyTorch tensors (required by Torchmetrics)
            preds_dict = {
                "boxes": torch.as_tensor(preds.boxes),
                "scores": torch.as_tensor(preds.scores),
                "labels": torch.as_tensor(preds.labels),
            }
            targets_dict = {
                "boxes": torch.as_tensor(targets.boxes),
                "scores": torch.as_tensor(targets.scores),
                "labels": torch.as_tensor(targets.labels),
            }
            self.metric.update([preds_dict], [targets_dict])

    def compute(self) -> Dict[str, Any]:
        """
        Compute MAP scores.
        """
        return self.metric.compute()


class HeartAccuracyMetric:
    """
    Facilitating support for Torchmetric's Accuracy metric.
    """

    def __init__(self, is_logits: bool = True, **kwargs):
        """
        :param is_logits: bool indicating if predictions are logits
        :param **kwargs: arguments passed to Torchmetric's Accuracy metric
        """
        from torchmetrics.classification import (  # pylint: disable=C0415
            BinaryAccuracy, MulticlassAccuracy, MultilabelAccuracy)

        self.is_logits = is_logits
        self._metric: Union[BinaryAccuracy, MulticlassAccuracy, MultilabelAccuracy]
        self._task = kwargs.pop("task")
        if self._task == "binary":
            self._metric = BinaryAccuracy(**kwargs)
        elif self._task == "multiclass":
            self._metric = MulticlassAccuracy(**kwargs)
        elif self._task == "multilabel":
            self._metric = MultilabelAccuracy(**kwargs)

    def reset(self) -> None:
        """
        Clear contents of current metric's cache of predictions and targets.
        """
        return self._metric.reset()

    def update(self, preds_batch: Sequence[np.ndarray], targets_batch: Sequence[np.ndarray]) -> None:
        """
        Add predictions and targets to metric's cache for later calculation.
        :param preds_batch: predictions in numpy array format
        :param targets_batch: groundtruth targets in numpy array format
        """
        import torch  # pylint: disable=C0415

        if self.is_logits:
            preds = torch.as_tensor(np.argmax(preds_batch, axis=2).ravel())
        else:
            preds = torch.as_tensor(preds_batch)
        targets = torch.as_tensor(targets_batch)
        self._metric.update(preds, targets)

    def compute(self) -> Dict[str, float]:
        """
        Compute accuracy score.
        """
        return {"accuracy": self._metric.compute().item()}


class RobustnessBiasMetric:
    """
    A metric which describes Robustness Bias for features
    of datasets. Currently supports only classification tasks.
    """

    def __init__(self, metadata: Sequence[Dict[str, Any]], labels: np.ndarray, interval: int = 100):
        """
        :param metadata: Sequence[Dict[str, Any]] - the metadata computed during attack
            which contains delta between benign and adversarial images
        :param labels: List[str] - classification labels
        :param interval: int - tau
        """
        self._state: Dict = {}
        self._labels: np.ndarray = labels
        self._metadata: Sequence[Dict[str, Any]] = metadata
        self._interval: int = interval

    def reset(self):
        """
        Reset the metric to default values.
        """
        self._state = {}

    def update(self, preds_batch: Sequence[np.ndarray], targets_batch: Sequence[np.ndarray]) -> None:
        """
        Add predictions and targets to metric's cache for later calculation.
        :param preds_batch: predictions in numpy array format
        :param targets_batch: groundtruth targets in numpy array format
        """
        try:
            errors = np.stack([item["delta"] for item in self._metadata])
        except KeyError as key_error:
            raise KeyError(
                "Delta not computed for metadata. Set norm > 0 of JaticAttack to compute delta."
            ) from key_error

        # assuming the targets batch is the groundtruth or original predictions
        # and the preds batch are predictions for the augmented / attacked data
        success = (np.argmax(targets_batch, axis=2).ravel() != np.argmax(preds_batch, axis=2).ravel()).astype(int)

        taus = np.linspace(0, max(errors) + 1, self._interval)
        series: Dict = {}
        for tau in taus:
            for label in range(len(self._labels)):
                idxs_of_label = np.argwhere(np.argmax(targets_batch, axis=2).ravel() == label).ravel()
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

    def compute(self) -> Dict:
        """
        Returns the computed metric
        """
        return self._state


class AccuracyPerturbationMetric:
    """
    A metric for easily calculating the clean and robust accuracy
    as well as the perturbation between clean and adversarial input.
    """

    def __init__(
        self,
        benign_predictions: Sequence[np.ndarray],
        metadata: Sequence[Dict[str, Any]],
        accuracy_type: str = "robust",
    ):
        """
        :param accuracy_type: str - the type of accuracy to calculate. Choice of "adversarial" or "robust".
            - Robust accuracy is the accuracy of the model on all samples
            - Adversarial accuracy is the accuracy of the model only samples which were
              correctly predicted in the non-adversarial scenario
        """
        self._state: Dict = {}
        self._benign_predictions = benign_predictions
        self._metadata = metadata
        self._accuracy_type = accuracy_type

    def reset(self):
        """
        Reset the metric to default values.
        """
        self._state = {}

    def update(self, preds_batch: Sequence[np.ndarray], targets_batch: Sequence[np.ndarray]):
        """
        Updates the metric value.
        """
        y_orig = np.argmax(self._benign_predictions, axis=2).ravel()
        y_pred = np.argmax(preds_batch, axis=2).ravel()

        try:
            mean_delta = np.stack([item["delta"] for item in self._metadata]).mean()
        except KeyError as key_error:
            raise KeyError(
                "Delta not computed for metadata. Set norm > 0 of JaticAttack to compute delta."
            ) from key_error

        y_corr = y_orig == targets_batch

        clean_acc = np.sum(y_corr) / len(y_orig)
        attack_acc: float = 0.0
        if self._accuracy_type == "adversarial":
            attack_acc = np.sum((y_pred == y_orig) & y_corr) / np.sum(y_corr)
        elif self._accuracy_type == "robust":
            attack_acc = np.mean(y_pred == targets_batch)

        self._state = {
            "clean_accuracy": clean_acc,
            f"{self._accuracy_type}_accuracy": attack_acc,
            "mean_delta": mean_delta,
        }

    def compute(self) -> Dict[str, float]:
        """
        Returns the computed metric
        in Tuple (clean_accuracy, robust_accuracy, average_perturbation)
        """
        return self._state


class BlackBoxAttackQualityMetric:
    """
    A metric for extracting the black box quality metrics.
    """

    def __init__(self, attack: JaticAttack):
        """
        :param attack: JaticAttack - the black-box attack (currently only HopSkipJump supported)
        """
        self._state: Dict = {}
        self._attack = attack._attack

    def reset(self):
        """
        Reset the metric to default values.
        """
        self._state = {}

    def update(self):
        """
        Updates the metric value.
        """
        total_queries = self._attack.total_queries
        adv_query_idx = self._attack.adv_query_idx
        adv_queries = [len(item) for item in adv_query_idx]
        benign_queries = [total_queries[i] - n_adv for i, n_adv in enumerate(adv_queries)]
        adv_perturb_total = self._attack.perturbs
        adv_perturb_iter = self._attack.perturbs_iter
        adv_confs_total = self._attack.confs

        self._state = {
            "total_queries": total_queries,
            "adv_queries": adv_queries,
            "benign_queries": benign_queries,
            "adv_query_idx": adv_query_idx,
            "adv_perturb_total": adv_perturb_total,
            "adv_perturb_iter": adv_perturb_iter,
            "adv_confs_total": adv_confs_total,
        }

    def compute(self) -> Dict[str, Any]:
        """
        Returns the computed metric
        in dict
        """
        return self._state
