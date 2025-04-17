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

import logging
import uuid
from collections.abc import Sequence
from typing import Any, Optional

import numpy as np
from numpy.typing import NDArray

from heart_library.attacks.attack import JaticAttack
from heart_library.estimators.object_detection import JaticPyTorchObjectDetectionOutput

logger: logging.Logger = logging.getLogger(__name__)


class HeartMAPMetric:
    """
    Facilitating support for Torchmetric's MAP metric.

    Examples
    --------

    We can define a MAP metric and evaluate it on a JaticPyTorchObjectDetector's performance:

    >>> from maite.workflows import evaluate
    >>> from heart_library.estimators.object_detection import JaticPyTorchObjectDetector
    >>> import torch
    >>> import numpy
    >>> from datasets import load_dataset
    >>> from torchvision.transforms import transforms
    >>> from copy import deepcopy

    Define the JaticPyTorchObjectDetector, in this case passing in a resnet model:

    >>> MEAN = [0.485, 0.456, 0.406]
    >>> STD = [0.229, 0.224, 0.225]
    >>> preprocessing = (MEAN, STD)

    >>> detector = JaticPyTorchObjectDetector(
    ...     model_type="detr_resnet50_dc5",
    ...     input_shape=(3, 800, 800),
    ...     clip_values=(0, 1),
    ...     attack_losses=(
    ...         "loss_ce",
    ...         "loss_bbox",
    ...         "loss_giou",
    ...     ),
    ...     device_type="cpu",
    ...     optimizer=torch.nn.CrossEntropyLoss(),
    ...     preprocessing=preprocessing,
    ... )

    Prepare images for detection:

    >>> data = load_dataset("guydada/quickstart-coco", split="train[20:25]")
    >>> preprocess = transforms.Compose([transforms.Resize(800), transforms.CenterCrop(800), transforms.ToTensor()])

    >>> data = data.map(lambda x: {"image": preprocess(x["image"]), "label": None})

    Execute object detection and return JaticPyTorchObjectDetectionOutput:

    >>> detections = detector(data)

    Define data with detections:

    >>> class ImageDataset:
    ...     def __init__(self, images, groundtruth, threshold=0.8):
    ...         self.images = images
    ...         self.groundtruth = groundtruth
    ...         self.threshold = threshold
    ...
    ...     def __len__(self) -> int:
    ...         return len(self.images)
    ...
    ...     def __getitem__(self, ind: int) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    ...         image = np.asarray(self.images[ind]["image"]).astype(np.float32)
    ...         filtered_detection = self.groundtruth[ind]
    ...         filtered_detection.boxes = filtered_detection.boxes[filtered_detection.scores > self.threshold]
    ...         filtered_detection.labels = filtered_detection.labels[filtered_detection.scores > self.threshold]
    ...         filtered_detection.scores = filtered_detection.scores[filtered_detection.scores > self.threshold]
    ...         return (image, filtered_detection, None)

    >>> data_with_detections = ImageDataset(data, deepcopy(detections), threshold=0.9)

    Set the MAP parameters and evaluate:

    >>> map_args = {
    ...     "box_format": "xyxy",
    ...     "iou_type": "bbox",
    ...     "iou_thresholds": [0.5],
    ...     "rec_thresholds": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    ...     "max_detection_thresholds": [1, 10, 100],
    ...     "class_metrics": False,
    ...     "extended_summary": False,
    ...     "average": "macro",
    ... }

    >>> metric = HeartMAPMetric(**map_args)

    >>> results, _, metadata = evaluate(
    ...     model=detector,
    ...     dataset=data_with_detections,
    ...     metric=metric,
    ... )

    >>> results["map"]
    tensor(1.)
    """

    metadata: dict[str, Any]

    def __init__(self, metadata_id: Optional[str] = None, **kwargs: Any) -> None:  # noqa ANN401
        """HeartMAPMetric initialization.
        Args:
            **kwargs: arguments passed to Torchmetric's MAP metric."""
        from torchmetrics.detection.mean_ap import MeanAveragePrecision

        self._metric = MeanAveragePrecision(**kwargs)
        self.metadata = {"id": metadata_id if metadata_id is not None else str(uuid.uuid4())}

    def reset(self) -> None:
        """Clear contents of current metric's cache of predictions and targets.

        Returns:
            _type_: None.
        """
        return self._metric.reset()

    def update(
        self,
        preds_batch: Sequence[JaticPyTorchObjectDetectionOutput],
        targets_batch: Sequence[JaticPyTorchObjectDetectionOutput],
    ) -> None:
        """Add predictions and targets to metric's cache for later calculation.

        Args:
            preds_batch (`Sequence[JaticPyTorchObjectDetectionOutput]`): predictions in ObjectDetectionTarget format.
            targets_batch (`Sequence[JaticPyTorchObjectDetectionOutput]`):
                groundtruth targets in ObjectDetectionTarget format.
        """
        import torch

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
            self._metric.update([preds_dict], [targets_dict])

    def compute(self) -> dict[str, Any]:
        """Compute MAP scores.

        Returns:
            dict[str, Any]: Final value from the state of the metric.
        """
        return self._metric.compute()


class HeartAccuracyMetric:
    """Facilitating support for Torchmetric's Accuracy metric."""

    metadata: dict[str, Any]

    def __init__(self, is_logits: bool = True, metadata_id: Optional[str] = None, **kwargs: Any) -> None:  # noqa ANN401
        """HeartAccuracyMetric initialization.

        Args:
            is_logits (bool, optional): bool indicating if predictions are logits. Defaults to True.
            **kwargs: arguments passed to Torchmetric's Accuracy metric.
        """
        from torchmetrics.classification import (
            BinaryAccuracy,
            MulticlassAccuracy,
            MultilabelAccuracy,
        )

        self.is_logits = is_logits
        self._metric: BinaryAccuracy | MulticlassAccuracy | MultilabelAccuracy
        self._task = kwargs.pop("task")
        if self._task == "binary":
            self._metric = BinaryAccuracy(**kwargs)
        elif self._task == "multiclass":
            self._metric = MulticlassAccuracy(**kwargs)
        elif self._task == "multilabel":
            self._metric = MultilabelAccuracy(**kwargs)
        self.metadata = {"id": metadata_id if metadata_id is not None else str(uuid.uuid4())}

    def reset(self) -> None:
        """Clear contents of current metric's cache of predictions and targets.

        Returns:
            _type_: None.
        """
        return self._metric.reset()

    def update(self, preds_batch: Sequence[NDArray[np.float32]], targets_batch: Sequence[NDArray[np.float32]]) -> None:
        """Add predictions and targets to metric's cache for later calculation.

        Args:
            preds_batch (Sequence[NDArray[np.float32]]): edictions in numpy array format.
            targets_batch (Sequence[NDArray[np.float32]]): groundtruth targets in numpy array format.
        """
        import torch

        if self.is_logits:
            preds = torch.as_tensor(np.argmax(np.asarray(preds_batch), axis=1).ravel())
        else:
            preds = torch.as_tensor(np.asarray(preds_batch)).ravel()
        targets = torch.as_tensor(np.asarray(targets_batch)).ravel()
        self._metric.update(preds, targets)

    def compute(self) -> dict[str, float]:
        """Compute accuracy score.

        Returns:
            dict[str, float]: Final value from the state of the metric.
        """
        return {"accuracy": self._metric.compute().item()}


class RobustnessBiasMetric:
    """
    A metric which describes Robustness Bias for features
    of datasets. Currently supports only classification tasks.
    """

    metadata: dict[str, Any]

    def __init__(
        self,
        metadata: Sequence[dict[str, Any]],
        labels: NDArray[np.float32],
        interval: int = 100,
        metadata_id: Optional[str] = None,
    ) -> None:
        """RobustnessBiasMetric initialization.

        Args:
            metadata (Sequence[dict[str, Any]]): the metadata computed during attack
                which contains delta between benign and adversarial images.
            labels (NDArray[np.float32]): classification labels.
            interval (int, optional): tau. Defaults to 100.
        """
        self._state: dict[str, Any] = {}
        self._labels: np.ndarray = labels
        self._metadata: Sequence[dict[str, Any]] = metadata
        self._interval: int = interval
        self.metadata = {"id": metadata_id if metadata_id is not None else str(uuid.uuid4())}

    def reset(self) -> None:
        """Reset the metric to default values."""
        self._state = {}

    def update(self, preds_batch: Sequence[NDArray[np.float32]], targets_batch: Sequence[NDArray[np.float32]]) -> None:
        """Add predictions and targets to metric's cache for later calculation.

        Args:
            preds_batch (Sequence[NDArray[np.float32]]): predictions in numpy array format.
            targets_batch (Sequence[NDArray[np.float32]]): groundtruth targets in numpy array format.

        Raises:
            KeyError: if Delta not computed for metadata.
        """
        try:
            errors = np.stack([item["delta"] for item in self._metadata])
        except KeyError as key_error:
            raise KeyError(
                "Delta not computed for metadata. Set norm > 0 of JaticAttack to compute delta.",
            ) from key_error

        # assuming the targets batch is the groundtruth or original predictions
        # and the preds batch are predictions for the augmented / attacked data
        success = (
            np.argmax(np.asarray(targets_batch), axis=1).ravel() != np.argmax(np.asarray(preds_batch), axis=1).ravel()
        ).astype(int)

        taus = np.linspace(0, max(errors) + 1, self._interval)
        series: dict = {}
        for tau in taus:
            for label in range(len(self._labels)):
                idxs_of_label = np.argwhere(np.argmax(np.asarray(targets_batch), axis=1).ravel() == label).ravel()
                idxs_of_label_success = np.argwhere(success[idxs_of_label] == 1).ravel()
                idxs_of_label = idxs_of_label[idxs_of_label_success]
                errors_of_label = errors[idxs_of_label]
                idx_error_greater_tau = np.argwhere(errors_of_label > tau).ravel()
                if len(errors_of_label) != 0:
                    proportion = len(idx_error_greater_tau) / len(errors_of_label)
                    self.__populate_series(label, series, tau, proportion)

        self._state = series

    def compute(self) -> dict[str, Any]:
        """Returns the computed metric.

        Returns:
            dict[str, Any]: Final value from the state of the metric.
        """
        return self._state

    def __populate_series(self, label: int, series: dict[int, Any], tau: float, proportion: float) -> None:
        """Add tau, proportion pair to series based on if label already exists.

        Args:
            label (int): Index of series.
            series (dict[str, Any]): dict to be added to.
            tau (float): tau.
            proportion (float): proportion.
        """
        if label in series:
            series[label].append([tau, proportion])
        else:
            series[label] = [[tau, proportion]]

        self._state = series


class AccuracyPerturbationMetric:
    """
    A metric for easily calculating the clean and robust accuracy
    as well as the perturbation between clean and adversarial input.
    """

    metadata: dict[str, Any]

    def __init__(
        self,
        benign_predictions: Sequence[NDArray[np.float32]],
        metadata: Sequence[dict[str, Any]],
        accuracy_type: str = "robust",
        metadata_id: Optional[str] = None,
    ) -> None:
        """AccuracyPerturbationMetric initialization.

        Args:
            benign_predictions (Sequence[NDArray[np.float32]]): _description_
            metadata (Sequence[Dict[str, Any]]): _description_
            accuracy_type (str, optional): the type of accuracy to calculate. Choice of "adversarial" or "robust".
                - Robust accuracy is the accuracy of the model on all samples
                - Adversarial accuracy is the accuracy of the model only samples which were
                  correctly predicted in the non-adversarial scenario. Defaults to "robust".
        """
        self._state: dict = {}
        self._benign_predictions = benign_predictions
        self._metadata = metadata
        self._accuracy_type = accuracy_type
        self.metadata = {"id": metadata_id if metadata_id is not None else str(uuid.uuid4())}

    def reset(self) -> None:
        """Reset the metric to default values."""
        self._state = {}

    def update(self, preds_batch: Sequence[NDArray[np.float32]], targets_batch: Sequence[NDArray[np.float32]]) -> None:
        """Updates the metric value.

        Args:
            preds_batch (Sequence[NDArray[np.float32]]): Predicted values.
            targets_batch (Sequence[NDArray[np.float32]]): Target values.

        Raises:
            KeyError: if Delta not computed for metadata.
        """
        y_orig = np.argmax(np.stack(self._benign_predictions), axis=1).ravel()
        y_pred = np.argmax(np.stack(preds_batch), axis=1).ravel()

        try:
            mean_delta = np.stack([item["delta"] for item in self._metadata]).mean()
        except KeyError as key_error:
            raise KeyError(
                "Delta not computed for metadata. Set norm > 0 of JaticAttack to compute delta.",
            ) from key_error

        y_corr = y_orig == np.stack(targets_batch)

        clean_acc = np.sum(y_corr) / len(y_orig)
        attack_acc: float = 0.0
        if self._accuracy_type == "adversarial":
            attack_acc = np.sum((y_pred == y_orig) & y_corr) / np.sum(y_corr)
        elif self._accuracy_type == "robust":
            attack_acc = np.mean(y_pred == np.stack(targets_batch))

        self._state = {
            "clean_accuracy": clean_acc,
            f"{self._accuracy_type}_accuracy": attack_acc,
            "mean_delta": mean_delta,
        }

    def compute(self) -> dict[str, float]:
        """Returns the computed metric
        in Tuple (clean_accuracy, robust_accuracy, average_perturbation)

        Returns:
            dict[str, float]: Final value from the state of the metric.
        """
        return self._state


class BlackBoxAttackQualityMetric:
    """A metric for extracting the black box quality metrics."""

    metadata: dict[str, Any]

    def __init__(self, attack: JaticAttack, metadata_id: Optional[str] = None) -> None:
        """BlackBoxAttackQualityMetric initialization.
        Args:
            attack (JaticAttack): the black-box attack (currently only HopSkipJump supported)."""
        self._state: dict = {}
        self._attack = attack.get_attack()
        self.metadata = {"id": metadata_id if metadata_id is not None else str(uuid.uuid4())}

    def reset(self) -> None:
        """Reset the metric to default values."""
        self._state = {}

    def update(self) -> None:
        """Updates the metric value."""
        total_queries = getattr(self._attack, "total_queries", np.array([]))
        adv_query_idx = getattr(self._attack, "adv_query_idx", [])
        adv_queries = [len(item) for item in adv_query_idx]
        benign_queries = [total_queries[i] - n_adv for i, n_adv in enumerate(adv_queries)]
        adv_perturb_total = getattr(self._attack, "perturbs", [])
        adv_perturb_iter = getattr(self._attack, "perturbs_iter", [])
        adv_confs_total = getattr(self._attack, "confs", [])

        self._state = {
            "total_queries": total_queries,
            "adv_queries": adv_queries,
            "benign_queries": benign_queries,
            "adv_query_idx": adv_query_idx,
            "adv_perturb_total": adv_perturb_total,
            "adv_perturb_iter": adv_perturb_iter,
            "adv_confs_total": adv_confs_total,
        }

    def compute(self) -> dict[str, Any]:
        """Returns the computed metric
        in dict

        Returns:
            dict[str, Any]: Final value from the state of the metric.
        """
        return self._state
