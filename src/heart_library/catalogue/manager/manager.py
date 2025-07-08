"""Catalogue Manager"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional, Union

import numpy as np
from numpy.typing import NDArray


@dataclass
class AugmentationMetadata:
    """Augmetnation metadata"""

    augmentation_name: str
    max_iter: int
    batch_size: int
    learning_rate: float
    rotation_max: float
    scale_min: float
    scale_max: float
    distortion_scale_max: float
    learning_rate: float
    patch_shape: tuple[int, int, int]
    patch_location: tuple[int, int]
    patch_type: str
    attack_optimizer: str
    targeted: bool


@dataclass
class ModelMetadata:
    """Model metadata"""

    model_type: str
    input_shape: tuple[int, ...]
    attack_losses: tuple[str, ...]
    device_type: str
    model_optimizer: Optional[str] = None
    clip_values: Optional[tuple[Union[int, float, NDArray[np.float32]], Union[int, float, NDArray[np.float32]]]] = None
    channels_first: bool = True
    preprocessing_defences: Any = None
    postprocessing_defences: Any = None
    postprocessing: Any = None
    is_yolov8: Any = False


@dataclass
class TrainingMetadata:
    """Training metadata"""

    training_dataset_name: str
    num_input_samples: int
    input_data: NDArray[np.float32]
    target_data: dict[str, Any]
    time_to_train: float
    patch: NDArray[np.float32]
    patch_mask: NDArray[np.float32]
    mask_data: Optional[NDArray[np.float32]] = None
    fine_tune_data: Optional[NDArray[np.float32]] = None
    loss_evolution: Optional[NDArray[np.float32]] = None
    loss_gradient_norms: Optional[NDArray[np.float32]] = None


@dataclass
class EvaluationMetadata:
    """Evaluation metadata"""

    performance_metric_name: str
    performance_benign: float
    performance_adv_vs_gt: float
    performance_adv_vs_pred: float


class CatalogueManager(ABC):
    """This class defines the abstract base class for Catalogue Managers."""

    def __init__(self, local_catalogue_path: str) -> None:
        """Initialize the Catalogue."""
        # This dir path needs to change based on storage designs
        self._catalogue_dir: str = local_catalogue_path

    @abstractmethod
    def log_augmentation_metadata(self, metadata: AugmentationMetadata) -> None:
        """Abstract method for logging augmentation metadata."""
        raise NotImplementedError

    @abstractmethod
    def log_model_metadata(self, metadata: ModelMetadata) -> None:
        """Abstract method for logging model metadata."""
        raise NotImplementedError

    @abstractmethod
    def log_training_metadata(self, metadata: TrainingMetadata) -> None:
        """Abstract method for logging training metadata."""
        raise NotImplementedError

    @abstractmethod
    def log_evaluation_metadata(self, metadata: EvaluationMetadata) -> None:
        """Abstract method for logging evaluation metadata."""
        raise NotImplementedError
