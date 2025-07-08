"""HEART MLFlow Client"""

import logging
import os
from dataclasses import asdict
from typing import Optional, Union

import mlflow
import numpy as np
import yaml
from numpy.typing import NDArray
from PIL import Image

from heart_library.catalogue.manager import (
    AugmentationMetadata,
    CatalogueManager,
    EvaluationMetadata,
    ModelMetadata,
    TrainingMetadata,
)
from heart_library.config import HEART_DATA_PATH

logger: logging.Logger = logging.getLogger(__name__)


class MLFlowClient(CatalogueManager):
    """MLFlow Client"""

    _instance: Optional[CatalogueManager] = None

    def __init__(self, local_catalogue_path: Optional[str] = None) -> None:
        """Initialise the MLFlow client."""
        if not hasattr(self, "_initialized") and local_catalogue_path:
            super().__init__(local_catalogue_path=local_catalogue_path)

            self._tracking_uri = self._catalogue_dir
            mlflow.set_tracking_uri(self._tracking_uri)

            self._initialized = True
            self._experiment_name = None
            self._run = None

    def __new__(cls, local_catalogue_path: Optional[str]) -> CatalogueManager:
        """Retrieve existing singleton instance of MLFlow if available."""
        if cls._instance is None and local_catalogue_path:
            cls._instance = super().__new__(cls)
        return cls._instance

    def new_run(self, workflow_name: str, tags: dict[str, str]) -> None:
        """New MLFlow run"""
        if self._run is not None:
            mlflow.end_run()
        self._experiment_name = workflow_name
        self._experiment = mlflow.set_experiment(experiment_name=self._experiment_name)
        self._run = mlflow.start_run(tags=tags)

    def finish_run(self) -> None:
        """Finish current run"""
        if self._run is not None:
            mlflow.end_run()
            self._run = None

    def _save_as_artifact(self, item: Optional[Union[NDArray, dict, Image.Image]], artifact_path: str) -> None:
        """Save as an artifact"""
        if item is not None:
            local_path = f"{HEART_DATA_PATH}/{artifact_path}"
            if isinstance(item, np.ndarray):
                local_path = f"{local_path}.npy"
                np.save(local_path, item)
            elif isinstance(item, dict):
                local_path = f"{local_path}.yaml"
                with open(local_path, "w") as file:
                    yaml.dump(item, file)
            elif isinstance(item, Image.Image):
                local_path = f"{local_path}.png"
                item.save(local_path)
            mlflow.log_artifact(local_path=local_path)
            os.remove(local_path)

    def log_augmentation_metadata(self, metadata: AugmentationMetadata) -> None:
        """Method for logging augmentation metadata."""

        logger.info("Logging augmentation metadata.")
        mlflow.log_params(asdict(metadata))

    def log_model_metadata(self, metadata: ModelMetadata) -> None:
        """Method for logging estimator metadata."""
        logger.info("Logging estimator metadata.")
        mlflow.log_params(asdict(metadata))

    def log_training_metadata(self, metadata: TrainingMetadata) -> None:
        """Method for logging training metadata."""
        logger.info("Logging training metadata.")

        mlflow.log_param("training_dataset_name", metadata.training_dataset_name)
        mlflow.log_param("num_input_samples", metadata.num_input_samples)
        self._save_as_artifact(metadata.input_data, "training_input_data")
        self._save_as_artifact(metadata.target_data, "training_target_data")
        self._save_as_artifact(
            Image.fromarray((metadata.patch * 255).transpose(1, 2, 0).astype(np.uint8)),
            "training_patch",
        )
        self._save_as_artifact(metadata.patch_mask, "training_patch_mask")
        self._save_as_artifact(metadata.fine_tune_data, "training_fine_tune_data")
        self._save_as_artifact(metadata.loss_evolution, "training_loss_evolution")
        self._save_as_artifact(metadata.loss_gradient_norms, "training_loss_gradient_norms")
        mlflow.log_metric("time_to_train", metadata.time_to_train)

    def log_evaluation_metadata(self, metadata: EvaluationMetadata) -> None:
        """Method for logging evaluation metadata."""
        logger.info("Logging evaluation metadata.")
        mlflow.log_metric(f"{metadata.performance_metric_name}_benign", metadata.performance_benign)
        mlflow.log_metric(f"{metadata.performance_metric_name}_adv_vs_gt", metadata.performance_adv_vs_gt)
        mlflow.log_metric(f"{metadata.performance_metric_name}_adv_vs_pred", metadata.performance_adv_vs_pred)
