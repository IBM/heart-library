"""Create Patch"""

import logging
import math
import random
import time
from typing import Any

import numpy as np
import requests
import torch
from art.attacks.evasion import AdversarialPatchPyTorch
from datasets import load_dataset
from numpy.typing import NDArray
from torchvision.transforms import transforms

from heart_library.attacks.attack import JaticAttack
from heart_library.catalogue.manager import AugmentationMetadata, EvaluationMetadata, ModelMetadata, TrainingMetadata
from heart_library.catalogue.workflows.workflow import Workflow
from heart_library.estimators.object_detection import JaticPyTorchObjectDetectionOutput, JaticPyTorchObjectDetector
from heart_library.metrics import HeartMAPMetric
from heart_library.utils import adjust_bboxes_resize

logger: logging.Logger = logging.getLogger(__name__)


def create_mask(image: NDArray[np.float32], bboxes: NDArray[np.float32]) -> NDArray[Any]:
    """
    Create a mask using bounding box coordinates.

    Args:
        image (np.ndarray): Input image as a NumPy array (HWC format).
        bboxes (list of tuples): List of bounding boxes in [x_min, y_min, x_max, y_max] format.

    Returns:
        NDArray: Mask with 1s for bounding box regions and 0s elsewhere.
    """
    # Get image dimensions
    _, height, width = np.asarray(image).shape

    # Initialize mask
    mask = np.zeros((height, width), dtype=np.uint8)

    # Iterate through bounding boxes and set mask values
    for bbox in bboxes:
        x_min, y_min, x_max, y_max = bbox
        mask[math.floor(y_min) : math.ceil(y_max), math.floor(x_min) : math.ceil(x_max)] = True  # Set region to 1

    return mask


class ImageDataset:
    """Image dataset"""

    metadata: dict[str, Any] = {"id": "COCO_OD_DATASET"}

    def __init__(self, data: NDArray[np.float32], target_label: int = -1) -> None:
        """Initialise"""
        self._data = data
        self._target_label = target_label

    def __len__(self) -> int:
        """Get the length"""
        return len(self._data)

    def __getitem__(self, ind: int) -> tuple[NDArray[np.float32], JaticPyTorchObjectDetectionOutput, dict[str, Any]]:
        """Get an item of the dataset"""
        image = self._data[ind]["image"]
        detection = {}
        detection["boxes"] = np.asarray(self._data[ind]["resized_bbox"])[:1]
        # detection["labels"] = np.asarray(self._data[ind]['objects']['label'])[:1]
        if self._target_label != -1:
            detection["labels"] = [self._target_label] * len(detection["boxes"])
            detection["scores"] = np.asarray(
                [[round(random.uniform(0.95, 1), 5)] for _ in range(len(detection["labels"]))],  # noqa: S311
            )
        else:
            detection["labels"] = np.asarray(self._data[ind]["objects"]["label"])[:1]
            detection["scores"] = np.asarray(
                [[round(random.uniform(0.5, 1), 5)] for _ in range(len(detection["labels"]))],  # noqa: S311
            )
        return (image, JaticPyTorchObjectDetectionOutput(detection), {"mask": create_mask(image, detection["boxes"])})


class CreateObjectDetectionPatch(Workflow):
    """
    The CreateObjectDetectionPatch workflow can be used to generate a foundation patch for a given
    dataset, model and AdversarialPatch attack and correctly record its associated
    metadata in the patch catalogue.
    """

    PATCH_TYPE: str = "FOUNDATION"

    def __init__(self, local_catalogue_path: str) -> None:
        """Initialise the patch creation workflow."""
        logger.info("Setting up workflow for creating patches.")
        super().__init__(local_catalogue_path=local_catalogue_path)

    def run(self, model_type: str, num_data_samples: int = 2, max_iter: int = 10) -> bool:
        """Run the patch creation workflow."""
        logger.info("Starting patch creation.")
        self._manager.new_run(workflow_name=self.PATCH_TYPE, tags={"mlflow.workflow_task": "OBJECT_DETECTION"})

        # Setup the model
        model_type = model_type
        input_shape = (3, 200, 200)
        clip_values = (0, 1)
        attack_losses = ("loss_classifier", "loss_box_reg", "loss_objectness", "loss_rpn_box_reg")

        model_metadata = ModelMetadata(
            model_type=model_type,
            input_shape=input_shape,
            model_optimizer="None",
            attack_losses=attack_losses,
            clip_values=clip_values,
            channels_first=True,
            device_type="cpu",
        )
        self._manager.log_model_metadata(model_metadata)

        detector = JaticPyTorchObjectDetector(
            model_type=model_type,
            input_shape=input_shape,
            clip_values=clip_values,
            attack_losses=attack_losses,
            device_type=model_metadata.device_type,
        )

        ## Setup the dataset
        coco_label_url = "https://gist.githubusercontent.com/tersekmatija/9d00c4683d52d94cf348acae29e8db1a/raw/8d1b042e7dd6061c760422f51e8e8488c6ad68c7/coco-labels-91.txt"
        response = requests.get(coco_label_url, timeout=90)
        data = response.text
        coco_labels = ["background"] + data.splitlines()

        dataset = load_dataset("rafaelpadilla/coco2017", split="train[20:30]")
        preprocess = transforms.Compose(
            [
                transforms.Resize((200, 200)),
                transforms.ToTensor(),
            ],
        )
        data = dataset.map(
            lambda x: {
                "image": preprocess(x["image"]),
                "resized_bbox": adjust_bboxes_resize(
                    x["objects"]["bbox"],
                    np.asarray(x["image"]).shape[1],
                    np.asarray(x["image"]).shape[0],
                    200,
                    200,
                ),
            },
        )
        sample_data = torch.utils.data.Subset(data, list(range(num_data_samples)))
        coco_data = ImageDataset(sample_data, target_label=coco_labels.index("train"))
        coco_data = torch.utils.data.Subset(coco_data, list(range(len(coco_data))))

        ## Setup the augmentation
        rotation_max = 0.0
        scale_min = 0.3
        scale_max = 0.3
        distortion_scale_max = 0.0
        learning_rate = 0.01
        max_iter = max_iter
        batch_size = 1
        patch_shape = (3, 100, 100)
        patch_location = (50, 50)  # to apply _random_overlay i.e. random translation of the patch
        patch_type = "circle"
        optimizer = "pgd"
        targeted = True

        attack = JaticAttack(
            AdversarialPatchPyTorch(
                detector,
                rotation_max=rotation_max,
                patch_location=patch_location,
                scale_min=scale_min,
                scale_max=scale_max,
                optimizer=optimizer,
                distortion_scale_max=distortion_scale_max,
                learning_rate=learning_rate,
                max_iter=max_iter,
                batch_size=batch_size,
                patch_shape=patch_shape,
                patch_type=patch_type,
                verbose=True,
                targeted=targeted,
            ),
        )

        augmentation_metadata = AugmentationMetadata(
            augmentation_name="AdversarialPatchPyTorch",
            max_iter=max_iter,
            batch_size=batch_size,
            learning_rate=learning_rate,
            rotation_max=rotation_max,
            scale_min=scale_min,
            scale_max=scale_max,
            distortion_scale_max=distortion_scale_max,
            patch_shape=patch_shape,
            patch_location=patch_location,
            patch_type=patch_type,
            attack_optimizer=optimizer,
            targeted=targeted,
        )

        self._manager.log_augmentation_metadata(augmentation_metadata)

        ## Execute patch generation

        start_time = time.time()
        adv_images, _, metadata = attack(coco_data)
        time_to_train = time.time() - start_time
        self._patch = metadata[0]["patch"]
        target_data = {
            "boxes": np.asarray([item[1].boxes[0] for item in coco_data]),
            "scores": np.asarray([item[1].scores[0] for item in coco_data]).ravel(),
            "labels": np.asarray([item[1].labels[0] for item in coco_data]).ravel(),
        }

        training_metadata = TrainingMetadata(
            training_dataset_name="COCO",
            num_input_samples=num_data_samples,
            input_data=np.asarray([np.array(item[0]) for item in coco_data]),
            target_data=target_data,
            time_to_train=time_to_train,
            patch=metadata[0]["patch"],
            patch_mask=metadata[0]["mask"],
        )
        self._manager.log_training_metadata(training_metadata)

        ## Setup the metrics
        map_args = {
            "box_format": "xyxy",
            "iou_type": "bbox",
            "iou_thresholds": [0.5],
            "rec_thresholds": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            "max_detection_thresholds": [1, 10, 100],
            "class_metrics": False,
            "extended_summary": False,
            "average": "macro",
        }
        metric = HeartMAPMetric(**map_args)

        ## Execute patch evaluation
        gt_data = ImageDataset(sample_data)
        gt_data = torch.utils.data.Subset(gt_data, list(range(len(gt_data))))
        gt = [item[1] for item in gt_data]

        detections = detector(gt_data)
        metric.reset()
        metric.update(detections, gt)
        self._benign_metrics = metric.compute()

        adv_detections = detector(adv_images)
        metric.reset()
        metric.update(adv_detections, gt)
        adv_vs_gt_metrics = metric.compute()

        metric.reset()
        metric.update(adv_detections, detections)
        adv_vs_pred_metrics = metric.compute()

        evaluation_metadata = EvaluationMetadata(
            performance_metric_name="map_50",
            performance_benign=self._benign_metrics["map_50"].item(),
            performance_adv_vs_gt=adv_vs_gt_metrics["map_50"].item(),
            performance_adv_vs_pred=adv_vs_pred_metrics["map_50"].item(),
        )
        self._manager.log_evaluation_metadata(evaluation_metadata)

        ## Clean-up
        self._manager.finish_run()
        logger.info("Finished patch creation.")

        return True
