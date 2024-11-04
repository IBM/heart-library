"""
Module providing object detection estimators
"""

from heart_library.estimators.object_detection.pytorch import (
    COCO_DETR_LABELS, COCO_FASTER_RCNN_LABELS, COCO_YOLO_LABELS,
    SUPPORTED_DETECTORS, JaticPyTorchObjectDetectionOutput,
    JaticPyTorchObjectDetector)

__all__ = (
    "JaticPyTorchObjectDetector",
    "JaticPyTorchObjectDetectionOutput",
    "SUPPORTED_DETECTORS",
    "COCO_YOLO_LABELS",
    "COCO_FASTER_RCNN_LABELS",
    "COCO_DETR_LABELS",
)
