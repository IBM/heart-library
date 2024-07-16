"""
Module providing object detection estimators
"""

from heart_library.estimators.object_detection.pytorch_detr import (
    JaticPyTorchDETR, JaticPyTorchObjectDetectionOutput)
from heart_library.estimators.object_detection.pytorch_faster_rcnn import \
    JaticPyTorchFasterRCNN

__all__ = ("JaticPyTorchObjectDetectionOutput", "JaticPyTorchDETR", "JaticPyTorchFasterRCNN")
