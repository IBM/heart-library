"""Module providing catalogue workflows."""

from heart_library.catalogue.workflows.create.object_detection_patch import CreateObjectDetectionPatch
from heart_library.catalogue.workflows.evaluate.patch import EvaluateObjectDetectionPatch
from heart_library.catalogue.workflows.optimize.patch import OptimizeObjectDetectionPatch
from heart_library.catalogue.workflows.search.patch import SearchPatches

__all__ = (
    "CreateObjectDetectionPatch",
    "EvaluateObjectDetectionPatch",
    "OptimizeObjectDetectionPatch",
    "SearchPatches",
)
