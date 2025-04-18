"""Module providing classification estimators"""

from heart_library.estimators.classification.pytorch import JaticPyTorchClassifier
from heart_library.estimators.classification.query_efficient_bb import HeartQueryEfficientGradientEstimationClassifier

__all__ = ("HeartQueryEfficientGradientEstimationClassifier", "JaticPyTorchClassifier")
