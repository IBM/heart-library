"""
Module providing evasion attacks
"""

from heart_library.attacks.evasion.laser_attack import HeartLaserBeamAttack
from heart_library.attacks.evasion.query_efficient_bb_attack import \
    HeartQueryEfficientBlackBoxAttack

__all__ = (
    "HeartQueryEfficientBlackBoxAttack",
    "HeartLaserBeamAttack",
)
