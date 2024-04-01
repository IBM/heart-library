# MIT License
#
# Copyright (C) HEART Authors 2024
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
This module implements a HEART compatible ART Query Efficient Black Box attack
"""
from typing import Optional, Union

import numpy as np
from art.attacks import EvasionAttack
from art.attacks.evasion import FastGradientMethod
from art.estimators import BaseEstimator, LossGradientsMixin

from heart_library.estimators.classification import HeartQueryEfficientGradientEstimationClassifier


class HeartQueryEfficientBlackBoxAttack(EvasionAttack):
    """
    HEART defined extension of ART core Query Efficient Black Box attack.
    """

    attack_params = EvasionAttack.attack_params + [
        "norm",
        "eps",
        "eps_step",
        "targeted",
        "num_random_init",
        "batch_size",
        "minimal",
        "summary_writer",
    ]
    _estimator_requirements = (BaseEstimator, LossGradientsMixin)

    def __init__(
        self,
        estimator,
        num_basis: int = 20,
        sigma: float = 1 / 64.0,
        round_samples: float = 0.0,
        norm: Union[int, float, str] = np.inf,
        eps: Union[int, float, np.ndarray] = 0.3,
        eps_step: Union[int, float, np.ndarray] = 0.1,
        targeted: bool = False,
        num_random_init: int = 0,
        batch_size: int = 32,
        minimal: bool = False,
        **kwargs
    ):
        super().__init__(estimator=estimator, **kwargs)

        query_efficient_estimator = HeartQueryEfficientGradientEstimationClassifier(
            estimator, num_basis, sigma, round_samples
        )
        self._attack = FastGradientMethod(
            query_efficient_estimator,
            norm=norm,
            eps=eps,
            eps_step=eps_step,
            targeted=targeted,
            num_random_init=num_random_init,
            batch_size=batch_size,
            minimal=minimal,
        )

    def generate(self, x: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        return self._attack.generate(x)
