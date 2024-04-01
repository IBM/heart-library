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
This module implements a HEART compatible ART QueryEfficientGradientEstimationClassifier.
"""
import numpy as np
from art.estimators.classification import QueryEfficientGradientEstimationClassifier
from art.utils import clip_and_round


class HeartQueryEfficientGradientEstimationClassifier(
    QueryEfficientGradientEstimationClassifier
):  # pylint: disable=R0901
    """
    HEART compatible extension of ART core QueryEfficientGradientEstimationClassifier
    """

    def __init__(
        self,
        classifier,
        num_basis: int = 20,
        sigma: float = 1 / 64.0,
        round_samples: float = 0.0,
    ):
        super().__init__(classifier=classifier, num_basis=num_basis, sigma=sigma, round_samples=round_samples)

    def predict(self, x: np.ndarray, batch_size: int = 128, **kwargs) -> np.ndarray:  # pylint: disable=W0221
        """
        Perform prediction of the classifier for input `x`. Rounds results first.

        :param x: Features in array of shape (nb_samples, nb_features) or (nb_samples, nb_pixels_1, nb_pixels_2,
                  nb_channels) or (nb_samples, nb_channels, nb_pixels_1, nb_pixels_2).
        :param batch_size: Size of batches.
        :return: Array of predictions of shape `(nb_inputs, nb_classes)`.
        """
        x = clip_and_round(x, self.clip_values, self.round_samples).astype(np.float32)
        return self._classifier.predict(x, batch_size=batch_size)
