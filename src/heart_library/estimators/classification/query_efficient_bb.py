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
"""This module implements a HEART compatible ART QueryEfficientGradientEstimationClassifier."""

from typing import Any

import numpy as np
from art.estimators.classification import QueryEfficientGradientEstimationClassifier
from art.utils import clip_and_round
from numpy.typing import NDArray


class HeartQueryEfficientGradientEstimationClassifier(
    QueryEfficientGradientEstimationClassifier,
):
    """HEART compatible extension of ART core QueryEfficientGradientEstimationClassifier

    Args:
        QueryEfficientGradientEstimationClassifier (QueryEfficientGradientEstimationClassifier):
            ART QueryEfficientGradientEstimationClassifier.

    Examples
    --------

    We can create a default HeartQueryEfficientGradientEstimationClassifier and pass in sample data for prediction:

    >>> from torchvision.models import resnet18, ResNet18_Weights
    >>> from heart_library.estimators.classification.pytorch import JaticPyTorchClassifier
    >>> from heart_library.estimators.classification.query_efficient_bb \
        import HeartQueryEfficientGradientEstimationClassifier
    >>> import torch
    >>> import numpy

    Define the JaticPyTorchClassifier, in this case passing in a resnet model:

    >>> model = resnet18(ResNet18_Weights)
    >>> loss_fn = torch.nn.CrossEntropyLoss(reduction="sum")
    >>> optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    >>> jptc = JaticPyTorchClassifier(
    ...    model=model, loss=loss_fn, optimizer=optimizer, input_shape=(3, 32, 32), nb_classes=10, clip_values=(0, 255),
    ...    preprocessing=(0.0, 255)
    ... )

    Define the HeartQueryEfficientGradientEstimationClassifier by passing in the JaticPyTorchClassifier.

    >>> hqegec = HeartQueryEfficientGradientEstimationClassifier(
    ...    classifier = jptc
    ... )

    Define the data in expected format of 4-d and 3 channels for prediction.

    >>> arr = np.zeros((3,3,4,2))
    >>> hqegec.predict(arr)[0][0]
    0.57479644
    """

    def __init__(
        self,
        classifier: Any,  # noqa ANN401
        num_basis: int = 20,
        sigma: float = 1 / 64.0,
        round_samples: float = 0.0,
    ) -> None:
        """HeartQueryEfficientGradientEstimationClassifier initialization.

        Args:
            classifier (Any): An instance of a classification estimator whose loss_gradient is being approximated.
            num_basis (int, optional): The number of samples to draw to approximate the gradient. Defaults to 20.
            sigma (float, optional): Scaling on the Gaussian noise N(0,1). Defaults to 1/64.0.
            round_samples (float, optional): The resolution of the input domain to round the data to,
                e.g., 1.0, or 1/255. Set to 0 to disable. Defaults to 0.0.
        """
        super().__init__(classifier=classifier, num_basis=num_basis, sigma=sigma, round_samples=round_samples)

    def predict(
        self,
        x: NDArray[np.float32],
        batch_size: int = 128,
        **kwargs: Any,  # noqa ARG002
    ) -> NDArray[np.float32]:
        """Perform prediction of the classifier for input `x`. Rounds results first.

        Args:
            x (NDArray[np.float32]): Features in array of shape (nb_samples, nb_features) or
                (nb_samples, nb_pixels_1, nb_pixels_2, nb_channels) or
                (nb_samples, nb_channels, nb_pixels_1, nb_pixels_2).
            batch_size (int, optional): Size of batches. Defaults to 128.

        Returns:
            NDArray[np.float32]: Array of predictions of shape `(nb_inputs, nb_classes)`.
        """
        x = clip_and_round(x, self.clip_values, self.round_samples).astype(np.float32)
        return self._classifier.predict(x, batch_size=batch_size)
