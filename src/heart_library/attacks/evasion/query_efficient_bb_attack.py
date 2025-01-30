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
"""This module implements a HEART compatible ART Query Efficient Black Box attack"""

from typing import Any, Optional, Union

import numpy as np
from art.attacks import EvasionAttack
from art.attacks.evasion import FastGradientMethod
from art.estimators import BaseEstimator, LossGradientsMixin
from numpy.typing import NDArray

from heart_library.estimators.classification import HeartQueryEfficientGradientEstimationClassifier


class HeartQueryEfficientBlackBoxAttack(EvasionAttack):
    """HEART defined extension of ART core Query Efficient Black Box attack.

    Args:
        EvasionAttack (EvasionAttack): ART core Query Efficient Black Box attack.

    Examples
    --------

    We can create a QueryEfficientBlackBoxAttack by defining the image data, model parameters, and attack specification:

    >>> from torchvision.models import resnet18, ResNet18_Weights
    >>> from heart_library.estimators.classification.pytorch import JaticPyTorchClassifier
    >>> import torch
    >>> from datasets import load_dataset
    >>> from heart_library.attacks.evasion.query_efficient_bb_attack import HeartQueryEfficientBlackBoxAttack
    >>> from heart_library.attacks.attack import JaticAttack

    Define the JaticPyTorchClassifier inputs, in this case for image classification:

    >>> data = load_dataset("cifar10", split="test[0:10]")
    >>> model = resnet18(ResNet18_Weights)
    >>> loss_fn = torch.nn.CrossEntropyLoss(reduction="sum")
    >>> optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    >>> jptc = JaticPyTorchClassifier(
    ...     model=model,
    ...     loss=loss_fn,
    ...     optimizer=optimizer,
    ...     input_shape=(3, 32, 32),
    ...     nb_classes=10,
    ...     clip_values=(0, 255),
    ...     preprocessing=(0.0, 255),
    ... )

    Define the HeartQueryEfficientBlackBoxAttack, wrap in HEART attack class and execute:

    >>> query_attack = HeartQueryEfficientBlackBoxAttack(estimator=jptc, eps=0.2)
    >>> attack = JaticAttack(query_attack, norm=2)

    Generate adversarial images:

    >>> x_adv, y, metadata = attack(data=data)
    >>> x_adv[0][0][0][0][0]
    158.0
    """

    attack_params: list[str] = EvasionAttack.attack_params + [
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
        estimator: Any,  # noqa ANN401
        num_basis: int = 20,
        sigma: float = 1 / 64.0,
        round_samples: float = 0.0,
        norm: Union[float, str] = np.inf,
        eps: Union[float, NDArray[np.float32]] = 0.3,
        eps_step: Union[float, NDArray[np.float32]] = 0.1,
        targeted: bool = False,
        num_random_init: int = 0,
        batch_size: int = 32,
        minimal: bool = False,
        **kwargs: Any,  # noqa ANN401
    ) -> None:
        """HeartQueryEfficientBlackBoxAttack initialization.

        Args:
            estimator (Any): An estimator.
            num_basis (int, optional): The number of samples to draw to approximate the gradient. Defaults to 20.
            sigma (float, optional): Standard deviation random Gaussian Noise. Defaults to 1/64.0.
            round_samples (float, optional):
                The resolution of the input domain to round the data to, e.g., 1.0, or 1/255. Defaults to 0.0.
            norm (Union[int, float, str], optional): Order of the norm. Possible values: "inf", np.inf or 2.
                Defaults to np.inf.
            eps (Union[int, float, NDArray[np.float32]], optional):
                Maximum perturbation that the attacker can introduce. Defaults to 0.3.
            eps_step (Union[int, float, NDArray[np.float32]], optional):
                Attack step size (input variation) at each iteration. Defaults to 0.1.
            targeted (bool, optional): Indicates whether the attack is targeted (True) or untargeted (False).
                Defaults to False.
            num_random_init (int, optional): Number of random initialisations within the epsilon ball.
                For num_random_init=0 starting at the original input. Defaults to 0.
            batch_size (int, optional): Batch size. Defaults to 32.
            minimal (bool, optional): Indicates if computing the minimal perturbation (True). Defaults to False.
        """
        super().__init__(estimator=estimator, **kwargs)

        query_efficient_estimator = HeartQueryEfficientGradientEstimationClassifier(
            estimator,
            num_basis,
            sigma,
            round_samples,
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

    def generate(
        self,
        x: NDArray[np.float32],
        y: Optional[NDArray[np.float32]] = None,  # noqa: ARG002
        **kwargs: Any,  # noqa: ARG002 ANN401
    ) -> NDArray[np.float32]:
        """Generate adversarial examples and return them as an array.

        Args:
            x (NDArray[np.float32]): An array with the original inputs to be attacked.
            y (Optional[NDArray[np.float32]], optional):
                Correct labels or target labels for `x`, depending on if the attack is targeted or not.
                This parameter is only used by some of the attacks. Defaults to None.

        Returns:
            NDArray[np.float32]: An array holding the adversarial examples.
        """
        return self._attack.generate(x)
