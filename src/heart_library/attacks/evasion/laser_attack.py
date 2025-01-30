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
This module extends ART's `LaserAttack` attack to support HEART.

| Paper link: https://arxiv.org/abs/2103.06504
"""

import logging
from collections.abc import Callable
from typing import Any, Optional, Union

import numpy as np
from art.attacks.evasion.laser_attack.laser_attack import LaserAttack, LaserBeam, LaserBeamGenerator
from art.attacks.evasion.laser_attack.utils import AdversarialObject, DebugInfo, ImageGenerator
from art.summary_writer import SummaryWriter
from numpy.typing import NDArray

logger: logging.Logger = logging.getLogger(__name__)


def _greedy_search(
    image: NDArray[np.float32],
    estimator: Any,  # noqa ANN401
    iterations: int,
    actual_class: int,
    actual_class_confidence: float,
    adv_object_generator: Any,  # noqa ANN401
    image_generator: Any,  # noqa ANN401
    debug: Optional[Any] = None,  # noqa ANN401
) -> tuple[Optional[Any], Optional[int]]:
    """Extending ART's greedy search algorithm to support HEART. Specifically supports channel first images.

    Args:
        image (NDArray[np.float32]): Image to attack.
        estimator (Any): Predictor of the image class.
        iterations (int): Maximum number of iterations of the algorithm.
        actual_class (int):
        actual_class_confidence (float):
        adv_object_generator (Any): Object responsible for adversarial object generation.
        image_generator (Any): Object responsible for image generation.
        debug (Optional[Any], optional): Optional debug handler. Defaults to None.

    Returns:
        Tuple[Optional[Any], Optional[int]]: None, None.
    """
    params = adv_object_generator.random()
    for _ in range(iterations):
        predicted_class: int = -1
        for sign in [-1, 1]:
            params_prim = adv_object_generator.update_params(params, sign=sign)
            adversarial_image = image_generator.update_image(image, params_prim)

            # channels first
            adversarial_image = adversarial_image.transpose(0, 3, 1, 2).astype(np.float32)

            prediction = estimator.predict(adversarial_image)
            __debug_report(debug, params_prim, adversarial_image)
            predicted_class = prediction.argmax()
            confidence_adv = prediction[0][actual_class]
            if confidence_adv <= actual_class_confidence:
                params = params_prim
                actual_class_confidence = confidence_adv
                break

        if predicted_class != actual_class:
            return params, predicted_class
    return None, None


def __debug_report(
    debug: Optional[Any],  # noqa ANN401
    params_prim: AdversarialObject,
    adversarial_image: NDArray[np.float32],
) -> None:
    """Log info and save image in the preset directory, based on the debug handler.

    Args:
        debug (Optional[Any]): Optional debug handler.
        params_prim (AdversarialObject): Object that will be printed out.
        adversarial_image (NDArray[np.float32]): Image to save.
    """
    if debug is not None:
        DebugInfo.report(debug, params_prim, np.squeeze(adversarial_image, 0))


class HeartLaserAttack(LaserAttack):
    """Extension of ART's implementation of a generic laser attack case which
    supports channel first images.

    Args:
        LaserAttack (LaserAttack): Generic laser attack case.

    Examples
    --------

    We can create a HeartLaserAttack by defining the image data, model parameters, and attack specification:

    >>> from torchvision.models import resnet18, ResNet18_Weights
    >>> from heart_library.estimators.classification.pytorch import JaticPyTorchClassifier
    >>> import torch
    >>> from datasets import load_dataset
    >>> from heart_library.attacks.evasion import HeartLaserBeamAttack
    >>> from heart_library.attacks.attack import JaticAttack
    >>> from art.attacks.evasion.laser_attack.laser_attack import LaserBeamGenerator, LaserBeam

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

    Define the HeartLaserAttack, wrap in HEART attack class and execute:

    >>> laser_min = LaserBeam.from_array([380, 0, 0, 0])
    >>> laser_max = LaserBeam.from_array([780, 3.14, 32, 32])
    >>> laser_generator = LaserBeamGenerator(laser_min, laser_max)
    >>> laser_attack = HeartLaserAttack(jptc, 5, laser_generator=laser_generator, random_initializations=10)
    >>> attack = JaticAttack(laser_attack, norm=2)

    Generate adversarial images:

    >>> x_adv, y, metadata = attack(data=data)
    """

    def __init__(
        self,
        estimator: Any,  # noqa ANN401
        iterations: int,
        laser_generator: Any,  # noqa ANN401
        image_generator: Any = ImageGenerator(),  # noqa B008
        random_initializations: int = 1,
        optimisation_algorithm: Callable = _greedy_search,
        debug: Optional[Any] = None,  # noqa ANN401
    ) -> None:
        """HeartLaserAttack Initialization

        Args:
            estimator (Any): Predictor of the image class.
            iterations (int): Maximum number of iterations of the algorithm.
            laser_generator (Any): Object responsible for generation laser beams images and their update.
            image_generator (Any, optional): Object responsible for image generation. Defaults to ImageGenerator().
            random_initializations (int, optional): How many times repeat the attack. Defaults to 1.
            optimisation_algorithm (Callable, optional): Algorithm used to generate adversarial example.
                May be replaced. Defaults to greedy_search.
            debug (Optional[Any], optional): Optional debug handler. Defaults to None.
        """
        super().__init__(
            estimator=estimator,
            iterations=iterations,
            laser_generator=laser_generator,
            image_generator=image_generator,
            random_initializations=random_initializations,
            optimisation_algorithm=optimisation_algorithm,
            debug=debug,
        )

    def generate(
        self,
        x: NDArray[np.float32],
        y: Optional[NDArray[np.float32]] = None,
        **kwargs: Any,  # noqa: ARG002 ANN401
    ) -> NDArray[np.float32]:
        """Generate adversarial examples.

        Args:
            x (NDArray[np.float32]): Images to attack as a tensor in NHWC order.
            y (Optional[NDArray[np.float32]], optional): Array of correct classes. Defaults to None.

        Raises:
            ValueError: If input dimension is unrecognized, != 4.

        Returns:
            NDArray[np.float32]: Array of adversarial images.
        """

        if x.ndim != 4:  # pragma: no cover
            raise ValueError("Unrecognized input dimension. Only tensors NHWC are acceptable.")

        # channels first
        x = x.transpose(0, 2, 3, 1)

        parameters = self.generate_parameters(x, y)
        adversarial_images = np.zeros_like(x)
        for image_index in range(x.shape[0]):
            laser_params, _ = parameters[image_index]
            if laser_params is None:
                adversarial_images[image_index] = x[image_index]
                continue
            adversarial_image = self._image_generator.update_image(x[image_index], laser_params)
            adversarial_images[image_index] = adversarial_image

        return adversarial_images.transpose(0, 3, 1, 2)

    def _generate_params_for_single_input(
        self,
        x: np.ndarray,
        y: Optional[int] = None,
    ) -> tuple[Optional[AdversarialObject], Optional[int]]:  # pragma: no cover
        """Generate adversarial example params for a single image.

        Args:
            x (np.ndarray): Image to attack as a tensor (NRGB = (1, ...))
            y (Optional[int], optional): Correct class of the image.
                If not provided, it is set to the prediction of the model. Defaults to None.

        Returns:
            Tuple[Optional[AdversarialObject], Optional[int]]: Adversarial object params and adversarial class number.
        """

        image = np.expand_dims(x, 0)

        # channels first
        prediction = self.estimator.predict(image.transpose(0, 3, 1, 2))

        actual_class = y if y is not None else prediction.argmax()
        actual_class_confidence = prediction[0][actual_class]

        for _ in range(self.random_initializations):
            laser_params, predicted_class = self._attack_single_image(image, actual_class, actual_class_confidence)
            if laser_params is not None:
                logger.info("Found adversarial params: %s", laser_params)
                return laser_params, predicted_class
        logger.warning("Couldn't find adversarial laser parameters")

        return None, None

    def _check_params(self) -> None:  # pragma: no cover
        """Remove restriction against channel first images.

        Raises:
            ValueError: If summary writer is not type bool or str.
            ValueError: If the number of iterations <= 0.
            ValueError: If the random initializations <= 0.
        """

        if not isinstance(self._summary_writer_arg, (bool, str, SummaryWriter)):
            raise ValueError("The argument `summary_writer` has to be either of type bool or str.")
        if self.iterations <= 0:
            raise ValueError("The iterations number has to be positive.")
        if self.random_initializations <= 0:
            raise ValueError("The random initializations has to be positive.")


class HeartLaserBeamAttack(HeartLaserAttack):
    """Extension of ART's implementation of the `LaserBeam` attack, which
    supports channel first images.

    Args:
        HeartLaserAttack (HeartLaserAttack): HEART Laserbeam attack.

    | Paper link: https://arxiv.org/abs/2103.06504

    Examples
    --------

    We can create a HeartLaserBeamAttack by defining the image data, model parameters, and attack specification:

    >>> from torchvision.models import resnet18, ResNet18_Weights
    >>> from heart_library.estimators.classification.pytorch import JaticPyTorchClassifier
    >>> import torch
    >>> from datasets import load_dataset
    >>> from heart_library.attacks.evasion import HeartLaserBeamAttack
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

    Define the HeartLaserBeamAttack, wrap in HEART attack class and execute:

    >>> laser_attack = HeartLaserBeamAttack(jptc, 5, max_laser_beam=(580, 3.14, 100, 100), random_initializations=10)
    >>> attack = JaticAttack(laser_attack, norm=2)

    Generate adversarial images:

    >>> x_adv, y, metadata = attack(data=data)
    >>> x_adv[0][0][0][0][0]
    1.0
    """

    def __init__(
        self,
        estimator: Any,  # noqa ANN401
        iterations: int,
        max_laser_beam: Union[Any, tuple[float, float, float, int]],  # noqa ANN401
        min_laser_beam: Union[Any, tuple[float, float, float, int]] = (380.0, 0.0, 1.0, 1),  # noqa ANN401
        random_initializations: int = 1,
        image_generator: Any = ImageGenerator(),  # noqa B008
        debug: Optional[Any] = None,  # noqa ANN401
    ) -> None:
        """HeartLaserBeamAttack initialization.

        Args:
            estimator (Any): Predictor of the image class.
            iterations (int): Maximum number of iterations of the algorithm.
            max_laser_beam (Union[Any, Tuple[float, float, float, int]]): LaserBeam with maximal parameters or tuple
                (wavelength, angle::radians, bias, width) of the laser parameters.
            min_laser_beam (Union[Any, Tuple[float, float, float, int]], optional):
                LaserBeam with minimal parameters or tuple (wavelength, angle::radians, bias, width)
                of the laser parameters. Defaults to (380.0, 0.0, 1.0, 1).
            random_initializations (int, optional): How many times repeat the attack.. Defaults to 1.
            image_generator (Any, optional): Object responsible for image generation. Defaults to ImageGenerator().
            debug (Optional[Any], optional): Optional debug handler. Defaults to None.
        """

        if isinstance(min_laser_beam, tuple):
            min_laser_beam_obj = LaserBeam.from_array(list(min_laser_beam))
        else:
            min_laser_beam_obj = min_laser_beam
        if isinstance(max_laser_beam, tuple):
            max_laser_beam_obj = LaserBeam.from_array(list(max_laser_beam))
        else:
            max_laser_beam_obj = max_laser_beam

        super().__init__(
            estimator,
            iterations,
            LaserBeamGenerator(min_laser_beam_obj, max_laser_beam_obj),
            image_generator=image_generator,
            random_initializations=random_initializations,
            debug=debug,
        )
