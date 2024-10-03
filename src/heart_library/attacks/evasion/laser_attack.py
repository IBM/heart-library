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
from typing import Callable, Optional, Tuple, Union

import numpy as np
from art.attacks.evasion.laser_attack.laser_attack import (AdvObjectGenerator,
                                                           LaserAttack,
                                                           LaserBeam,
                                                           LaserBeamGenerator)
from art.attacks.evasion.laser_attack.utils import (AdversarialObject,
                                                    DebugInfo, ImageGenerator)
from art.summary_writer import SummaryWriter

logger = logging.getLogger(__name__)


def greedy_search(
    image: np.ndarray,
    estimator,
    iterations: int,
    actual_class: int,
    actual_class_confidence: float,
    adv_object_generator: AdvObjectGenerator,
    image_generator: ImageGenerator,
    debug: Optional[DebugInfo] = None,
) -> Tuple[Optional[AdversarialObject], Optional[int]]:  # pragma: no cover
    """
    Extending ART's greedy search algorithm to support HEART. Specifically supports channel first images.

    :param image: Image to attack.
    :param estimator: Predictor of the image class.
    :param iterations: Maximum number of iterations of the algorithm.
    :param actual_class:
    :param actual_class_confidence:
    :param adv_object_generator: Object responsible for adversarial object generation.
    :param image_generator: Object responsible for image generation.
    :param debug: Optional debug handler.
    """
    params = adv_object_generator.random()
    for _ in range(iterations):
        for sign in [-1, 1]:
            params_prim = adv_object_generator.update_params(params, sign=sign)
            adversarial_image = image_generator.update_image(image, params_prim)

            # channels first
            adversarial_image = adversarial_image.transpose(0, 3, 1, 2).astype(np.float32)

            prediction = estimator.predict(adversarial_image)
            if debug is not None:
                DebugInfo.report(debug, params_prim, np.squeeze(adversarial_image, 0))
            predicted_class = prediction.argmax()
            confidence_adv = prediction[0][actual_class]
            if confidence_adv <= actual_class_confidence:
                params = params_prim
                actual_class_confidence = confidence_adv
                break

        if predicted_class != actual_class:
            return params, predicted_class
    return None, None


class HeartLaserAttack(LaserAttack):
    """
    Extension of ART's implementation of a generic laser attack case which
    supports channel first images.
    """

    def __init__(
        self,
        estimator,
        iterations: int,
        laser_generator: AdvObjectGenerator,
        image_generator: ImageGenerator = ImageGenerator(),
        random_initializations: int = 1,
        optimisation_algorithm: Callable = greedy_search,
        debug: Optional[DebugInfo] = None,
    ) -> None:
        """
        :param estimator: Predictor of the image class.
        :param iterations: Maximum number of iterations of the algorithm.
        :param laser_generator: Object responsible for generation laser beams images and their update.
        :param image_generator: Object responsible for image generation.
        :param random_initializations: How many times repeat the attack.
        :param optimisation_algorithm: Algorithm used to generate adversarial example. May be replaced.
        :param debug: Optional debug handler.
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

    def generate(self, x: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        """
        Generate adversarial examples.

        :param x: Images to attack as a tensor in NHWC order
        :param y: Array of correct classes
        :return: Array of adversarial images
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

        adversarial_images = adversarial_images.transpose(0, 3, 1, 2)
        return adversarial_images

    def _generate_params_for_single_input(
        self, x: np.ndarray, y: Optional[int] = None
    ) -> Tuple[Optional[AdversarialObject], Optional[int]]:  # pragma: no cover
        """
        Generate adversarial example params for a single image.

        :param x: Image to attack as a tensor (NRGB = (1, ...))
        :param y: Correct class of the image. If not provided, it is set to the prediction of the model.
        :return: Adversarial object params and adversarial class number.
        """

        image = np.expand_dims(x, 0)

        # channels first
        prediction = self.estimator.predict(image.transpose(0, 3, 1, 2))

        if y is not None:
            actual_class = y
        else:
            actual_class = prediction.argmax()
        actual_class_confidence = prediction[0][actual_class]

        for _ in range(self.random_initializations):
            laser_params, predicted_class = self._attack_single_image(image, actual_class, actual_class_confidence)
            if laser_params is not None:
                logger.info("Found adversarial params: %s", laser_params)
                return laser_params, predicted_class
        logger.warning("Couldn't find adversarial laser parameters")

        return None, None

    def _check_params(self) -> None:  # pragma: no cover
        """
        Remove restriction against channel first images.
        """
        if not isinstance(self._summary_writer_arg, (bool, str, SummaryWriter)):
            raise ValueError("The argument `summary_writer` has to be either of type bool or str.")
        if self.iterations <= 0:
            raise ValueError("The iterations number has to be positive.")
        if self.random_initializations <= 0:
            raise ValueError("The random initializations has to be positive.")


class HeartLaserBeamAttack(HeartLaserAttack):
    """
    Extension of ART's implementation of the `LaserBeam` attack, which
    supports channel first images.

    | Paper link: https://arxiv.org/abs/2103.06504
    """

    def __init__(
        self,
        estimator,
        iterations: int,
        max_laser_beam: Union[LaserBeam, Tuple[float, float, float, int]],
        min_laser_beam: Union[LaserBeam, Tuple[float, float, float, int]] = (380.0, 0.0, 1.0, 1),
        random_initializations: int = 1,
        image_generator: ImageGenerator = ImageGenerator(),
        debug: Optional[DebugInfo] = None,
    ) -> None:  # pragma: no cover
        """
        :param estimator: Predictor of the image class.
        :param iterations: Maximum number of iterations of the algorithm.
        :param max_laser_beam: LaserBeam with maximal parameters or tuple (wavelength, angle::radians, bias, width)
            of the laser parameters.
        :param min_laser_beam: LaserBeam with minimal parameters or tuple (wavelength, angle::radians, bias, width)
            of the laser parameters.
        :param image_generator: Object responsible for image generation.
        :param random_initializations: How many times repeat the attack.
        :param debug: Optional debug handler.
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
