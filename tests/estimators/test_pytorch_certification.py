# MIT License
#
# Copyright (C) The Hardened Extension Adversarial Robustness Toolbox (HEART) Authors 2025
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

import importlib
import logging

import pytest
from art.utils import load_dataset

from tests.utils import HEARTTestError, get_cifar10_image_classifier_pt

logger = logging.getLogger(__name__)


@pytest.mark.optional
@pytest.mark.skipif(importlib.util.find_spec("timm") is None, reason="timm is not installed")
def test_jatic_support_certification(heart_warning):
    try:
        import maite.protocols.image_classification as ic
        import torch

        from heart_library.estimators.classification.certification.derandomized_smoothing import (
            DRSJaticPyTorchClassifier,
        )

        cjptc = get_cifar10_image_classifier_pt(is_certified=True)

        assert isinstance(cjptc, ic.Model)

        import numpy as np

        (x_train, _), (_, _), _, _ = load_dataset("cifar10")

        img = x_train[[1]].transpose(0, 3, 1, 2).astype("float32")
        output = cjptc(img)
        assert np.argmax(output) == 9

        cjptc.apply_defense(training_data={"images": img, "labels": np.array([9])}, nb_epochs=1, update_batchnorm=True)
        output = cjptc(img)
        assert np.argmax(output) == 9

        # test creation of classifier using timm
        cjptc = DRSJaticPyTorchClassifier(
            model="vit_small_patch16_224",
            loss=torch.nn.CrossEntropyLoss(),
            optimizer=torch.optim.Adam,
            optimizer_params={"lr": 1e-4},
            input_shape=(3, 32, 32),
            nb_classes=10,
            clip_values=(0, 1),
            ablation_size=32,
            replace_last_layer=True,
            load_pretrained=True,
        )
        assert isinstance(cjptc, ic.Model)

    except HEARTTestError as e:
        heart_warning(e)
