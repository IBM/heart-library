# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (HEART) Authors 2024
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

import maite.protocols.image_classification as ic
import pytest
from art.utils import load_dataset

from heart_library.estimators.classification.pytorch import JaticPyTorchClassifier
from tests.utils import HEARTTestError, get_cifar10_image_classifier_pt

logger = logging.getLogger(__name__)


@pytest.mark.required
def test_jatic_support_classification(heart_warning):
    try:
        jptc = get_cifar10_image_classifier_pt()

        assert isinstance(jptc, ic.Model)

        import numpy as np

        (x_train, _), (_, _), _, _ = load_dataset("cifar10")

        img = x_train[[1]].transpose(0, 3, 1, 2).astype("float32")
        output = jptc(img)

        assert np.argmax(output) == 9

    except HEARTTestError as e:
        heart_warning(e)


@pytest.mark.required
def test_jatic_support_classification_using_hf(heart_warning):
    try:
        import torch

        # test creation of classifier using HuggingFace
        loss_fn = torch.nn.CrossEntropyLoss(reduction="sum")
        jptc = JaticPyTorchClassifier(
            model="nateraw/vit-base-patch16-224-cifar10",
            loss=loss_fn,
            input_shape=(3, 224, 224),
            nb_classes=10,
            clip_values=(0, 1),
            preprocessing=(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225],
            ),
            provider="huggingface",
        )

        assert isinstance(jptc, ic.Model)

    except HEARTTestError as e:
        heart_warning(e)


@pytest.mark.optional
@pytest.mark.skipif(importlib.util.find_spec("timm") is None, reason="timm is not installed")
def test_jatic_support_classification_using_timm(heart_warning):
    try:
        import torch

        # test creation of classifier using timm
        loss_fn = torch.nn.CrossEntropyLoss(reduction="sum")
        jptc = JaticPyTorchClassifier(
            model="resnet50.a1_in1k",
            loss=loss_fn,
            input_shape=(3, 224, 224),
            nb_classes=10,
            clip_values=(0, 1),
            preprocessing=([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            provider="timm",
        )
        assert isinstance(jptc, ic.Model)

    except HEARTTestError as e:
        heart_warning(e)
