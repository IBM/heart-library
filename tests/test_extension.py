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

import logging

import pytest

from tests.utils import HEARTTestError, get_mnist_image_classifier_pt

logger = logging.getLogger(__name__)


@pytest.mark.required
@pytest.mark.framework_agnostic
def test_import_art(heart_warning):
    try:
        import art

        assert art.__version__ >= "1.17.1"

    except HEARTTestError as e:
        heart_warning(e)


@pytest.mark.required
@pytest.mark.framework_agnostic
def test_import_heart(heart_warning):
    try:
        import heart_library

        assert heart_library.__version__ == "0.7.0"

    except HEARTTestError as e:
        heart_warning(e)


@pytest.fixture
def fix_get_mnist_subset(get_mnist_dataset):
    (x_train_mnist, y_train_mnist), (x_test_mnist, y_test_mnist) = get_mnist_dataset
    n_train = 100
    n_test = 11
    return x_train_mnist[:n_train], y_train_mnist[:n_train], x_test_mnist[:n_test], y_test_mnist[:n_test]


@pytest.mark.required
def test_heart_classifier_ext(fix_get_mnist_subset):
    (x_train_mnist, _, _, _) = fix_get_mnist_subset
    classifier = get_mnist_image_classifier_pt()
    scores = classifier.predict(x_train_mnist)
    assert scores[0][0] == pytest.approx(0.11191566, 0.001)
