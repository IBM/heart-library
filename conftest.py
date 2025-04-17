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
import logging
import warnings

import pytest
import numpy as np

from tests.utils import (
    HEARTTestFixtureNotImplementedError,
    load_dataset,
    master_seed,
)

logger = logging.getLogger(__name__)

deep_learning_frameworks = [
    "pytorch",
]

heart_supported_frameworks = []
heart_supported_frameworks.extend(deep_learning_frameworks)

master_seed(1234)


def get_default_framework():
    return "pytorch"


def pytest_addoption(parser):
    parser.addoption(
        "--framework",
        action="store",
        default=get_default_framework(),
        help="HEART tests allow you to specify which framework to use. The default framework used is `tensorflow`. "
        "Other options available are {0}".format(heart_supported_frameworks),
    )


@pytest.fixture(scope="session")
def framework(request):
    ml_framework = request.config.getoption("--framework")

    if ml_framework not in heart_supported_frameworks:
        raise Exception(
            "framework value {0} is unsupported. Please use one of these valid values: {1}".format(
                ml_framework, " ".join(heart_supported_frameworks)
            )
        )
    return ml_framework


@pytest.fixture(autouse=True)
def framework_agnostic(request, framework):
    if request.node.get_closest_marker("framework_agnostic"):
        if framework != get_default_framework():
            pytest.skip("framework agnostic test skipped for framework : {}".format(framework))


@pytest.fixture
def heart_warning(request):
    def _heart_warning(exception):
        if type(exception) is HEARTTestFixtureNotImplementedError:
            if request.node.get_closest_marker("framework_agnostic"):
                if not request.node.get_closest_marker("parametrize"):
                    raise Exception(
                        "This test has marker framework_agnostic decorator which means it will only be ran "
                        "once. However the HEART test exception was thrown, hence it is never run fully. "
                    )

            # NotImplementedErrors are raised in HEART whenever a test model does not exist for a specific
            # model/framework combination. By catching there here, we can provide a report at the end of each
            # pytest run list all models requiring to be implemented.
            warnings.warn(UserWarning(exception))
        else:
            raise exception

    return _heart_warning


@pytest.fixture(scope="session")
def load_mnist_dataset():
    logging.info("Loading mnist")
    (x_train_mnist, y_train_mnist), (x_test_mnist, y_test_mnist), _, _ = load_dataset("mnist")
    yield (x_train_mnist, y_train_mnist), (x_test_mnist, y_test_mnist)


@pytest.fixture(scope="function")
def get_mnist_dataset(load_mnist_dataset, mnist_shape):
    (x_train_mnist, y_train_mnist), (x_test_mnist, y_test_mnist) = load_mnist_dataset

    x_train_mnist = np.reshape(x_train_mnist, (x_train_mnist.shape[0],) + mnist_shape).astype(np.float32)
    x_test_mnist = np.reshape(x_test_mnist, (x_test_mnist.shape[0],) + mnist_shape).astype(np.float32)

    yield (x_train_mnist, y_train_mnist), (x_test_mnist, y_test_mnist)


@pytest.fixture()
def mnist_shape(framework):
    if framework == "pytorch":
        return (1, 28, 28)
    else:
        return (28, 28, 1)
