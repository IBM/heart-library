"""This module contains utility functions to be used within unit tests."""

import logging
import os
import tempfile
import urllib.request
from typing import Any, Optional, Union

import numpy as np
import torch
from six.moves.urllib.error import HTTPError, URLError
from six.moves.urllib.request import urlretrieve
from tqdm import tqdm

from heart_library import config
from heart_library.estimators.classification.pytorch import JaticPyTorchClassifier

DATASET_TYPE = tuple[
    tuple[np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray],
    float,
    float,
]

CLIP_VALUES_TYPE = tuple[Union[int, float, np.ndarray], Union[int, float, np.ndarray]]

logger = logging.getLogger(__name__)


class HEARTTestError(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)


class HEARTTestFixtureNotImplementedError(HEARTTestError):
    def __init__(self, message: str, fixture_name: str, framework: str, parameters_dict: str = "") -> None:
        super().__init__(
            f"Could NOT run test for framework: {framework} due to fixture: {fixture_name}. Message was: '"
            f"{message}' for the following parameters: {parameters_dict}",
        )


def get_mnist_image_classifier_pt(
    from_logits: bool = False,
    load_init: bool = True,
    use_maxpool: bool = True,
) -> JaticPyTorchClassifier:
    """
    Standard PyTorch classifier for unit testing.

    :param from_logits: Flag if model should predict logits (True) or probabilities (False).
    :type from_logits: `bool`
    :param load_init: Load the initial weights if True.
    :type load_init: `bool`
    :param use_maxpool: If to use a classifier with maxpool or not
    :type use_maxpool: `bool`
    :return: PyTorchClassifier
    """

    # Define the network
    model_class = _use_maxpool(from_logits=from_logits, load_init=load_init) if use_maxpool else _without_maxpool()
    model = model_class()
    # Define a loss function and optimizer
    loss_fn = torch.nn.CrossEntropyLoss(reduction="sum")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Get classifier
    return JaticPyTorchClassifier(
        model=model,
        loss=loss_fn,
        optimizer=optimizer,
        input_shape=(1, 28, 28),
        nb_classes=10,
        clip_values=(0, 1),
    )


def _use_maxpool(from_logits: bool, load_init: bool) -> Any:  # noqa ANN401
    """Create model for pytorch with maxpool.

    Args:
        from_logits (bool): Flag if model should predict logits (True) or probabilities (False).
        load_init (bool): Load the initial weights if True.

    Returns:
        Any: PyTorch classifier.
    """

    class Model(torch.nn.Module):
        """
        Create model for pytorch.

        The weights and biases are identical to the TensorFlow model in get_classifier_tf().
        """

        def __init__(self) -> None:
            super().__init__()

            self.conv = torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=7)
            self.relu = torch.nn.ReLU()
            self.pool = torch.nn.MaxPool2d(4, 4)
            self.fullyconnected = torch.nn.Linear(25, 10)

            if load_init:
                w_conv2d = np.load(
                    os.path.join(
                        os.path.dirname(os.path.dirname(__file__)),
                        "utils/resources/models",
                        "W_CONV2D_MNIST.npy",
                    ),
                )
                b_conv2d = np.load(
                    os.path.join(
                        os.path.dirname(os.path.dirname(__file__)),
                        "utils/resources/models",
                        "B_CONV2D_MNIST.npy",
                    ),
                )
                w_dense = np.load(
                    os.path.join(
                        os.path.dirname(os.path.dirname(__file__)),
                        "utils/resources/models",
                        "W_DENSE_MNIST.npy",
                    ),
                )
                b_dense = np.load(
                    os.path.join(
                        os.path.dirname(os.path.dirname(__file__)),
                        "utils/resources/models",
                        "B_DENSE_MNIST.npy",
                    ),
                )

                w_conv2d_pt = w_conv2d.reshape((1, 1, 7, 7))

                self.conv.weight = torch.nn.Parameter(torch.Tensor(w_conv2d_pt))
                self.conv.bias = torch.nn.Parameter(torch.Tensor(b_conv2d))
                self.fullyconnected.weight = torch.nn.Parameter(torch.Tensor(np.transpose(w_dense)))
                self.fullyconnected.bias = torch.nn.Parameter(torch.Tensor(b_dense))

        def forward(self, x: int) -> torch.Tensor:
            """
            Forward function to evaluate the model
            :param x: Input to the model
            :return: Prediction of the model
            """
            x = self.conv(x)
            x = self.relu(x)
            x = self.pool(x)
            x = x.reshape(-1, 25)
            x = self.fullyconnected(x)
            if not from_logits:
                x = torch.nn.functional.softmax(x, dim=1)
            return x

    return Model


def _without_maxpool(from_logits: bool, load_init: bool) -> Any:  # noqa ANN401
    """Create model for pytorch without maxpool.

    Args:
        from_logits (bool): Flag if model should predict logits (True) or probabilities (False).
        load_init (bool): Load the initial weights if True.

    Returns:
        Any: PyTorch classifier.
    """

    class Model(torch.nn.Module):
        """
        Create model for pytorch.
        Here the model does not use maxpooling. Needed for certification tests.
        """

        def __init__(self) -> None:
            super().__init__()

            self.conv = torch.nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=(4, 4),
                dilation=(1, 1),
                padding=(0, 0),
                stride=(3, 3),
            )

            self.fullyconnected = torch.nn.Linear(in_features=1296, out_features=10)

            self.relu = torch.nn.ReLU()

            if load_init:
                w_conv2d = np.load(
                    os.path.join(
                        os.path.dirname(os.path.dirname(__file__)),
                        "utils/resources/models",
                        "W_CONV2D_NO_MPOOL_MNIST.npy",
                    ),
                )
                b_conv2d = np.load(
                    os.path.join(
                        os.path.dirname(os.path.dirname(__file__)),
                        "utils/resources/models",
                        "B_CONV2D_NO_MPOOL_MNIST.npy",
                    ),
                )
                w_dense = np.load(
                    os.path.join(
                        os.path.dirname(os.path.dirname(__file__)),
                        "utils/resources/models",
                        "W_DENSE_NO_MPOOL_MNIST.npy",
                    ),
                )
                b_dense = np.load(
                    os.path.join(
                        os.path.dirname(os.path.dirname(__file__)),
                        "utils/resources/models",
                        "B_DENSE_NO_MPOOL_MNIST.npy",
                    ),
                )

                self.conv.weight = torch.nn.Parameter(torch.Tensor(w_conv2d))
                self.conv.bias = torch.nn.Parameter(torch.Tensor(b_conv2d))
                self.fullyconnected.weight = torch.nn.Parameter(torch.Tensor(w_dense))
                self.fullyconnected.bias = torch.nn.Parameter(torch.Tensor(b_dense))

        def forward(self, x: int) -> torch.Tensor:
            """
            Forward function to evaluate the model
            :param x: Input to the model
            :return: Prediction of the model
            """
            x = self.conv(x)
            x = self.relu(x)
            x = x.reshape(-1, 1296)
            x = self.fullyconnected(x)
            if not from_logits:
                x = torch.nn.functional.softmax(x, dim=1)
            return x

    return Model


def _load_cifar10_image_classifier_pt(from_logits=False):
    class Model(torch.nn.Module):
        """
        Create model for pytorch.
        Here the model does not use maxpooling. Needed for certification tests.
        """

        def __init__(self) -> None:
            super().__init__()

            self.conv = torch.nn.Conv2d(
                in_channels=3,
                out_channels=16,
                kernel_size=(4, 4),
                dilation=(1, 1),
                padding=(0, 0),
                stride=(3, 3),
            )

            self.fullyconnected = torch.nn.Linear(in_features=1600, out_features=10)

            self.relu = torch.nn.ReLU()

            w_conv2d = np.load(
                os.path.join(
                    os.path.dirname(os.path.dirname(__file__)),
                    "utils/resources/models",
                    "W_CONV2D_NO_MPOOL_CIFAR10.npy",
                ),
            )
            b_conv2d = np.load(
                os.path.join(
                    os.path.dirname(os.path.dirname(__file__)),
                    "utils/resources/models",
                    "B_CONV2D_NO_MPOOL_CIFAR10.npy",
                ),
            )
            w_dense = np.load(
                os.path.join(
                    os.path.dirname(os.path.dirname(__file__)),
                    "utils/resources/models",
                    "W_DENSE_NO_MPOOL_CIFAR10.npy",
                ),
            )
            b_dense = np.load(
                os.path.join(
                    os.path.dirname(os.path.dirname(__file__)),
                    "utils/resources/models",
                    "B_DENSE_NO_MPOOL_CIFAR10.npy",
                ),
            )

            self.conv.weight = torch.nn.Parameter(torch.Tensor(w_conv2d))
            self.conv.bias = torch.nn.Parameter(torch.Tensor(b_conv2d))
            self.fullyconnected.weight = torch.nn.Parameter(torch.Tensor(w_dense))
            self.fullyconnected.bias = torch.nn.Parameter(torch.Tensor(b_dense))

        def forward(self, x: int) -> torch.Tensor:
            """
            Forward function to evaluate the model
            :param x: Input to the model
            :return: Prediction of the model
            """
            x = self.conv(x)
            x = self.relu(x)
            x = x.reshape(-1, 1600)
            x = self.fullyconnected(x)
            if not from_logits:
                x = torch.nn.functional.softmax(x, dim=1)
            return x

    return Model()


def get_cifar10_image_classifier_pt(from_logits=False, is_jatic=True, is_certified=False):
    """
    Standard PyTorch classifier for unit testing.

    :param from_logits: Flag if model should predict logits (True) or probabilities (False).
    :type from_logits: `bool`
    :param load_init: Load the initial weights if True.
    :type load_init: `bool`
    :param use_maxpool: If to use a classifier with maxpool or not
    :type use_maxpool: `bool`
    :return: PyTorchClassifier
    """
    import torch
    from art.estimators.classification.pytorch import PyTorchClassifier

    from heart_library.estimators.classification.certification.derandomized_smoothing import DRSJaticPyTorchClassifier
    from heart_library.estimators.classification.pytorch import JaticPyTorchClassifier

    # Define the network
    model = _load_cifar10_image_classifier_pt(from_logits=from_logits)

    # Define a loss function and optimizer
    loss_fn = torch.nn.CrossEntropyLoss(reduction="sum")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    if is_jatic:
        if is_certified:
            return DRSJaticPyTorchClassifier(
                model=model,
                loss=loss_fn,
                optimizer=optimizer,
                input_shape=(3, 32, 32),
                nb_classes=10,
                clip_values=(0, 1),
                ablation_size=32,
            )
        return JaticPyTorchClassifier(
            model=model,
            loss=loss_fn,
            optimizer=optimizer,
            input_shape=(3, 32, 32),
            nb_classes=10,
            clip_values=(0, 1),
        )
    return PyTorchClassifier(
        model=model,
        loss=loss_fn,
        optimizer=optimizer,
        input_shape=(3, 32, 32),
        nb_classes=10,
        clip_values=(0, 1),
    )


def master_seed(
    seed: int = 1234,
    set_random: bool = True,
    set_numpy: bool = True,
    set_tensorflow: bool = False,
    set_mxnet: bool = False,
    set_torch: bool = False,
) -> None:
    """
    Set the seed for all random number generators used in the library. This ensures experiments reproducibility and
    stable testing.

    :param seed: The value to be seeded in the random number generators.
    :type seed: `int`
    :param set_random: The flag to set seed for `random`.
    :type set_random: `bool`
    :param set_numpy: The flag to set seed for `numpy`.
    :type set_numpy: `bool`
    :param set_tensorflow: The flag to set seed for `tensorflow`.
    :type set_tensorflow: `bool`
    :param set_mxnet: The flag to set seed for `mxnet`.
    :type set_mxnet: `bool`
    :param set_torch: The flag to set seed for `torch`.
    :type set_torch: `bool`
    """
    import numbers

    if not isinstance(seed, numbers.Integral):
        raise TypeError("The seed for random number generators has to be an integer.")

    for flag, func in [
        (set_random, _set_python_seed),
        (set_numpy, _set_numpy_seed),
        (set_tensorflow, _set_tensorflow_seed),
        (set_mxnet, _set_mxnet_seed),
        (set_torch, _set_pytorch_seed),
    ]:
        if flag:
            func(seed)


def _set_python_seed(seed: int) -> None:
    import random

    logger.info("Setting random seed for Python.")
    random.seed(seed)


def _set_numpy_seed(seed: int) -> None:
    logger.info("Setting random seed for Numpy.")
    np.random.default_rng(seed)
    np.random.RandomState(seed)


def _set_tensorflow_seed(seed: int) -> None:
    try:
        import tensorflow as tf

        logger.info("Setting random seed for TensorFlow.")
        if tf.__version__[0] == "2":
            tf.random.set_seed(seed)
        else:
            tf.set_random_seed(seed)
    except ImportError:
        logger.info("Could not set random seed for TensorFlow.")


def _set_mxnet_seed(seed: int) -> None:
    try:
        import mxnet as mx

        logger.info("Setting random seed for MXNet.")
        mx.random.seed(seed)
    except ImportError:
        logger.info("Could not set random seed for MXNet.")


def _set_pytorch_seed(seed: int) -> None:
    try:
        logger.info("Setting random seed for PyTorch.")
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        logger.info("Could not set random seed for PyTorch.")


def load_dataset(
    name: str,
) -> DATASET_TYPE:
    """
    Loads or downloads the dataset corresponding to `name`. Options are: `mnist`, `cifar10`, `stl10`, `iris`, `nursery`
    and `diabetes`.

    :param name: Name of the dataset.
    :return: The dataset separated in training and test sets as `(x_train, y_train), (x_test, y_test), min, max`.
    :raises NotImplementedError: If the dataset is unknown.
    """
    if "mnist" in name:
        return load_mnist()

    raise NotImplementedError(f"There is no loader for dataset '{name}'.")


def load_mnist(
    raw: bool = False,
) -> DATASET_TYPE:
    """
    Loads MNIST dataset from `config.ART_DATA_PATH` or downloads it if necessary.

    :param raw: `True` if no preprocessing should be applied to the data. Otherwise, data is normalized to 1.
    :return: `(x_train, y_train), (x_test, y_test), min, max`.
    """
    path = get_file(
        "mnist.npz",
        path=config.HEART_DATA_PATH,
        url="https://s3.amazonaws.com/img-datasets/mnist.npz",
        verbose=True,
    )

    print(path)

    dict_mnist = np.load(path)
    x_train = dict_mnist["x_train"]
    y_train = dict_mnist["y_train"]
    x_test = dict_mnist["x_test"]
    y_test = dict_mnist["y_test"]
    dict_mnist.close()

    # Add channel axis
    min_, max_ = 0.0, 255.0
    if not raw:
        min_, max_ = 0.0, 1.0
        x_train = np.expand_dims(x_train, axis=3)
        x_test = np.expand_dims(x_test, axis=3)
        x_train, y_train = preprocess(x_train, y_train)
        x_test, y_test = preprocess(x_test, y_test)

    return (x_train, y_train), (x_test, y_test), min_, max_


def get_file(
    filename: str,
    url: Union[str, list[str]],
    path: Optional[str] = None,
    verbose: bool = False,
) -> str:
    """
    Downloads a file from a URL if it not already in the cache. The file at indicated by `url` is downloaded to the
    path `path` (default is ~/.art/data). and given the name `filename`. Files in tar, tar.gz, tar.bz, and zip formats
    can also be extracted. This is a simplified version of the function with the same name in Keras.

    :param filename: Name of the file.
    :param url: Download URL.
    :param path: Folder to store the download. If not specified, `~/.art/data` is used instead.
    :param verbose: If true, print download progress bar.
    :return: Path to the downloaded file.
    """

    # Sanitize URL
    urls = url if isinstance(url, list) else [url]

    # Instantiate base path
    base = os.path.expanduser(path) if path else os.path.expanduser("~/.art/data")
    if not os.access(base, os.W_OK):
        base = os.path.join(tempfile.gettempdir(), ".art")
    os.makedirs(base, exist_ok=True)

    download_path = os.path.join(base, filename)
    target = download_path

    if os.path.exists(target):
        return target

    _download_file(urls, download_path, verbose)

    return download_path


def _download_file(urls: list[str], download_path: str, verbose: bool) -> None:
    import ssl

    # Create SSL context that does not require verification
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    context.check_hostname = False
    context.verify_mode = ssl.CERT_NONE

    # Instantiate urllib opener
    opener = urllib.request.build_opener(urllib.request.HTTPSHandler(context=context))
    urllib.request.install_opener(opener)

    last_exception = None
    for link in urls:
        last_exception = _attempt_download(link, download_path, verbose)
        if last_exception is None:
            return  # Successfully Downloaded
    if os.path.exists(download_path):
        os.remove(download_path)
    raise Exception("Failed to download file") from last_exception


def _attempt_download(link: str, download_path: str, verbose: bool) -> Union[Exception, None]:
    # Constrain download link to only http: or https:
    if not link.startswith(("http:", "https:")):
        raise ValueError(f"Invalid URL (must start with http or https): {link}")

    try:
        if verbose:
            _download_with_progress(link, download_path)
        else:
            urlretrieve(link, download_path)  # noqa S310 (Constrained to http:/https:)
        return None
    except (HTTPError, URLError) as e:
        return e


def _download_with_progress(link: str, download_path: str) -> None:
    with tqdm() as t:
        # Initialize counter for tracking downloaded blocks in verbose mode
        last_block = [0]

        def hook(blocks, block_size, total):
            if total not in (None, -1):
                t.total = total
            t.update((blocks - last_block[0]) * block_size)
            last_block[0] = blocks

        urlretrieve(link, download_path, reporthook=hook)  # noqa S310 (Constrained to http:/https:)


def preprocess(
    x: np.ndarray,
    y: np.ndarray,
    nb_classes: int = 10,
    clip_values: Optional["CLIP_VALUES_TYPE"] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Scales `x` to [0, 1] and converts `y` to class categorical confidences.

    :param x: Data instances.
    :param y: Labels.
    :param nb_classes: Number of classes in dataset.
    :param clip_values: Original data range allowed value for features, either one respective scalar or one value per
           feature.
    :return: Rescaled values of `x`, `y`.
    """
    if clip_values is None:
        min_, max_ = np.amin(x), np.amax(x)
    else:
        min_, max_ = clip_values

    normalized_x = (x - min_) / (max_ - min_)
    categorical_y = to_categorical(y, nb_classes)

    return normalized_x, categorical_y


def to_categorical(labels: Union[np.ndarray, list[float]], nb_classes: Optional[int] = None) -> np.ndarray:
    """
    Convert an array of labels to binary class matrix.

    :param labels: An array of integer labels of shape `(nb_samples,)`.
    :param nb_classes: The number of classes (possible labels).
    :return: A binary matrix representation of `y` in the shape `(nb_samples, nb_classes)`.
    """
    labels = np.array(labels, dtype=int)
    if nb_classes is None:
        nb_classes = np.max(labels) + 1
    categorical = np.zeros((labels.shape[0], nb_classes), dtype=np.float32)
    categorical[np.arange(labels.shape[0]), np.squeeze(labels)] = 1
    return categorical
