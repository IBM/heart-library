import logging
import os
import shutil
import tarfile
import zipfile
from typing import List, Optional, Tuple, Union

import numpy as np
from tqdm.auto import tqdm

from heart_library import config

DATASET_TYPE = Tuple[
    Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], float, float
]

CLIP_VALUES_TYPE = Tuple[Union[int, float, np.ndarray], Union[int, float, np.ndarray]]

logger = logging.getLogger(__name__)


class HEARTTestException(Exception):
    def __init__(self, message):
        super().__init__(message)


class HEARTTestFixtureNotImplemented(HEARTTestException):
    def __init__(self, message, fixture_name, framework, parameters_dict=""):
        super().__init__(
            "Could NOT run test for framework: {0} due to fixture: {1}. Message was: '"
            "{2}' for the following parameters: {3}".format(framework, fixture_name, message, parameters_dict)
        )


def get_mnist_image_classifier_pt(from_logits=False, load_init=True, use_maxpool=True, is_jatic=True):
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

    from heart_library.estimators.classification.pytorch import JaticPyTorchClassifier
    from art.estimators.classification.pytorch import PyTorchClassifier

    if use_maxpool:

        class Model(torch.nn.Module):
            """
            Create model for pytorch.

            The weights and biases are identical to the TensorFlow model in get_classifier_tf().
            """

            def __init__(self):
                super(Model, self).__init__()

                self.conv = torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=7)
                self.relu = torch.nn.ReLU()
                self.pool = torch.nn.MaxPool2d(4, 4)
                self.fullyconnected = torch.nn.Linear(25, 10)

                if load_init:
                    w_conv2d = np.load(
                        os.path.join(
                            os.path.dirname(os.path.dirname(__file__)), "utils/resources/models", "W_CONV2D_MNIST.npy"
                        )
                    )
                    b_conv2d = np.load(
                        os.path.join(
                            os.path.dirname(os.path.dirname(__file__)), "utils/resources/models", "B_CONV2D_MNIST.npy"
                        )
                    )
                    w_dense = np.load(
                        os.path.join(
                            os.path.dirname(os.path.dirname(__file__)), "utils/resources/models", "W_DENSE_MNIST.npy"
                        )
                    )
                    b_dense = np.load(
                        os.path.join(
                            os.path.dirname(os.path.dirname(__file__)), "utils/resources/models", "B_DENSE_MNIST.npy"
                        )
                    )

                    w_conv2d_pt = w_conv2d.reshape((1, 1, 7, 7))

                    self.conv.weight = torch.nn.Parameter(torch.Tensor(w_conv2d_pt))
                    self.conv.bias = torch.nn.Parameter(torch.Tensor(b_conv2d))
                    self.fullyconnected.weight = torch.nn.Parameter(torch.Tensor(np.transpose(w_dense)))
                    self.fullyconnected.bias = torch.nn.Parameter(torch.Tensor(b_dense))

            def forward(self, x):
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

    else:

        class Model(torch.nn.Module):
            """
            Create model for pytorch.
            Here the model does not use maxpooling. Needed for certification tests.
            """

            def __init__(self):
                super(Model, self).__init__()

                self.conv = torch.nn.Conv2d(
                    in_channels=1, out_channels=16, kernel_size=(4, 4), dilation=(1, 1), padding=(0, 0), stride=(3, 3)
                )

                self.fullyconnected = torch.nn.Linear(in_features=1296, out_features=10)

                self.relu = torch.nn.ReLU()

                if load_init:
                    w_conv2d = np.load(
                        os.path.join(
                            os.path.dirname(os.path.dirname(__file__)),
                            "utils/resources/models",
                            "W_CONV2D_NO_MPOOL_MNIST.npy",
                        )
                    )
                    b_conv2d = np.load(
                        os.path.join(
                            os.path.dirname(os.path.dirname(__file__)),
                            "utils/resources/models",
                            "B_CONV2D_NO_MPOOL_MNIST.npy",
                        )
                    )
                    w_dense = np.load(
                        os.path.join(
                            os.path.dirname(os.path.dirname(__file__)),
                            "utils/resources/models",
                            "W_DENSE_NO_MPOOL_MNIST.npy",
                        )
                    )
                    b_dense = np.load(
                        os.path.join(
                            os.path.dirname(os.path.dirname(__file__)),
                            "utils/resources/models",
                            "B_DENSE_NO_MPOOL_MNIST.npy",
                        )
                    )

                    self.conv.weight = torch.nn.Parameter(torch.Tensor(w_conv2d))
                    self.conv.bias = torch.nn.Parameter(torch.Tensor(b_conv2d))
                    self.fullyconnected.weight = torch.nn.Parameter(torch.Tensor(w_dense))
                    self.fullyconnected.bias = torch.nn.Parameter(torch.Tensor(b_dense))

            def forward(self, x):
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

    # Define the network
    model = Model()
    # Define a loss function and optimizer
    loss_fn = torch.nn.CrossEntropyLoss(reduction="sum")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Get classifier
    ptc = JaticPyTorchClassifier(
        model=model, loss=loss_fn, optimizer=optimizer, input_shape=(1, 28, 28), nb_classes=10, clip_values=(0, 1)
    )

    return ptc


def get_cifar10_image_classifier_pt(from_logits=False, is_jatic=True):
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

    from heart_library.estimators.classification.pytorch import JaticPyTorchClassifier
    from art.estimators.classification.pytorch import PyTorchClassifier

    class Model(torch.nn.Module):
        """
        Create model for pytorch.
        Here the model does not use maxpooling. Needed for certification tests.
        """

        def __init__(self):
            super(Model, self).__init__()

            self.conv = torch.nn.Conv2d(
                in_channels=3, out_channels=16, kernel_size=(4, 4), dilation=(1, 1), padding=(0, 0), stride=(3, 3)
            )

            self.fullyconnected = torch.nn.Linear(in_features=1600, out_features=10)

            self.relu = torch.nn.ReLU()

            w_conv2d = np.load(
                os.path.join(
                    os.path.dirname(os.path.dirname(__file__)),
                    "utils/resources/models",
                    "W_CONV2D_NO_MPOOL_CIFAR10.npy",
                )
            )
            b_conv2d = np.load(
                os.path.join(
                    os.path.dirname(os.path.dirname(__file__)),
                    "utils/resources/models",
                    "B_CONV2D_NO_MPOOL_CIFAR10.npy",
                )
            )
            w_dense = np.load(
                os.path.join(
                    os.path.dirname(os.path.dirname(__file__)),
                    "utils/resources/models",
                    "W_DENSE_NO_MPOOL_CIFAR10.npy",
                )
            )
            b_dense = np.load(
                os.path.join(
                    os.path.dirname(os.path.dirname(__file__)),
                    "utils/resources/models",
                    "B_DENSE_NO_MPOOL_CIFAR10.npy",
                )
            )

            self.conv.weight = torch.nn.Parameter(torch.Tensor(w_conv2d))
            self.conv.bias = torch.nn.Parameter(torch.Tensor(b_conv2d))
            self.fullyconnected.weight = torch.nn.Parameter(torch.Tensor(w_dense))
            self.fullyconnected.bias = torch.nn.Parameter(torch.Tensor(b_dense))

        def forward(self, x):
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

    # Define the network
    model = Model()

    # Define a loss function and optimizer
    loss_fn = torch.nn.CrossEntropyLoss(reduction="sum")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    if is_jatic:
        jptc = JaticPyTorchClassifier(
            model=model,
            loss=loss_fn,
            optimizer=optimizer,
            input_shape=(3, 32, 32),
            nb_classes=10,
            clip_values=(0, 1),
        )
    else:
        jptc = PyTorchClassifier(
            model=model,
            loss=loss_fn,
            optimizer=optimizer,
            input_shape=(3, 32, 32),
            nb_classes=10,
            clip_values=(0, 1),
        )

    return jptc


def master_seed(seed=1234, set_random=True, set_numpy=True, set_tensorflow=False, set_mxnet=False, set_torch=False):
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

    # Set Python seed
    if set_random:
        import random

        random.seed(seed)

    # Set Numpy seed
    if set_numpy:
        np.random.seed(seed)
        np.random.RandomState(seed)

    # Now try to set seed for all specific frameworks
    if set_tensorflow:
        try:
            import tensorflow as tf

            logger.info("Setting random seed for TensorFlow.")
            if tf.__version__[0] == "2":
                tf.random.set_seed(seed)
            else:
                tf.set_random_seed(seed)
        except ImportError:
            logger.info("Could not set random seed for TensorFlow.")

    if set_mxnet:
        try:
            import mxnet as mx

            logger.info("Setting random seed for MXNet.")
            mx.random.seed(seed)
        except ImportError:
            logger.info("Could not set random seed for MXNet.")

    if set_torch:
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
    )

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
    filename: str, url: Union[str, List[str]], path: Optional[str] = None, extract: bool = False, verbose: bool = False
) -> str:
    """
    Downloads a file from a URL if it not already in the cache. The file at indicated by `url` is downloaded to the
    path `path` (default is ~/.art/data). and given the name `filename`. Files in tar, tar.gz, tar.bz, and zip formats
    can also be extracted. This is a simplified version of the function with the same name in Keras.

    :param filename: Name of the file.
    :param url: Download URL.
    :param path: Folder to store the download. If not specified, `~/.art/data` is used instead.
    :param extract: If true, tries to extract the archive.
    :param verbose: If true, print download progress bar.
    :return: Path to the downloaded file.
    """
    if isinstance(url, str):
        url_list = [url]
    else:
        url_list = url

    if path is None:
        path_ = os.path.expanduser(config.ART_DATA_PATH)
    else:
        path_ = os.path.expanduser(path)
    if not os.access(path_, os.W_OK):
        path_ = os.path.join("/tmp", ".art")

    if not os.path.exists(path_):
        os.makedirs(path_)

    if extract:
        extract_path = os.path.join(path_, filename)
        full_path = extract_path + ".tar.gz"
    else:
        full_path = os.path.join(path_, filename)

    # Determine if dataset needs downloading
    download = not os.path.exists(full_path)

    if download:
        logger.info("Downloading data from %s", url)
        error_msg = "URL fetch failure on {}: {} -- {}"
        try:
            for url_i in url_list:
                try:
                    # The following two lines should prevent occasionally occurring
                    # [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed (_ssl.c:847)
                    import ssl

                    from six.moves.urllib.error import HTTPError, URLError
                    from six.moves.urllib.request import urlretrieve

                    ssl._create_default_https_context = ssl._create_unverified_context

                    if verbose:
                        with tqdm() as t_bar:
                            last_block = [0]

                            def progress_bar(blocks: int = 1, block_size: int = 1, total_size: Optional[int] = None):
                                """
                                :param blocks: Number of blocks transferred so far [default: 1].
                                :param block_size: Size of each block (in tqdm units) [default: 1].
                                :param total_size: Total size (in tqdm units). If [default: None] or -1, remains
                                                   unchanged.
                                """
                                if total_size not in (None, -1):
                                    t_bar.total = total_size
                                displayed = t_bar.update((blocks - last_block[0]) * block_size)
                                last_block[0] = blocks
                                return displayed

                            urlretrieve(url_i, full_path, reporthook=progress_bar)
                    else:
                        urlretrieve(url_i, full_path)

                except HTTPError as exception:  # pragma: no cover
                    raise Exception(
                        error_msg.format(url_i, exception.code, exception.msg)  # type: ignore
                    ) from HTTPError  # type: ignore
                except URLError as exception:  # pragma: no cover
                    raise Exception(
                        error_msg.format(url_i, exception.errno, exception.reason)  # type: ignore
                    ) from HTTPError  # type: ignore
        except (Exception, KeyboardInterrupt):  # pragma: no cover
            if os.path.exists(full_path):
                os.remove(full_path)
            raise

    if extract:
        if not os.path.exists(extract_path):
            _extract(full_path, path_)
        return extract_path

    return full_path


def _extract(full_path: str, path: str) -> bool:
    archive: Union[zipfile.ZipFile, tarfile.TarFile]
    if full_path.endswith("tar"):  # pragma: no cover
        if tarfile.is_tarfile(full_path):
            archive = tarfile.open(full_path, "r:")
    elif full_path.endswith("tar.gz"):  # pragma: no cover
        if tarfile.is_tarfile(full_path):
            archive = tarfile.open(full_path, "r:gz")
    elif full_path.endswith("zip"):  # pragma: no cover
        if zipfile.is_zipfile(full_path):
            archive = zipfile.ZipFile(full_path)
        else:
            return False
    else:
        return False
    try:
        archive.extractall(path)
    except (tarfile.TarError, RuntimeError, KeyboardInterrupt):  # pragma: no cover
        if os.path.exists(path):
            if os.path.isfile(path):
                os.remove(path)
            else:
                shutil.rmtree(path)
        raise
    return True


def preprocess(
    x: np.ndarray,
    y: np.ndarray,
    nb_classes: int = 10,
    clip_values: Optional["CLIP_VALUES_TYPE"] = None,
) -> Tuple[np.ndarray, np.ndarray]:
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


def to_categorical(labels: Union[np.ndarray, List[float]], nb_classes: Optional[int] = None) -> np.ndarray:
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
