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
"""Utility methods for converting data types to ART compatible versions."""

import logging
import uuid
from collections.abc import Sequence
from typing import Any, Optional, Union

import numpy as np
from datasets import Dataset as HFDataset
from datasets.iterable_dataset import IterableDataset as HFIterableDataset
from numpy.typing import NDArray
from PIL.Image import Image as PILImage
from torch import Tensor, is_tensor
from torch.utils.data.dataset import Subset as TorchSubsetDataset

logger: logging.Logger = logging.getLogger(__name__)

EMPTY_STRING = ""


class ImageDataset:
    """
    MAITE aligned dataset

    Examples
    --------

    We can define a white-box attack and generate adversarial images:

    >>> import numpy as np
    >>> import torch
    >>> import torchvision
    >>> from heart_library.estimators.classification.pytorch import JaticPyTorchClassifier
    >>> from heart_library.utils import ImageDataset
    >>> from datasets import load_dataset
    >>> from heart_library.metrics import AccuracyPerturbationMetric
    >>> from heart_library.attacks.attack import JaticAttack
    >>> from art.attacks.evasion import ProjectedGradientDescentPyTorch
    >>> from copy import deepcopy
    >>> from torchvision import transforms

    Load an applicable dataset:

    >>> data = load_dataset("CDAO/xview-subset-classification", split="test[0:14]")

    Define the model:

    >>> model = torchvision.models.resnet18(False)
    >>> _ = model.eval()

    Wrap the model:

    >>> jptc = JaticPyTorchClassifier(
    ...     model=model,
    ...     loss=torch.nn.CrossEntropyLoss(),
    ...     input_shape=(3, 224, 224),
    ...     nb_classes=6,
    ...     clip_values=(0, 1),
    ... )

    Transform dataset:

    >>> IMAGE_H, IMAGE_W = 224, 224

    >>> preprocess = transforms.Compose([transforms.Resize((IMAGE_H, IMAGE_W)), transforms.ToTensor()])

    >>> data = data.map(lambda x: {"image": preprocess(x["image"]), "label": x["label"]})
    >>> to_image = lambda x: transforms.ToPILImage()(torch.Tensor(x))

    Define and wrap the attacks:

    >>> evasion_attack_undefended = ProjectedGradientDescentPyTorch(estimator=jptc, max_iter=10, eps=0.03)
    >>> attack_undefended = JaticAttack(evasion_attack_undefended, norm=2)

    Generate adversarial images:

    >>> x_adv, y, metadata = attack_undefended(data=data)

    >>> data_with_detections = ImageDataset(data, deepcopy(jptc(data)), metadata)
    """

    metadata: dict[str, Any]

    def __init__(
        self,
        images: list[NDArray[np.float32]],
        targets: NDArray[np.float32],
        metadata: list[dict[str, Any]],
        metadata_id: Optional[str] = None,
    ) -> None:
        """ImageDataset initialization.

        Args:
            images (List[NDArray[np.float32]]): Array representation of images.
            targets (NDArray[np.float32]): Targets.
            metadata (List[dict[str, Any]]): Metadata.
        """
        self._images = images
        self._targets = targets
        self._metadata = metadata
        self.metadata = {"id": metadata_id if metadata_id is not None else str(uuid.uuid4())}

    def __len__(self) -> int:
        """Returns image count.

        Returns:
            int: Image count.
        """
        return len(self._images)

    def __getitem__(self, ind: int) -> tuple[NDArray[np.float32], NDArray[np.float32], dict[str, Any]]:
        """Returns images, targets, metadata.

        Args:
            ind (int): Index of image, target, metadata combination to be returned.

        Returns:
            Tuple[NDArray[np.float32], NDArray[np.float32], dict[str, Any]]:
                Specified image, target, metadata combination.
        """
        return self._images[ind], self._targets[ind], self._metadata[ind]


def hf_dataset_to_maite(
    dataset: Any,  # noqa ANN401
    image_label: str = EMPTY_STRING,
    target_label: str = EMPTY_STRING,
    meta_label: str = EMPTY_STRING,
    indices: Optional[Sequence[int]] = None,
) -> ImageDataset:
    """Convert HF dataset to MAITE aligned dataset

    Args:
        dataset (Any): Image data.
        image_label (str, optional): Image label. Defaults to EMPTY_STRING.
        target_label (str, optional): Target label. Defaults to EMPTY_STRING.
        meta_label (str, optional): Metadata label. Defaults to EMPTY_STRING.
        indices (Optional[Sequence[int]], optional): Indices. Defaults to None.

    Raises:
        ValueError: if image feature not found in dataset.
        ValueError: if image feature not found in dataset.

    Returns:
        ImageDataset: MAITE aligned dataset.
    """
    indices = [] if indices is None else indices

    images: list[Any] = []
    target_list: list[Any] = []
    metadata: list[Any] = []

    if len(indices) == 0:
        for item in dataset:
            # Generate list of images based on image label provided
            images = __handle_image_labels(image_label, dataset, images, item)

            # Generate lists of targets and metadata based on labels provided
            target_list, metadata = __handle_target_meta_labels(
                target_label,
                target_list,
                meta_label,
                metadata,
                dataset,
                item,
            )
    else:
        for idx in indices:
            item = dataset[idx]
            # Generate list of images based on image label provided
            images = __handle_image_labels(image_label, dataset, images, item)

            # Generate lists of targets and metadata based on labels provided
            target_list, metadata = __handle_target_meta_labels(
                target_label,
                target_list,
                meta_label,
                metadata,
                dataset,
                item,
            )

    targets = np.array([None] * len(images)) if len(target_list) == 0 else np.array(target_list)

    if len(metadata) == 0:
        metadata = [{} for _ in range(len(images))]

    return ImageDataset(images, targets, metadata)


def __handle_image_labels(
    image_label: str,
    dataset: HFDataset,
    images: list[Any],
    item: dict[str, Any],
) -> list[Any]:
    """Generate list of images based on indices and image label provided.

    Args:
        image_label (str): Label of image feature in dataset.
        dataset (Dataset): HF dataset.
        images (list[Any]): Empty list for images to be appended to.
        item (dict[str, Any]): Item within dataset.

    Raises:
        ValueError: Image feature not found in dataset.

    Returns:
        list[Any]: List of images.
    """
    if image_label is not EMPTY_STRING and image_label in dataset.features:
        images.append(item[image_label])
    elif "img" in dataset.features:
        images.append(item["img"])
    elif "image" in dataset.features:
        images.append(item["image"])
    else:
        raise ValueError("Image feature not found in dataset.")
    return images


def __handle_target_meta_labels(
    target_label: str,
    target_list: list[Any],
    meta_label: str,
    metadata: list[Any],
    dataset: HFDataset,
    item: dict[str, Any],
) -> tuple[list[Any], list[Any]]:
    """Generate lists of targets and metadata based on indices and labels provided.

    Args:
        target_label (str): Label of target feature in dataset.
        target_list (list[Any]): Empty list for targets to be appended to.
        meta_label (str): Label of metadata feature in dataset.
        metadata (list[Any]): Empty list for metadata to be appended to.
        dataset (Dataset): HF dataset.
        item (dict[str, Any]): Item within dataset.

    Returns:
        tuple[list[Any], list[Any]]: Lists of targets and metadata.
    """
    if target_label is not EMPTY_STRING:
        target_list.append(item[target_label])
    elif "label" in dataset.features:
        target_list.append(item["label"])

    if meta_label is not EMPTY_STRING:
        metadata.append(item[meta_label])
    return target_list, metadata


def torch_subset_to_maite(dataset: Any) -> ImageDataset:  # noqa ANN401
    """Convert Torch subset dataset to MAITE aligned dataset.

    Args:
        dataset (Any): Torch dataset.

    Returns:
        ImageDataset: MAITE aligned dataset.
    """
    images = []
    data = iter(dataset)
    target_list = []
    for item in data:
        images.append(item[0])
        target_list.append(item[1])
    targets = np.array(target_list)
    metadata: list[dict[str, Any]] = [{} for _ in range(len(images))]

    return ImageDataset(images, targets, metadata)


def process_inputs_for_art(
    data: Any,  # noqa ANN401
) -> tuple[NDArray[np.float32], Optional[Union[NDArray[np.float32], list[dict[str, Any]]]], list[dict[str, Any]]]:
    """Convert JATIC supported data to ART supported data.

    Args:
        data (Any): JATIC supported data.

    Raises:
        ValueError: if Dataset does not implement __getitem__ or __iter__.
        ValueError: if Images and targets are not the same length.
        ValueError: if Images are empty.

    Returns:
        Tuple[NDArray[np.float32], Optional[Union[NDArray[np.float32], List[dict[str, Any]]]], List[dict[str, Any]]]:
             ART supported data.
    """
    images = np.array([])
    image_list: list[Any] = []
    targets: Optional[Any] = None
    metadata: Any = []

    # Convert Hugging Face and Torch Subset data to MAITE aligned ImageDataset.
    data = __handle_hf_torch_data(data)

    conditions: dict[bool, Any] = {
        # if data is a Sequence
        isinstance(data, Sequence): __handle_sequence,
        # if Tuple of batched data, convert to np.ndarray
        isinstance(data, tuple) and isinstance(data[0], (list, np.ndarray, Tensor)): __handle_tuple,
        # if dict, assume np.ndarray and set values if present.
        isinstance(data, dict) and "images" in data: __handle_dict,
        # if torch.Tensor, convert images to np.ndarray. No targets or metadata.
        is_tensor(data): __handle_tensor,
        # if np.ndarray, convert images to np.ndarray. No targets or metadata.
        isinstance(data, np.ndarray): __handle_ndarray,
    }
    for condition, result in conditions.items():
        if condition:
            return result(data, images, targets, metadata)
    # if Dataset, convert to np.ndarray
    return __handle_dataset(data, images, image_list, targets, metadata)


def __handle_ndarray(
    data: NDArray[np.float32],
    images: np.ndarray,
    targets: Optional[Any],  # noqa: ANN401
    metadata: list,
) -> tuple[NDArray[np.float32], None, list]:
    """Process ndarray format.

    Args:
        data (NDArray[np.float32]): Input data.
        images (np.ndarray): Initialized image list.
        targets (Optional[Any]): Initialized targets list.
        metadata (list): Initialized metadata list.

    Returns:
        tuple[NDArray[np.float32], None, list]: Images, targets metadata.
    """
    images = data
    targets = None
    metadata = []
    return images, targets, metadata


def __handle_tensor(
    data: Tensor,
    images: NDArray[np.float32],
    targets: Optional[Any],  # noqa: ANN401
    metadata: list,
) -> tuple[NDArray[np.float32], None, list]:
    """Process Tensor format.

    Args:
        data (NDArray[np.float32]): Input data.
        images (np.ndarray): Initialized image list.
        targets (Optional[Any]): Initialized targets list.
        metadata (list): Initialized metadata list.

    Returns:
        tuple[NDArray[np.float32], None, list]: Images, targets metadata.
    """
    images = np.asarray(data)
    targets = None
    metadata = []
    return images, targets, metadata


def __handle_dict(
    data: dict[str, Any],
    images: NDArray[np.float32],
    targets: Optional[Any],  # noqa: ANN401
    metadata: list,
) -> tuple[NDArray[np.float32], NDArray[np.float32], list[dict[str, Any]]]:
    """Process dict format.

    Args:
        data (NDArray[np.float32]): Input data.
        images (np.ndarray): Initialized image list.
        targets (Optional[Any]): Initialized targets list.
        metadata (list): Initialized metadata list.

    Returns:
        tuple[NDArray[np.float32], None, list]: Images, targets metadata.
    """
    images = data["images"]
    # Generate targets
    targets = __handle_dict_labels(data, targets)
    metadata = data.get("metadata", [])
    return images, targets, metadata


def __handle_tuple(
    data: tuple,
    images: NDArray[np.float32],
    targets: Optional[Any],  # noqa: ANN401
    metadata: list,
) -> tuple[NDArray[np.float32], NDArray[np.float32], list[dict[str, Any]]]:
    """Process tuple format.

    Args:
        data (NDArray[np.float32]): Input data.
        images (np.ndarray): Initialized image list.
        targets (Optional[Any]): Initialized targets list.
        metadata (list): Initialized metadata list.

    Returns:
        tuple[NDArray[np.float32], None, list]: Images, targets metadata.
    """
    images = np.asarray(data[0]).astype(np.float32)
    targets = data[1]
    metadata = data[2]

    # Generate targets if data is in tuple format.
    targets = __handle_tuple_targets(targets)
    return images, targets, metadata


def __handle_sequence(
    data: Sequence,
    images: NDArray[np.float32],
    targets: Optional[Any],  # noqa: ANN401
    metadata: list,
) -> tuple[NDArray[np.float32], None, list]:
    """Process Sequence format.

    Args:
        data (NDArray[np.float32]): Input data.
        images (np.ndarray): Initialized image list.
        targets (Optional[Any]): Initialized targets list.
        metadata (list): Initialized metadata list.

    Returns:
        tuple[NDArray[np.float32], None, list]: Images, targets metadata.
    """
    # Generate images, targets, metadata if data is in Sequence format.
    images, targets, metadata = __handle_sequence_targets(data, images, targets, metadata)
    return images, targets, metadata


def __handle_dataset(
    data: Any,  # noqa: ANN401
    images: NDArray[np.float32],
    image_list: list,
    targets: Optional[Any],  # noqa: ANN401
    metadata: list,
) -> tuple[NDArray[np.float32], Any, list]:
    """Process other dataset format.

    Args:
        data (NDArray[np.float32]): Input data.
        images (np.ndarray): Initialized image list.
        image_list (list): Temporary image list to be processed.
        targets (Optional[Any]): Initialized targets list.
        metadata (list): Initialized metadata list.

    Returns:
        tuple[NDArray[np.float32], None, list]: Images, targets metadata.
    """
    # check for __getitem__ and __iter__.
    image_list, targets, metadata = __handle_dataset_attr(data, image_list, targets, metadata)

    # Check to see of image list is nonempty and same length as targets.
    __check_dataset_lengths(image_list, targets)

    # Handle dataset images of type PIL, NDArray, tensor, list.
    images = __handle_dataset_images_types(image_list, images)

    # Convert dataset targets to correct type.
    targets = __handle_dataset_targets(targets)
    return images, targets, metadata


def __handle_hf_torch_data(data: Any) -> ImageDataset:  # noqa ANN401
    """Convert Hugging Face and Torch Subset data to MAITE aligned ImageDataset.

    Args:
        data (Any): Input data to be converted.

    Returns:
        ImageDataset: MAITE aligned image data.
    """
    # convert Hugging Face
    if isinstance(data, (HFDataset, HFIterableDataset)):
        data = hf_dataset_to_maite(data)

    # convert Hugging Face if wrapped in Torch Subset
    elif isinstance(data, TorchSubsetDataset) and isinstance(data.dataset, HFDataset):
        data = hf_dataset_to_maite(data.dataset, indices=data.indices)

    elif isinstance(data, TorchSubsetDataset):
        data = torch_subset_to_maite(data)
    return data


def __handle_dict_labels(data: dict[str, Any], targets: Any) -> NDArray[np.float32]:  # noqa ANN401
    """Generate targets if data is dict format.

    Args:
        data (dict[str, Any]): Input data.
        targets (Any): Empty initialization.

    Returns:
        NDArray[np.float32]: Targets in np array format.
    """
    if "labels" in data:
        targets = data["labels"]

        # for object detection, convert to correct format if exist
        if hasattr(targets[0], "boxes") and hasattr(targets[0], "scores"):
            targets = [
                {
                    "boxes": np.asarray(t.boxes).astype(np.float32),
                    "scores": np.asarray(t.scores).astype(np.float32),
                    "labels": np.asarray(t.labels).astype(np.int64),
                }
                for t in targets
            ]
        elif isinstance(targets[0], np.ndarray):
            targets = np.asarray(targets)
        elif not any(targets):
            targets = None
        else:
            targets = np.asarray(targets)
    return targets


def __handle_tuple_targets(targets: Any) -> NDArray[np.float32]:  # noqa ANN401
    """Generate targets if data is in tuple format.

    Args:
        targets (Any): Second item in data of tuple format.

    Returns:
        NDArray[np.float32]: Targets in np array format.
    """
    # for object detection, convert to correct format if exist
    if hasattr(targets[0], "boxes") and hasattr(targets[0], "scores"):
        targets = [
            {
                "boxes": np.asarray(t.boxes).astype(np.float32),
                "scores": np.asarray(t.scores).astype(np.float32),
                "labels": np.asarray(t.labels).astype(np.int64),
            }
            for t in targets
        ]
    elif isinstance(targets[0], np.ndarray) or is_tensor(targets[0]):
        targets = np.asarray(targets)
    elif not any(targets):
        targets = None
    else:
        targets = np.asarray(targets)
    return targets


def __handle_sequence_targets(
    data: Any,  # noqa: ANN401
    images: NDArray[np.float32],
    targets: Optional[Any],  # noqa: ANN401
    metadata: Any,  # noqa: ANN401
) -> tuple[NDArray[np.float32], None, list]:
    """Generate images, targets, metadata if data is in Sequence format.

    Args:
        data (Any): Input data.
        images (NDArray[np.float32]): Initialized image list.
        targets (Optional[Any]): Initialized targets list.
        metadata (Any): Initialized metadata list.

    Returns:
        tuple[NDArray[np.float32], None, list]: np array of images and empty targets, metadata.
    """
    # here assuming data is a tensor - what if a numpy array?
    # what if each batch (len 1 or >1), has different shape?
    # - in this case will stack fail as different dim images
    # - should auto pad or resize? what if this occurs in other data formats?
    # a sequence of image batches

    if data[0].ndim == 4:
        images = np.vstack([np.asarray(batch) for batch in data])
        targets = None
        metadata = []
    # a sequence of single images
    else:
        images = np.stack([np.asarray(batch) for batch in data])
        targets = None
        metadata = []
    return images, targets, metadata


def __handle_dataset_attr(
    data: Any,  # noqa: ANN401
    image_list: list,
    targets: Any,  # noqa: ANN401
    metadata: list,
) -> tuple[list, list, list]:
    """If dataset and has specified attributes, generate images, targets, metadata.

    Args:
        data (Any): Input data.
        image_list (list): Initialized image list.
        targets (list): Initialized targets list.
        metadata (list): Initialized metadata list.

    Raises:
        ValueError: If dataset does not implement __getitem__ or __iter__.

    Returns:
        tuple[list, list, list]: images, targets, metadata
    """
    if hasattr(data, "__getitem__"):
        image_list = []
        targets = []
        metadata = []
        for item in data:
            image_list.append(item[0])
            targets.append(item[1])
            metadata.append(item[2])

    elif hasattr(data, "__iter__"):
        image_list = [item[0] for item in data]
        targets = [item[1] for item in data]
        metadata = [item[2] for item in data if len(item) == 3]

    else:
        raise ValueError("Dataset must implement __getitem__ or __iter__.")
    return image_list, targets, metadata


def __check_dataset_lengths(image_list: list, targets: list) -> None:
    """Check to see of image list is nonempty and same length as targets.

    Args:
        image_list (list): Images.
        targets (list): Targets.

    Raises:
        ValueError: If images and targets are not the same length.
        ValueError: If image list is empty.
    """
    # check images, targets and meta are same length
    if len(image_list) != len(targets):
        raise ValueError("Images and targets must be same length.")

    # check not empty data
    if not len(image_list) > 0:
        raise ValueError("Images should not be empty.")


def __handle_dataset_images_types(image_list: list, images: NDArray[np.float32]) -> NDArray[np.float32]:
    """Handle dataset images of type PIL, NDArray, tensor, list.

    Args:
        image_list (list): Images.
        images (NDArray[np.float32]): np array formatted images.

    Returns:
        NDArray[np.float32]: Images in NDArray format.
    """
    # if images are in PIL format, convert to np.ndarray
    if isinstance(image_list[0], PILImage):
        images = np.asarray(image_list).transpose(0, 3, 1, 2).astype(np.float32)

    # if images are np.ndarray or torch.Tensor, convert to np.ndarray
    # facilitating batches
    elif isinstance(image_list[0], np.ndarray) or is_tensor(image_list[0]):
        if image_list[0].ndim == 3:
            images = np.asarray(image_list).astype(np.float32)
        else:
            images = np.concatenate([np.asarray(batch) for batch in image_list])

    # if images are a list, convert to np.ndarray
    elif isinstance(image_list[0], list):
        images = np.asarray(image_list).astype(np.float32)
    return images


def __handle_dataset_targets(targets: Any) -> Any:  # noqa ANN401
    """Convert dataset targets to correct type.

    Args:
        targets (list): Targets.

    Returns:
        Any: Targets in NDArray or list if object detection.
    """
    # for object detection, convert to correct format if exist
    if hasattr(targets[0], "boxes") and hasattr(targets[0], "scores"):
        targets = [
            {
                "boxes": np.asarray(t.boxes).astype(np.float32),
                "scores": np.asarray(t.scores).astype(np.float32),
                "labels": np.asarray(t.labels).astype(np.int64),
            }
            for t in targets
        ]
    elif isinstance(targets[0], np.ndarray) or is_tensor(targets[0]):
        targets = np.asarray(targets)
    elif not any(targets):
        targets = None
    else:
        targets = np.asarray(targets)
    return targets
