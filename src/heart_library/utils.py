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
Utility methods for converting data types to ART compatible versions.
"""
import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from datasets import Dataset as HFDataset
from datasets.iterable_dataset import IterableDataset as HFIterableDataset
from numpy.typing import NDArray
from PIL.Image import Image as PILImage
from torch import Tensor, is_tensor
from torch.utils.data.dataset import Subset as TorchSubsetDataset

logger = logging.getLogger(__name__)

EMPTY_STRING = ""


class ImageDataset:
    """
    MAITE aligned dataset
    """

    def __init__(self, images: List[np.ndarray], targets: np.ndarray, metadata: List[Dict[str, Any]]):
        self.images = images
        self.targets = targets
        self.metadata = metadata

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, ind: int) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        return self.images[ind], self.targets[ind], self.metadata[ind]


def hf_dataset_to_maite(
    dataset: HFDataset,
    image_label: str = EMPTY_STRING,
    target_label: str = EMPTY_STRING,
    meta_label: str = EMPTY_STRING,
    indices: Optional[Sequence[int]] = None,
) -> ImageDataset:
    """
    Convert HF dataset to MAITE aligned dataset
    """
    if indices is None:
        indices = []

    images = []
    target_list = []
    metadata = []

    if len(indices) == 0:
        for item in dataset:

            if image_label is not EMPTY_STRING and image_label in dataset.features:
                images.append(item[image_label])
            elif "img" in dataset.features:
                images.append(item["img"])
            elif "image" in dataset.features:
                images.append(item["image"])
            else:
                raise ValueError("Image feature not found in dataset.")

            if target_label is not EMPTY_STRING:
                target_list.append(item[target_label])
            elif "label" in dataset.features:
                target_list.append(item["label"])

            if meta_label is not EMPTY_STRING:
                metadata.append(item[meta_label])
    else:
        for idx in indices:
            item = dataset[idx]
            if image_label is not EMPTY_STRING and image_label in dataset.features:
                images.append(item[image_label])
            elif "img" in dataset.features:
                images.append(item["img"])
            elif "image" in dataset.features:
                images.append(item["image"])
            else:
                raise ValueError("Image feature not found in dataset.")

            if target_label is not EMPTY_STRING:
                target_list.append(item[target_label])
            elif "label" in dataset.features:
                target_list.append(item["label"])

            if meta_label is not EMPTY_STRING:
                metadata.append(item[meta_label])

    if len(target_list) == 0:
        targets = np.array([None] * len(images))
    else:
        targets = np.array(target_list)

    if len(metadata) == 0:
        metadata = [{} for i in range(len(images))]

    return ImageDataset(images, targets, metadata)


def torch_subset_to_maite(dataset: TorchSubsetDataset) -> ImageDataset:
    """
    Convert Torch subset dataset to MAITE aligned dataset
    """
    images = []
    data = iter(dataset)
    target_list = []
    for item in data:
        images.append(item[0])
        target_list.append(item[1])
    targets = np.array(target_list)
    metadata: List[Dict[str, Any]] = [{} for i in range(len(images))]

    return ImageDataset(images, targets, metadata)


def process_inputs_for_art(
    data: Union[
        HFDataset,
        HFIterableDataset,
        TorchSubsetDataset,
        np.ndarray,
        Tensor,
        Dict,
        Tuple,
        Sequence,
    ]
) -> Tuple[NDArray, Optional[NDArray], List[Dict[str, Any]]]:
    """
    Convert JATIC supported data to ART supported data
    """
    images = np.array([])
    # convert Hugging Face
    if isinstance(data, (HFDataset, HFIterableDataset)):
        data = hf_dataset_to_maite(data)

    # convert Hugging Face if wrapped in Torch Subset
    elif isinstance(data, TorchSubsetDataset) and isinstance(data.dataset, HFDataset):
        data = hf_dataset_to_maite(data.dataset, indices=data.indices)

    elif isinstance(data, TorchSubsetDataset):
        data = torch_subset_to_maite(data)

    # if np.ndarray, convert images to np.ndarray. No targets or metadata.
    if isinstance(data, np.ndarray):
        images = data
        targets = None
        metadata: List[Dict[str, Any]] = []

    # if torch.Tensor, convert images to np.ndarray. No targets or metadata.
    elif is_tensor(data):
        images = np.asarray(data)
        targets = None
        metadata = []

    # if Dict, assume np.ndarray and set values if present.
    elif isinstance(data, Dict) and "images" in data:
        images = data["images"]
        targets = None
        metadata = []

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

        if "metadata" in data:
            metadata = data["metadata"]

    # if Tuple of batched data, convert to np.ndarray
    elif isinstance(data, tuple) and isinstance(data[0], (list, np.ndarray, Tensor)):
        images = np.asarray(data[0]).astype(np.float32)
        targets = data[1]
        metadata = data[2]

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

    # if data is a Sequence
    elif isinstance(data, Sequence):
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

    # if Dataset, convert to np.ndarray
    else:
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

        # check images, targets and meta are same length
        if len(image_list) != len(targets):
            raise ValueError("Images and targets must be same length.")

        # check not empty data
        if not len(image_list) > 0:
            raise ValueError("Images should not be empty.")

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
        elif isinstance(image_list[0], List):
            images = np.asarray(image_list).astype(np.float32)

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
    return images, targets, metadata
