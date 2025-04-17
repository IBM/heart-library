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
from datasets import Dataset, load_dataset

from tests.utils import HEARTTestError

logger = logging.getLogger(__name__)


# local Hugging Face (hf) slice data captured from remote dataset splits
HF_SLICE_DIR = "utils/resources/datasets"
HF_SLICE_ARROW = "data-00000-of-00001.arrow"

# local hf slice metadata corresponding to saved dataset splits
HF_SLICE_METADATA = {
    "hf_cifar10_test_0-10": {  # directory name for the slice in HF_SLICE_DIR
        "name": "cifar10",  # full hf dataset name including slashes & symbols
        "split": "test[0:10]",  # split used when the slice was saved locally
    },
    "hf_guydada_quickstart-coco_train_25-27": {
        "name": "guydada/quickstart-coco",
        "split": "train[25:27]",
    },
    "hf_AI-Lab-Makerere_beans_test_0-5": {
        "name": "AI-Lab-Makerere/beans",
        "split": "test[0:5]",
    },
    "hf_mnist_test_0-10": {
        "name": "mnist",
        "split": "test[0:10]",
    },
    "hf_squad_validation_0-10": {
        "name": "squad",
        "split": "validation[0:10]",
    },
}

# load hf slices on pytest init that are necessary for unit testing
hf_slices = {}
for hf_slice, metadata in HF_SLICE_METADATA.items():
    hf_name, hf_split = (metadata.get("name"), metadata.get("split"))
    try:
        hf_slices[hf_slice] = Dataset.from_file(f"{HF_SLICE_DIR}/{hf_slice}/{HF_SLICE_ARROW}")
    except OSError:
        logger.warning(f"\n[{hf_slice}] failed to load from: {HF_SLICE_DIR}, attempting remote load via Hugging Face.")
        hf_slices[hf_slice] = load_dataset(hf_name, split=hf_split)


@pytest.mark.required
def test_process_inputs_for_art_using_hf_datasets(heart_warning):
    try:
        import numpy as np
        import torch

        from heart_library.utils import process_inputs_for_art

        # local HF slice data
        hf_cifar10_slice = hf_slices["hf_cifar10_test_0-10"]
        hf_coco_slice = hf_slices["hf_guydada_quickstart-coco_train_25-27"]

        # define HF dataset and convert to art (using local cifar10 test[0:10] slice)
        x, y, m = process_inputs_for_art(hf_cifar10_slice)
        assert isinstance(x, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert isinstance(m, list)

        # define HF dataset, select subset, convert to art (using local cifar10 test[0:10] slice)
        data = torch.utils.data.Subset(hf_cifar10_slice, list(range(10)))
        x, y, m = process_inputs_for_art(data)
        assert isinstance(x, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert isinstance(m, list)

        # huggingface iterable dataset (remote full cifar10 streamed call without local slice)
        from functools import partial

        num_samples = 5
        hf_iterable_cifar10 = load_dataset("cifar10", split="test", streaming=True)
        sample_data = hf_iterable_cifar10.take(num_samples)

        def gen_from_iterable_dataset(iterable_ds):
            yield from iterable_ds

        data = Dataset.from_generator(partial(gen_from_iterable_dataset, sample_data), features=sample_data.features)
        x, _, _ = process_inputs_for_art(sample_data)
        assert isinstance(x, np.ndarray)

        # object detection
        from torchvision.transforms import transforms

        from heart_library.estimators.object_detection import JaticPyTorchObjectDetector

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        preprocessing = (mean, std)

        detector = JaticPyTorchObjectDetector(
            model_type="detr_resnet50",
            device_type="cpu",
            input_shape=(3, 800, 800),
            clip_values=(0, 1),
            attack_losses=(
                "loss_ce",
                "loss_bbox",
                "loss_giou",
            ),
            preprocessing=preprocessing,
        )

        data = hf_coco_slice  # using local guydada/quickstart-coco train[25:27] slice
        preprocess = transforms.Compose(
            [
                transforms.Resize(800),
                transforms.CenterCrop(800),
                transforms.ToTensor(),
            ],
        )

        data = data.map(lambda x: {"image": preprocess(x["image"]), "label": None})
        detections = detector(sample_data)  # using sample_data from hf iterable dataset
        resnet_images = data["image"]
        data = {"images": resnet_images, "labels": detections}
        x, y, m = process_inputs_for_art(data)
        assert isinstance(x, list)
        assert isinstance(y, list)
        assert isinstance(m, list)

        data = {"images": resnet_images, "labels": [np.empty(2)]}
        x, y, m = process_inputs_for_art(data)
        assert isinstance(x, list)
        assert isinstance(y, np.ndarray)
        assert isinstance(m, list)

        data = {"images": resnet_images, "labels": [[]]}
        x, y, m = process_inputs_for_art(data)
        assert isinstance(x, list)
        assert y is None
        assert isinstance(m, list)

        # tuple object detection
        data = resnet_images, detections, []
        x, y, m = process_inputs_for_art(data)
        assert isinstance(x, np.ndarray)
        assert isinstance(y, list)
        assert isinstance(m, list)

        data = resnet_images, [np.empty(2)], []
        x, y, m = process_inputs_for_art(data)
        assert isinstance(x, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert isinstance(m, list)

        data = resnet_images, [[]], []
        x, y, m = process_inputs_for_art(data)
        assert isinstance(x, np.ndarray)
        assert y is None
        assert isinstance(m, list)

        # sequence
        data = [np.asarray(resnet_images)[0]]
        x, y, m = process_inputs_for_art(data)
        assert isinstance(x, np.ndarray)
        assert y is None
        assert isinstance(m, list)

    except HEARTTestError as e:
        heart_warning(e)


@pytest.mark.required
def test_process_inputs_for_art(heart_warning):
    try:
        import numpy as np
        import torch
        import torchvision

        from heart_library.utils import process_inputs_for_art

        # define torchvision dataset and convert to art
        data = torchvision.datasets.CIFAR10("../../data", train=False, download=True)
        data = torch.utils.data.Subset(data, list(range(10)))
        x, y, m = process_inputs_for_art(data)
        assert isinstance(x, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert isinstance(m, list)

        # define dict of data and convert to art
        data = {"images": x, "labels": y, "metadata": m}
        x, y, m = process_inputs_for_art(data)
        assert isinstance(x, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert isinstance(m, list)

        data = {"images": x, "labels": y}
        x, y, _ = process_inputs_for_art(data)
        assert isinstance(x, np.ndarray)
        assert isinstance(y, np.ndarray)

        data = {"images": x}
        x, _, _ = process_inputs_for_art(data)
        assert isinstance(x, np.ndarray)

        # define a tuple and convert to art
        data = (x, y, m)
        x, y, m = process_inputs_for_art(data)
        assert isinstance(x, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert isinstance(m, list)

        targets = [{"boxes": np.array([0]), "scores": np.array([0]), "labels": np.array([0])}]
        data = (x, targets, m)
        x, y, m = process_inputs_for_art(data)
        assert isinstance(x, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert isinstance(m, list)

        # tuple does not contain 3 items
        tuple_data = (x, y)
        with pytest.raises(IndexError):
            x, y, m = process_inputs_for_art(tuple_data)

        # define sequence and convert to art
        data = np.expand_dims(x, axis=0)
        x, y, m = process_inputs_for_art(data)
        assert isinstance(x, np.ndarray)
        assert isinstance(m, list)

        # define a tensor and convert to art
        data = torch.Tensor(x)
        x, y, m = process_inputs_for_art(data)
        assert isinstance(x, np.ndarray)

        # define an np.ndarray and convert to art
        x, y, m = process_inputs_for_art(x)
        assert isinstance(x, np.ndarray)

        # targeted dataset
        from typing import Any

        class TargetedImageDataset:
            def __init__(self, images):
                self.images = images

            def __len__(self) -> int:
                return len(self.images)

            def __getitem__(self, ind: int) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
                image = self.images[ind]
                return image, 2, {}

        data = TargetedImageDataset(x)
        x, y, m = process_inputs_for_art(data)
        assert isinstance(x, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert isinstance(m, list)

    except HEARTTestError as e:
        heart_warning(e)


@pytest.mark.required
def test_process_inputs_for_art_with_torchvision_transforms(heart_warning):
    try:
        from typing import Any

        import numpy as np
        import torch

        # dataset
        from torchvision.transforms import transforms

        from heart_library.utils import process_inputs_for_art

        def to_image(x: np.ndarray[np.float32]) -> Any:  # noqa ANN401
            return transforms.ToPILImage()(torch.Tensor(x))

        with pytest.raises(ValueError, match="Dataset must implement __getitem__ or __iter__."):
            process_inputs_for_art(to_image)

    except HEARTTestError as e:
        heart_warning(e)


@pytest.mark.required
def test_hf_dataset_to_maite(heart_warning):
    try:
        import maite.protocols.image_classification as ic

        from heart_library.utils import hf_dataset_to_maite

        # load local HF slice data
        hf_cifar10_slice = hf_slices["hf_cifar10_test_0-10"]
        hf_beans_slice = hf_slices["hf_AI-Lab-Makerere_beans_test_0-5"]
        hf_mnist_slice = hf_slices["hf_mnist_test_0-10"]
        hf_squad_slice = hf_slices["hf_squad_validation_0-10"]

        # define HF dataset and convert to art (using local cifar10 test[0:10] slice)
        maite_dataset = hf_dataset_to_maite(hf_cifar10_slice)
        assert isinstance(maite_dataset, ic.Dataset)

        meta_label = "image_file_path"
        image_label = "image"
        target_label = "label"

        # using local mnist test[0:10] slice
        maite_dataset = hf_dataset_to_maite(
            hf_mnist_slice,
            image_label=image_label,
            target_label=target_label,
        )

        assert isinstance(maite_dataset, ic.Dataset)

        # indices
        indices = range(5)

        # image label in dataset features (using local mnist test[0:10] slice)
        maite_dataset = hf_dataset_to_maite(
            hf_mnist_slice,
            image_label=image_label,
            target_label=target_label,
            indices=indices,
        )

        assert isinstance(maite_dataset, ic.Dataset)

        # "image" in dataset features (using local mnist test[0:10] slice)
        maite_dataset = hf_dataset_to_maite(
            hf_mnist_slice,
            target_label=target_label,
            indices=indices,
        )
        assert isinstance(maite_dataset, ic.Dataset)

        # meta

        # using local AI-Lab-Makerere/beans test[0:5] slice
        maite_dataset = hf_dataset_to_maite(
            hf_beans_slice,
            image_label=image_label,
            meta_label=meta_label,
        )
        assert isinstance(maite_dataset, ic.Dataset)

        # using local AI-Lab-Makerere/beans test[0:5] slice
        maite_dataset = hf_dataset_to_maite(
            hf_beans_slice,
            image_label=image_label,
            meta_label=meta_label,
            indices=indices,
        )
        assert isinstance(maite_dataset, ic.Dataset)

    except HEARTTestError as e:
        heart_warning(e)

    with pytest.raises(ValueError, match="Image feature not found in dataset."):
        # using local squad validation[0:10] slice
        maite_dataset = hf_dataset_to_maite(hf_squad_slice)

    with pytest.raises(ValueError, match="Image feature not found in dataset."):
        # using local squad validation[0:10] slice
        maite_dataset = hf_dataset_to_maite(hf_squad_slice, indices=indices)


@pytest.mark.required
def test_subset_to_maite(heart_warning):
    try:
        import maite.protocols.image_classification as ic
        import torch
        import torchvision

        # define torchvision subset dataset and convert to art
        data = torchvision.datasets.CIFAR10("../data", train=False, download=True)
        maite_dataset = torch.utils.data.Subset(data, list(range(10)))
        # require id metadata for dataset
        maite_dataset.metadata = {"id": "test_id"}
        assert isinstance(maite_dataset, ic.Dataset)

    except HEARTTestError as e:
        heart_warning(e)


@pytest.mark.required
def test_image_dataset(heart_warning):
    try:
        import maite.protocols.image_classification as ic
        import numpy as np

        from heart_library.utils import ImageDataset

        x = [np.array([0])]
        y = np.array([0])
        m = [{}]
        dataset = ImageDataset(x, y, m)
        assert isinstance(dataset, ic.Dataset)
        x1, y1, m1 = dataset[0]
        assert x1 == x[0]
        assert y1 == y[0]
        assert m1 == m[0]
        assert len(dataset) == 1

    except HEARTTestError as e:
        heart_warning(e)
