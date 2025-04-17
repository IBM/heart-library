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

from tests.utils import HEARTTestError

logger = logging.getLogger(__name__)


@pytest.mark.optional
@pytest.mark.skipif(importlib.util.find_spec("timm") is None, reason="timm is not installed")
def test_harden_workflow(heart_warning):
    try:
        from typing import Any

        import maite.protocols.image_classification as ic
        import numpy as np
        import torch
        from torchvision import transforms

        from heart_library.estimators.classification.pytorch import JaticPyTorchClassifier
        from heart_library.workflows.harden import harden

        (x_train, _), (_, _), _, _ = load_dataset("cifar10")

        class ImageDataset:
            def __init__(self, images: np.ndarray, labels: np.ndarray):
                self.images = images
                self.labels = labels

            def __len__(self) -> int:
                return len(self.images)

            def __getitem__(self, ind: int) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
                image: np.ndarray = self.images[ind]
                label: np.ndarray = self.labels[ind]
                return (image, label, {})

        img = x_train[[1]].transpose(0, 3, 1, 2).astype("float32")
        img = np.resize(img, (1, 3, 224, 224))
        dataset = ImageDataset(images=img, labels=np.array([5]))

        metadata = {
            "defense": {
                "class": "DRSJaticPyTorchClassifier",
                "drs_kwargs": {
                    "loss": torch.nn.CrossEntropyLoss(reduction="sum"),
                    "optimizer": torch.optim.Adam,
                    "optimizer_params": {"lr": 1e-4},
                    "input_shape": (3, 224, 224),
                    "nb_classes": 10,
                    "clip_values": (0, 1),
                    "ablation_size": 50,
                    "replace_last_layer": True,
                    "load_pretrained": True,
                },
                "train_kwargs": {
                    "nb_epochs": 100,
                    "verbose": True,
                    "scheduler": torch.optim.lr_scheduler.MultiStepLR,
                    "scheduler_params": {"milestones": [20, 40], "gamma": 0.1},
                    "transform": transforms.Compose([transforms.RandomHorizontalFlip()]),
                },
            },
        }

        jptc = JaticPyTorchClassifier(
            model="vit_small_patch16_224",
            provider="timm",
            loss=torch.nn.CrossEntropyLoss(),
            input_shape=(3, 224, 224),
            nb_classes=10,
            clip_values=(0, 1),
            optimizer=torch.optim.Adam,
            learning_rate=1e-4,
            metadata=metadata,
        )

        with pytest.raises(ValueError, match="""Data must not be None."""):
            hardened_model, _ = harden(model=jptc, data=None)

        hardened_model, _ = harden(model=jptc, data=dataset)

        isinstance(hardened_model, ic.Model)

        jptc = JaticPyTorchClassifier(
            model="vit_small_patch16_224",
            provider="timm",
            loss=torch.nn.CrossEntropyLoss(),
            input_shape=(3, 224, 224),
            nb_classes=10,
            clip_values=(0, 1),
            optimizer=torch.optim.Adam,
            learning_rate=1e-4,
        )
        with pytest.raises(
            ValueError,
            match="""To use this workflow, the model must be equipped with a supported defense
                         described in Model metadata.""",
        ):
            hardened_model, _ = harden(model=jptc, data=dataset)

    except HEARTTestError as e:
        heart_warning(e)
