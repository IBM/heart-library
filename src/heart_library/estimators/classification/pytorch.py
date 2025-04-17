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
"""This module implements a JATIC compatible ART PyTorchClassifier."""

import uuid
from collections.abc import Sequence
from typing import Any, Optional

from art.estimators.classification.pytorch import PyTorchClassifier
from maite.protocols import ArrayLike

from heart_library.utils import process_inputs_for_art


class JaticPyTorchClassifier(PyTorchClassifier):
    """JATIC compatible extension of ART core PyTorchClassifier

    Args:
        PyTorchClassifier (PyTorchClassifier): ART PyTorchClassifier.

    Examples
    --------

    We can create a default JaticPyTorchClassifier without a provider,
        using a specified model, loss function, and optimizer:

    >>> from torchvision.models import resnet18, ResNet18_Weights
    >>> from heart_library.estimators.classification.pytorch import JaticPyTorchClassifier
    >>> import torch

    Define the JaticPyTorchClassifier inputs, in this case for image classification:

    >>> model = resnet18(ResNet18_Weights)
    >>> loss_fn = torch.nn.CrossEntropyLoss(reduction="sum")
    >>> optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    >>> jptc = JaticPyTorchClassifier(
    ...     model=model,
    ...     loss=loss_fn,
    ...     optimizer=optimizer,
    ...     input_shape=(3, 32, 32),
    ...     nb_classes=10,
    ...     clip_values=(0, 255),
    ...     preprocessing=(0.0, 255),
    ... )
    >>> jptc.model.conv1
    Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    """

    metadata: dict[str, Any]

    def _init_huggingface(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        from art.estimators.classification.hugging_face import HuggingFaceClassifierPyTorch
        from torch.optim import Optimizer
        from transformers import AutoModelForImageClassification

        model = AutoModelForImageClassification.from_pretrained(kwargs["model"])
        if "optimizer" in kwargs and not isinstance(kwargs["optimizer"], Optimizer):
            kwargs["optimizer"] = kwargs["optimizer"](model.parameters(), lr=kwargs.pop("learning_rate", 0.01))
        kwargs["model"] = model
        hf_model = HuggingFaceClassifierPyTorch(**kwargs)
        kwargs["model"] = hf_model.model
        return kwargs

    def _init_timm(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        from art.estimators.classification.hugging_face import HuggingFaceClassifierPyTorch
        from timm import create_model
        from torch.optim import Optimizer

        model = create_model(
            kwargs["model"],
            pretrained=True,
            num_classes=kwargs["nb_classes"],
        )
        kwargs["model"] = model
        if "optimizer" in kwargs and not isinstance(kwargs["optimizer"], Optimizer):
            kwargs["optimizer"] = kwargs["optimizer"](
                kwargs["model"].parameters(),
                lr=kwargs.pop("learning_rate", 0.01),
            )
        hf_model = HuggingFaceClassifierPyTorch(**kwargs)
        kwargs["model"] = hf_model.model
        return kwargs

    def __init__(
        self,
        provider: str = "",
        id: Optional[str] = None,  # noqa ANN401
        metadata: Optional[dict[str, Any]] = None,
        **kwargs: Any,  # noqa ANN401
    ) -> None:
        """JaticPyTorchClassifier initialization.

        Args:
            provider (str, optional): Model framework to use - huggingface/timm Defaults to "".
        """
        if metadata:
            self.metadata = metadata
            self.metadata["id"] = id if id is not None else str(uuid.uuid4())
        else:
            self.metadata = {"id": id if id is not None else str(uuid.uuid4())}

        if provider == "huggingface":
            kwargs = self._init_huggingface(kwargs)
        elif provider == "timm":
            kwargs = self._init_timm(kwargs)

        super().__init__(**kwargs)

    def __call__(self, data: Sequence[ArrayLike]) -> Sequence[ArrayLike]:
        """Convert JATIC supported data to ART supported data and perform prediction.

        Args:
            data (Sequence[ArrayLike]): Array of images, targets, metadata.

        Returns:
            Sequence[ArrayLike]: Array of predictions.
        """

        # convert to ART supported type
        images, _, _ = process_inputs_for_art(data)

        # make prediction
        output = self.predict(images)

        # return as a sequence of N TargetType instances
        return list(output)

    def train(self, data: Sequence[ArrayLike], **kwargs: Any) -> None:  # noqa ANN401
        """Train the model using JATIC supported data format

        Args:
            data (Sequence[ArrayLike]): Array of images, targets, metadata.

        Returns:
            None
        """
        # convert to ART supported type
        images, labels, _ = process_inputs_for_art(data)

        # train the model
        self.fit(x=images, y=labels, **kwargs)
