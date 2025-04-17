"""This module implements a JATIC compatible ART DeRandomized Smoothing Certified Defense."""

import uuid
from collections.abc import Sequence
from typing import Any, Optional

from art.estimators.certification.derandomized_smoothing import PyTorchDeRandomizedSmoothing
from maite.protocols import ArrayLike

from heart_library.utils import process_inputs_for_art


class DRSJaticPyTorchClassifier(PyTorchDeRandomizedSmoothing):
    """JATIC compatible extension of ART core PyTorchDeRandomizedSmoothing

    Args:
        PyTorchDeRandomizedSmoothing (PyTorchDeRandomizedSmoothing): ART PyTorchDeRandomizedSmoothing.

    Examples
    --------

    We can create a DRSJaticPyTorchClassifier,
        using a timm ViT model, loss function, and optimizer:

    >>> import timm
    >>> from heart_library.estimators.classification.certification import DRSJaticPyTorchClassifier
    >>> import torch

    Define the DRSJaticPyTorchClassifier inputs, in this case for image classification:

    >>> model = timm.create_model("vit_small_patch16_224")
    >>> loss_fn = torch.nn.CrossEntropyLoss()
    >>> optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    >>> cjptc = DRSJaticPyTorchClassifier(
    ...     model=model,
    ...     loss=loss_fn,
    ...     optimizer=optimizer,
    ...     input_shape=(3, 224, 224),
    ...     nb_classes=1000,
    ...     clip_values=(0, 1),
    ...     ablation_size=32,
    ...     replace_last_layer=True,
    ...     load_pretrained=True,
    ... )
    """

    metadata: dict[str, Any]

    def __init__(self, id: Optional[str] = None, **kwargs: Any) -> None:  # noqa ANN401
        """DRSJaticPyTorchClassifier initialization.

        Args:
            id (str, optional): the (optional) id to identify the model in metadata.
        """
        self.metadata = {"id": id if id is not None else str(uuid.uuid4())}
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

    def apply_defense(self, training_data: Sequence[ArrayLike], **kwargs: Any) -> None:  # noqa ANN401
        """Apply the defense. In this case, the model undergoes training to
        enhance certified robustness.

        Args:
            training_data (Sequence[ArrayLike]): Array of images, targets, metadata.

        Returns:
            None
        """
        # transform the data for ART
        x_train, y_train, _ = process_inputs_for_art(training_data)

        # apply the fit for certification
        self.fit(x=x_train, y=y_train, **kwargs)
