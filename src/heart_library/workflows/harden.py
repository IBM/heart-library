"""Module describing hardening workflow."""

from typing import Any, Optional, Union, cast

import maite.protocols.image_classification as ic
from maite._internals.protocols.generic import MetricComputeReturnType
from timm.models.vision_transformer import VisionTransformer

from heart_library.estimators.classification.certification.derandomized_smoothing import DRSJaticPyTorchClassifier


def harden(
    *,
    model: ic.Model,
    data: Optional[Union[ic.DataLoader, ic.Dataset]] = None,
) -> tuple[DRSJaticPyTorchClassifier, Optional[MetricComputeReturnType]]:
    """
    Apply a defense to a model. Currently, only certified defense is supported.

    Parameters
    ----------
    model : A MAITE compliant image classification model
        Maite ic.Model object.

    data : Dataset | Dataloader, (default=None)
        Compatible maite aligned dataset or dataloader.

    Returns
    -------
    tuple[DRSJaticPyTorchClassifier, Optional[dict[str, Any]]]
        Tuple of returned maite compliant hardened model and an optional metric value
        associated with hardening.
    """
    metric_result: Optional[MetricComputeReturnType] = None
    # check if the model has been equipped with a defense
    defense: Optional[dict[str, Any]] = cast(Optional[dict[str, Any]], model.metadata.get("defense", None))

    hardened_model: Optional[DRSJaticPyTorchClassifier] = None
    # switch on the type of defense
    # is a preprocessing defense,
    if defense and defense["class"] == "DRSJaticPyTorchClassifier":
        if data is None:
            raise ValueError("Data must not be None.")
        if isinstance(model.model, VisionTransformer):
            hardened_model = DRSJaticPyTorchClassifier(
                model=model.model.pretrained_cfg["architecture"],
                **defense.get("drs_kwargs", {}),
            )
            hardened_model.model.load_state_dict(model.model.state_dict())
        else:
            raise ValueError("Model architecture is not supported for certification defense.")
        train_kwargs = defense.get("train_kwargs", {})
        if "scheduler" in train_kwargs and "scheduler_params" in train_kwargs:
            scheduler = train_kwargs.pop("scheduler")(
                optimizer=model.optimizer,
                **train_kwargs.pop("scheduler_params"),
            )
            train_kwargs["scheduler"] = scheduler
        hardened_model.apply_defense(training_data=data, **train_kwargs)
        return (hardened_model, metric_result)

    raise ValueError(
        """To use this workflow, the model must be equipped with a supported defense
                         described in Model metadata.""",
    )
