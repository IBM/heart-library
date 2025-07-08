"""Search Patches"""

import logging
from typing import Any, Optional

import numpy as np
from mlflow import search_runs
from mlflow.artifacts import load_image
from numpy.typing import NDArray

from heart_library.catalogue.workflows.workflow import Workflow

logger: logging.Logger = logging.getLogger(__name__)


class SearchPatches(Workflow):
    """
    The SearchPatch workflow can be used to optimize an existing foundation patch
    on given datasets and/or for given models.
    """

    def __init__(self, local_catalogue_path: Optional[str] = None) -> None:
        """Initialise the MLFlow client."""
        logger.info("Setting up workflow for searching patches.")
        super().__init__(local_catalogue_path=local_catalogue_path)

    def run(
        self,
        patch_types: list[str],
        patch_filter: str = "",
    ) -> tuple[Any, list[NDArray[np.float32]]]:
        """Run the patch search workflow."""
        logger.info("Searching for patches.")

        if patch_filter == "":
            runs = search_runs(experiment_names=patch_types)
        else:
            runs = search_runs(experiment_names=patch_types, filter_string=patch_filter)

        patches = []
        if len(runs) > 0:
            for _, run in runs.iterrows():
                patch = load_image(f"{run.artifact_uri}/training_patch.png")
                patches.append(patch)

        return runs, patches
