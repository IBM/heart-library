"""Patch Catalogue Evaluation"""

import logging

from heart_library.catalogue.workflows.workflow import Workflow

logger: logging.Logger = logging.getLogger(__name__)


class EvaluateObjectDetectionPatch(Workflow):
    """
    The EvaluatePatch workflow can be used to evaluate a foundation patch
    across a variety of models, metrics and datasets.
    """

    def __init__(self, local_catalogue_path: str) -> None:
        """Initialise the MLFlow client."""
        logger.info("Setting up workflow for evaluating patches.")
        super().__init__(local_catalogue_path=local_catalogue_path)

    def run(self) -> bool:
        """Run the patch evaluation workflow."""
        logger.info("Running patch evaluation workflow")
        raise NotImplementedError
