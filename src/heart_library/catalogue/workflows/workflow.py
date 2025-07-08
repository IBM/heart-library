"""Abstract HEART Workflow"""

from abc import ABC, abstractmethod
from typing import Any, Optional

from heart_library.catalogue.manager.mlflow.client import MLFlowClient


class Workflow(ABC):
    """This class defines the abstract base class for Workflows within the Catalogue."""

    def __init__(
        self,
        local_catalogue_path: Optional[str] = None,
    ) -> None:
        """Initialize the Workflow with a Catalogue Manager."""
        if local_catalogue_path is not None:
            self._manager = MLFlowClient(local_catalogue_path=local_catalogue_path)
        else:
            raise ValueError("A local patch catalogue path must be given.")

    @abstractmethod
    def run(self, *args: Any, **kwargs: Any) -> bool:  # noqa: ANN401
        """Abstract method for running the workflow."""
        raise NotImplementedError
