import importlib
import logging

import pytest

from tests.utils import HEARTTestError

logger = logging.getLogger(__name__)


@pytest.mark.optional
@pytest.mark.skipif(importlib.util.find_spec("mlflow") is None, reason="catalogue dependencies not installed")
def test_catalogue_search(heart_warning):
    try:
        from heart_library.catalogue.manager import CatalogueManager
        from heart_library.catalogue.workflows import SearchPatches
        from heart_library.catalogue.workflows.workflow import Workflow

        with pytest.raises(ValueError, match="A local patch catalogue path must be given."):
            workflow = SearchPatches()

        workflow = SearchPatches(local_catalogue_path="tests/catalogue/patches")

        workflow._manager._run = True  # noqa: SLF001
        meta, _ = workflow.run(["FOUNDATION"])

        assert meta.iloc[0].run_id == "31fa085f78424d93b0a0865c6e529422"

        assert isinstance(workflow, Workflow)
        assert isinstance(workflow._manager, CatalogueManager)  # noqa: SLF001

        meta, _ = workflow.run(["FOUNDATION"], patch_filter="metrics.map_50_adv_vs_pred<0.5")

        assert meta.iloc[0].run_id == "31fa085f78424d93b0a0865c6e529422"

    except HEARTTestError as e:
        heart_warning(e)


@pytest.mark.optional
@pytest.mark.skipif(importlib.util.find_spec("mlflow") is None, reason="catalogue dependencies not installed")
def test_catalogue_create(heart_warning):
    try:
        from heart_library.catalogue.workflows import CreateObjectDetectionPatch
        from heart_library.catalogue.workflows.workflow import Workflow

        workflow = CreateObjectDetectionPatch(local_catalogue_path="tests/catalogue/patches")
        success = workflow.run(model_type="fasterrcnn_resnet50_fpn", num_data_samples=2, max_iter=1)

        assert success
        assert isinstance(workflow, Workflow)

    except HEARTTestError as e:
        heart_warning(e)


@pytest.mark.optional
@pytest.mark.skipif(importlib.util.find_spec("mlflow") is None, reason="catalogue dependencies not installed")
def test_catalogue_optimize(heart_warning):
    try:
        from heart_library.catalogue.workflows import OptimizeObjectDetectionPatch
        from heart_library.catalogue.workflows.workflow import Workflow

        workflow = OptimizeObjectDetectionPatch(local_catalogue_path="tests/catalogue/patches")
        success = workflow.run(
            run_id="31fa085f78424d93b0a0865c6e529422",
            model_type="fasterrcnn_resnet50_fpn",
            num_data_samples=2,
            max_iter=1,
        )

        assert success
        assert isinstance(workflow, Workflow)

    except HEARTTestError as e:
        heart_warning(e)


@pytest.mark.optional
@pytest.mark.skipif(importlib.util.find_spec("mlflow") is None, reason="catalogue dependencies not installed")
def test_catalogue_evaluate(heart_warning):
    try:
        from heart_library.catalogue.workflows import EvaluateObjectDetectionPatch
        from heart_library.catalogue.workflows.workflow import Workflow

        workflow = EvaluateObjectDetectionPatch(local_catalogue_path="tests/catalogue/patches")

        with pytest.raises(NotImplementedError):
            workflow.run()

        assert isinstance(workflow, Workflow)

    except HEARTTestError as e:
        heart_warning(e)
