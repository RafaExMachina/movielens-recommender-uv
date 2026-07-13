"""Testes do serviço de Model Registry."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from recommender.tracking.model_registry import ModelRegistryService


def create_finished_run() -> SimpleNamespace:
    """Cria um run finalizado para os testes."""
    return SimpleNamespace(info=SimpleNamespace(status="FINISHED"))


def create_ready_version(
    version: str = "1",
    run_id: str = "run-123",
) -> SimpleNamespace:
    """Cria uma versão pronta para os testes."""
    return SimpleNamespace(
        version=version,
        run_id=run_id,
        status="READY",
    )


def test_register_creates_model_version_and_alias() -> None:
    """Uma nova execução deve criar uma versão e atribuir o alias."""
    client = MagicMock()
    ready_version = create_ready_version()

    client.get_run.return_value = create_finished_run()
    client.search_model_versions.return_value = []
    client.get_model_version.return_value = ready_version
    client.get_model_version_by_alias.return_value = ready_version

    service = ModelRegistryService(
        tracking_uri="http://localhost:5000",
        client=client,
    )

    with patch(
        "recommender.tracking.model_registry.mlflow.register_model",
        return_value=ready_version,
    ) as register_model:
        result = service.register(
            model_name="movielens-mlp-recommender",
            model_uri="runs:/run-123/model",
            run_id="run-123",
            alias="candidate",
            description="Modelo neural de teste.",
        )

    register_model.assert_called_once_with(
        model_uri="runs:/run-123/model",
        name="movielens-mlp-recommender",
        await_registration_for=300,
    )

    client.set_registered_model_alias.assert_called_once_with(
        "movielens-mlp-recommender",
        "candidate",
        "1",
    )

    assert result.version == "1"
    assert result.status == "READY"
    assert result.alias == "candidate"
    assert result.created_new_version is True
    assert result.registry_model_uri == ("models:/movielens-mlp-recommender/1")


def test_register_reuses_version_from_same_run() -> None:
    """O mesmo run não deve criar versões duplicadas."""
    client = MagicMock()
    existing_version = create_ready_version(
        version="3",
        run_id="run-123",
    )

    client.get_run.return_value = create_finished_run()
    client.search_model_versions.return_value = [existing_version]
    client.get_model_version.return_value = existing_version
    client.get_model_version_by_alias.return_value = existing_version

    service = ModelRegistryService(
        tracking_uri="http://localhost:5000",
        client=client,
    )

    with patch(
        "recommender.tracking.model_registry.mlflow.register_model"
    ) as register_model:
        result = service.register(
            model_name="movielens-mlp-recommender",
            model_uri="runs:/run-123/model",
            run_id="run-123",
            alias="candidate",
            description="Modelo neural de teste.",
        )

    register_model.assert_not_called()

    assert result.version == "3"
    assert result.created_new_version is False


def test_register_rejects_unfinished_run() -> None:
    """Runs não finalizados não podem ser registrados."""
    client = MagicMock()
    client.get_run.return_value = SimpleNamespace(
        info=SimpleNamespace(status="RUNNING")
    )

    service = ModelRegistryService(
        tracking_uri="http://localhost:5000",
        client=client,
    )

    with pytest.raises(RuntimeError, match="não está finalizado"):
        service.register(
            model_name="movielens-mlp-recommender",
            model_uri="runs:/run-123/model",
            run_id="run-123",
            alias="candidate",
            description="Modelo neural de teste.",
        )


def test_register_rejects_inconsistent_model_uri() -> None:
    """O URI deve pertencer ao run informado."""
    client = MagicMock()

    service = ModelRegistryService(
        tracking_uri="http://localhost:5000",
        client=client,
    )

    with pytest.raises(ValueError, match="não corresponde"):
        service.register(
            model_name="movielens-mlp-recommender",
            model_uri="runs:/outro-run/model",
            run_id="run-123",
            alias="candidate",
            description="Modelo neural de teste.",
        )
