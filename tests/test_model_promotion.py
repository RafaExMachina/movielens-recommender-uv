"""Testes do serviço de promoção de modelos."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from mlflow.exceptions import MlflowException

from recommender.tracking.model_promotion import (
    ModelPromotionService,
)


def create_model_version(
    *,
    version: str = "1",
    run_id: str = "run-123",
    status: str = "READY",
    current_stage: str = "None",
) -> SimpleNamespace:
    """Cria uma versão simulada do MLflow.

    Args:
        version: Número da versão.
        run_id: Identificador da execução de origem.
        status: Status da versão.
        current_stage: Estágio legado do MLflow.

    Returns:
        Objeto que representa uma versão registrada.
    """
    return SimpleNamespace(
        version=version,
        run_id=run_id,
        status=status,
        current_stage=current_stage,
    )


def test_promote_to_staging_sets_alias_and_stage() -> None:
    """A promoção para staging deve configurar alias e estágio."""
    client = MagicMock()

    ready_version = create_model_version()
    staging_version = create_model_version(
        current_stage="Staging",
    )

    client.get_model_version.return_value = ready_version

    client.get_model_version_by_alias.side_effect = [
        MlflowException("Alias staging ainda não existe."),
        ready_version,
    ]

    client.transition_model_version_stage.return_value = staging_version

    service = ModelPromotionService(
        tracking_uri="http://localhost:5000",
        client=client,
    )

    result = service.promote(
        model_name="movielens-mlp-recommender",
        version="1",
        run_id="run-123",
        target="staging",
        staging_alias="staging",
        production_alias="production",
        approved_by="Rafael Costa",
        approval_notes="Modelo encaminhado para staging.",
        require_staging_before_production=True,
        sync_legacy_stage=True,
    )

    client.set_registered_model_alias.assert_called_once_with(
        "movielens-mlp-recommender",
        "staging",
        "1",
    )

    client.transition_model_version_stage.assert_called_once_with(
        name="movielens-mlp-recommender",
        version="1",
        stage="Staging",
        archive_existing_versions=True,
    )

    assert result.registered_model_name == ("movielens-mlp-recommender")
    assert result.version == "1"
    assert result.run_id == "run-123"
    assert result.target == "staging"
    assert result.alias == "staging"
    assert result.legacy_stage == "Staging"
    assert result.already_promoted is False
    assert result.previous_alias_version is None
    assert result.approved_by == "Rafael Costa"
    assert result.alias_model_uri == ("models:/movielens-mlp-recommender@staging")


def test_promote_to_production_requires_staging() -> None:
    """Production deve exigir a mesma versão em staging."""
    client = MagicMock()

    client.get_model_version.return_value = create_model_version()

    client.get_model_version_by_alias.side_effect = MlflowException(
        "Alias staging não encontrado."
    )

    service = ModelPromotionService(
        tracking_uri="http://localhost:5000",
        client=client,
    )

    with pytest.raises(
        RuntimeError,
        match="esteja primeiro",
    ):
        service.promote(
            model_name="movielens-mlp-recommender",
            version="1",
            run_id="run-123",
            target="production",
            staging_alias="staging",
            production_alias="production",
            approved_by="Rafael Costa",
            approval_notes="Modelo aprovado para produção.",
            require_staging_before_production=True,
            sync_legacy_stage=True,
        )

    client.set_registered_model_alias.assert_not_called()


def test_promote_to_production_after_staging() -> None:
    """Uma versão em staging deve avançar para production."""
    client = MagicMock()

    ready_version = create_model_version()
    production_version = create_model_version(
        current_stage="Production",
    )

    client.get_model_version.return_value = ready_version

    client.get_model_version_by_alias.side_effect = [
        ready_version,
        MlflowException("Alias production ainda não existe."),
        ready_version,
    ]

    client.transition_model_version_stage.return_value = production_version

    service = ModelPromotionService(
        tracking_uri="http://localhost:5000",
        client=client,
    )

    result = service.promote(
        model_name="movielens-mlp-recommender",
        version="1",
        run_id="run-123",
        target="production",
        staging_alias="staging",
        production_alias="production",
        approved_by="Rafael Costa",
        approval_notes=("Modelo aprovado após passagem por staging."),
        require_staging_before_production=True,
        sync_legacy_stage=True,
    )

    client.set_registered_model_alias.assert_called_once_with(
        "movielens-mlp-recommender",
        "production",
        "1",
    )

    client.transition_model_version_stage.assert_called_once_with(
        name="movielens-mlp-recommender",
        version="1",
        stage="Production",
        archive_existing_versions=True,
    )

    assert result.target == "production"
    assert result.alias == "production"
    assert result.legacy_stage == "Production"
    assert result.already_promoted is False
    assert result.alias_model_uri == ("models:/movielens-mlp-recommender@production")


def test_promote_identifies_existing_alias() -> None:
    """Uma promoção repetida deve ser identificada."""
    client = MagicMock()

    ready_version = create_model_version()
    staging_version = create_model_version(
        current_stage="Staging",
    )

    client.get_model_version.return_value = ready_version

    client.get_model_version_by_alias.side_effect = [
        ready_version,
        ready_version,
    ]

    client.transition_model_version_stage.return_value = staging_version

    service = ModelPromotionService(
        tracking_uri="http://localhost:5000",
        client=client,
    )

    result = service.promote(
        model_name="movielens-mlp-recommender",
        version="1",
        run_id="run-123",
        target="staging",
        staging_alias="staging",
        production_alias="production",
        approved_by="Rafael Costa",
        approval_notes="Promoção repetida para staging.",
        require_staging_before_production=True,
        sync_legacy_stage=True,
    )

    assert result.already_promoted is True
    assert result.previous_alias_version == "1"


def test_promote_rejects_run_mismatch() -> None:
    """A versão deve pertencer ao run informado."""
    client = MagicMock()

    client.get_model_version.return_value = create_model_version(
        run_id="outro-run",
    )

    service = ModelPromotionService(
        tracking_uri="http://localhost:5000",
        client=client,
    )

    with pytest.raises(
        RuntimeError,
        match="não corresponde",
    ):
        service.promote(
            model_name="movielens-mlp-recommender",
            version="1",
            run_id="run-123",
            target="staging",
            staging_alias="staging",
            production_alias="production",
            approved_by="Rafael Costa",
            approval_notes="Teste de run incompatível.",
            require_staging_before_production=True,
            sync_legacy_stage=True,
        )


def test_promote_rejects_unready_version() -> None:
    """Versões não prontas não podem ser promovidas."""
    client = MagicMock()

    client.get_model_version.return_value = create_model_version(
        status="PENDING_REGISTRATION",
    )

    service = ModelPromotionService(
        tracking_uri="http://localhost:5000",
        client=client,
    )

    with pytest.raises(
        RuntimeError,
        match="não está pronta",
    ):
        service.promote(
            model_name="movielens-mlp-recommender",
            version="1",
            run_id="run-123",
            target="staging",
            staging_alias="staging",
            production_alias="production",
            approved_by="Rafael Costa",
            approval_notes="Teste de versão não pronta.",
            require_staging_before_production=True,
            sync_legacy_stage=True,
        )


def test_promote_rejects_invalid_target() -> None:
    """O destino deve ser staging ou production."""
    client = MagicMock()

    service = ModelPromotionService(
        tracking_uri="http://localhost:5000",
        client=client,
    )

    with pytest.raises(
        ValueError,
        match="target deve ser",
    ):
        service.promote(
            model_name="movielens-mlp-recommender",
            version="1",
            run_id="run-123",
            target="invalid",  # type: ignore[arg-type]
            staging_alias="staging",
            production_alias="production",
            approved_by="Rafael Costa",
            approval_notes="Destino inválido.",
            require_staging_before_production=True,
            sync_legacy_stage=True,
        )
