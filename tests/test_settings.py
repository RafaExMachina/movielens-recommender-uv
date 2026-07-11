"""Testes das configurações de ambiente."""

import pytest
from pydantic import ValidationError

from recommender.utils.settings import Settings


def test_settings_use_default_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Deve utilizar valores padrão quando não há variáveis definidas."""
    monkeypatch.delenv("APP_ENV", raising=False)
    monkeypatch.delenv("LOG_LEVEL", raising=False)
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)

    settings = Settings(_env_file=None)

    assert settings.app_env == "development"
    assert settings.log_level == "INFO"
    assert settings.mlflow_tracking_uri == "http://localhost:5000"


def test_environment_variables_override_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Variáveis de ambiente devem sobrescrever os valores padrão."""
    monkeypatch.setenv("APP_ENV", "test")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    monkeypatch.setenv(
        "MLFLOW_TRACKING_URI",
        "http://mlflow:5000",
    )

    settings = Settings(_env_file=None)

    assert settings.app_env == "test"
    assert settings.log_level == "DEBUG"
    assert settings.mlflow_tracking_uri == "http://mlflow:5000"


def test_invalid_environment_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Deve rejeitar um nome de ambiente não permitido."""
    monkeypatch.setenv("APP_ENV", "invalid")

    with pytest.raises(ValidationError):
        Settings(_env_file=None)
