"""Configurações de ambiente da aplicação."""

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

AppEnvironment = Literal["development", "test", "production"]
LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


class Settings(BaseSettings):
    """Carrega e valida configurações específicas do ambiente.

    As configurações podem ser fornecidas por variáveis de ambiente
    ou por um arquivo local ``.env``. Variáveis de ambiente possuem
    precedência sobre os valores presentes no arquivo.

    Attributes:
        app_env: Ambiente atual da aplicação.
        log_level: Nível de logging utilizado pela aplicação.
        mlflow_tracking_uri: Endereço do servidor de tracking do MLflow.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        frozen=True,
    )

    app_env: AppEnvironment = "development"
    log_level: LogLevel = "INFO"
    mlflow_tracking_uri: str = Field(
        default="http://localhost:5000",
        min_length=1,
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Retorna uma instância validada e armazenada em cache.

    Returns:
        Configurações atuais da aplicação.
    """
    return Settings()
