"""Serviço de registro de modelos no MLflow Model Registry."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

import mlflow
from mlflow import MlflowClient
from mlflow.entities.model_registry import ModelVersion

_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")


@dataclass(frozen=True)
class RegistryResult:
    """Resultado do registro de uma versão no Model Registry."""

    registered_model_name: str
    version: str
    run_id: str
    source_model_uri: str
    registry_model_uri: str
    alias: str
    alias_model_uri: str
    status: str
    tracking_uri: str
    created_new_version: bool

    def to_dict(self) -> dict[str, Any]:
        """Converte o resultado para um dicionário serializável.

        Returns:
            Dados do modelo registrado.
        """
        return asdict(self)


class ModelRegistryService:
    """Gerencia o registro e a identificação de versões no MLflow."""

    def __init__(
        self,
        tracking_uri: str,
        client: MlflowClient | None = None,
    ) -> None:
        """Configura os clientes de tracking e registry.

        Args:
            tracking_uri: Endereço do servidor MLflow.
            client: Cliente injetado, usado principalmente em testes.
        """
        self.tracking_uri = self._require_text(
            tracking_uri,
            "tracking_uri",
        ).rstrip("/")

        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_registry_uri(self.tracking_uri)

        self.client = client or MlflowClient()

    def register(
        self,
        *,
        model_name: str,
        model_uri: str,
        run_id: str,
        alias: str,
        description: str,
        await_registration_seconds: int = 300,
        registered_model_tags: Mapping[str, str] | None = None,
        model_version_tags: Mapping[str, str] | None = None,
    ) -> RegistryResult:
        """Registra uma versão de modelo de forma idempotente.

        Quando uma versão vinculada ao mesmo run já existe, ela é
        reutilizada em vez de criar uma duplicata.

        Args:
            model_name: Nome do modelo no Model Registry.
            model_uri: URI do artefato no formato ``runs:/...``.
            run_id: Identificador do run de treinamento.
            alias: Alias inicial atribuído à versão.
            description: Descrição do modelo registrado.
            await_registration_seconds: Tempo máximo de espera pelo registro.
            registered_model_tags: Tags aplicadas ao modelo registrado.
            model_version_tags: Tags aplicadas à versão criada.

        Returns:
            Metadados da versão registrada.

        Raises:
            ValueError: Se algum parâmetro for inválido.
            RuntimeError: Se o run ou a versão não estiverem prontos.
        """
        normalized_model_name = self._validate_name(
            model_name,
            "model_name",
        )
        normalized_alias = self._validate_name(
            alias,
            "alias",
        )
        normalized_run_id = self._require_text(
            run_id,
            "run_id",
        )
        normalized_model_uri = self._require_text(
            model_uri,
            "model_uri",
        )
        normalized_description = self._require_text(
            description,
            "description",
        )

        if await_registration_seconds <= 0:
            message = "await_registration_seconds deve ser maior que zero."
            raise ValueError(message)

        expected_prefix = f"runs:/{normalized_run_id}/"

        if not normalized_model_uri.startswith(expected_prefix):
            message = (
                "model_uri não corresponde ao run_id informado. "
                f"Esperado prefixo: {expected_prefix}"
            )
            raise ValueError(message)

        run = self.client.get_run(normalized_run_id)
        run_status = str(run.info.status)

        if run_status != "FINISHED":
            message = (
                f"O run {normalized_run_id} não está finalizado. "
                f"Status atual: {run_status}"
            )
            raise RuntimeError(message)

        existing_version = self._find_version_by_run(
            normalized_model_name,
            normalized_run_id,
        )

        created_new_version = existing_version is None

        model_version: ModelVersion

        if existing_version is None:
            model_version = mlflow.register_model(
                model_uri=normalized_model_uri,
                name=normalized_model_name,
                await_registration_for=await_registration_seconds,
            )
        else:
            model_version = existing_version

        version = str(model_version.version)

        self.client.update_registered_model(
            name=normalized_model_name,
            description=normalized_description,
        )

        for key, value in (registered_model_tags or {}).items():
            self.client.set_registered_model_tag(
                normalized_model_name,
                key,
                value,
            )

        for key, value in (model_version_tags or {}).items():
            self.client.set_model_version_tag(
                normalized_model_name,
                version,
                key,
                value,
            )

        ready_version = self.client.get_model_version(
            normalized_model_name,
            version,
        )
        status = str(ready_version.status)

        if status != "READY":
            message = (
                f"A versão {version} do modelo "
                f"{normalized_model_name!r} não está pronta. "
                f"Status atual: {status}"
            )
            raise RuntimeError(message)

        self.client.set_registered_model_alias(
            normalized_model_name,
            normalized_alias,
            version,
        )

        alias_version = self.client.get_model_version_by_alias(
            normalized_model_name,
            normalized_alias,
        )

        if str(alias_version.version) != version:
            message = (
                f"O alias {normalized_alias!r} não aponta para "
                f"a versão esperada {version}."
            )
            raise RuntimeError(message)

        return RegistryResult(
            registered_model_name=normalized_model_name,
            version=version,
            run_id=normalized_run_id,
            source_model_uri=normalized_model_uri,
            registry_model_uri=(f"models:/{normalized_model_name}/{version}"),
            alias=normalized_alias,
            alias_model_uri=(f"models:/{normalized_model_name}@{normalized_alias}"),
            status=status,
            tracking_uri=self.tracking_uri,
            created_new_version=created_new_version,
        )

    def _find_version_by_run(
        self,
        model_name: str,
        run_id: str,
    ) -> ModelVersion | None:
        """Procura uma versão já vinculada ao run.

        Args:
            model_name: Nome do modelo registrado.
            run_id: Identificador do run.

        Returns:
            Versão encontrada ou ``None``.
        """
        versions = self.client.search_model_versions(
            filter_string=f"name='{model_name}'"
        )

        for version in versions:
            if version.run_id == run_id:
                return version

        return None

    @staticmethod
    def _require_text(
        value: str,
        field_name: str,
    ) -> str:
        """Valida um texto obrigatório."""
        normalized = value.strip()

        if not normalized:
            message = f"{field_name} não pode ser vazio."
            raise ValueError(message)

        return normalized

    @classmethod
    def _validate_name(
        cls,
        value: str,
        field_name: str,
    ) -> str:
        """Valida nomes usados pelo Model Registry."""
        normalized = cls._require_text(value, field_name)

        if not _NAME_PATTERN.fullmatch(normalized):
            message = (
                f"{field_name} deve conter apenas letras, números, "
                "ponto, hífen ou sublinhado."
            )
            raise ValueError(message)

        return normalized
