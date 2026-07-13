"""Serviço de promoção de modelos no MLflow Model Registry."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any, Literal

import mlflow
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

PromotionTarget = Literal["staging", "production"]


@dataclass(frozen=True)
class PromotionResult:
    """Resultado de uma promoção no Model Registry."""

    registered_model_name: str
    version: str
    run_id: str
    target: str
    alias: str
    alias_model_uri: str
    status: str
    legacy_stage: str
    tracking_uri: str
    promoted_at_utc: str
    previous_alias_version: str | None
    already_promoted: bool
    approved_by: str
    approval_notes: str

    def to_dict(self) -> dict[str, Any]:
        """Converte o resultado para um dicionário JSON."""
        return asdict(self)


class ModelPromotionService:
    """Promove versões usando aliases, tags e lifecycle stages."""

    def __init__(
        self,
        tracking_uri: str,
        client: MlflowClient | None = None,
    ) -> None:
        """Inicializa o serviço.

        Args:
            tracking_uri: URI do servidor MLflow.
            client: Cliente opcional para injeção em testes.
        """
        normalized_uri = tracking_uri.strip().rstrip("/")

        if not normalized_uri:
            msg = "tracking_uri não pode ser vazio."
            raise ValueError(msg)

        self.tracking_uri = normalized_uri

        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_registry_uri(self.tracking_uri)

        self.client = client or MlflowClient()

    def promote(
        self,
        *,
        model_name: str,
        version: str,
        run_id: str,
        target: PromotionTarget,
        staging_alias: str,
        production_alias: str,
        approved_by: str,
        approval_notes: str,
        require_staging_before_production: bool = True,
        sync_legacy_stage: bool = True,
    ) -> PromotionResult:
        """Promove uma versão para staging ou production.

        Args:
            model_name: Nome do modelo registrado.
            version: Versão que será promovida.
            run_id: Run de treinamento esperado.
            target: Destino da promoção.
            staging_alias: Alias de staging.
            production_alias: Alias de produção.
            approved_by: Responsável pela aprovação.
            approval_notes: Justificativa da promoção.
            require_staging_before_production: Exige passagem por staging.
            sync_legacy_stage: Sincroniza lifecycle stage legado.

        Returns:
            Resultado da promoção.

        Raises:
            ValueError: Se algum parâmetro for inválido.
            RuntimeError: Se a versão não puder ser promovida.
        """
        normalized_name = self._require_text(
            model_name,
            "model_name",
        )
        normalized_version = self._require_text(
            version,
            "version",
        )
        normalized_run_id = self._require_text(
            run_id,
            "run_id",
        )
        normalized_approved_by = self._require_text(
            approved_by,
            "approved_by",
        )
        normalized_approval_notes = self._require_text(
            approval_notes,
            "approval_notes",
        )

        if target not in {"staging", "production"}:
            msg = "target deve ser 'staging' ou 'production'."
            raise ValueError(msg)

        target_alias = (
            self._require_text(
                staging_alias,
                "staging_alias",
            )
            if target == "staging"
            else self._require_text(
                production_alias,
                "production_alias",
            )
        )

        model_version = self.client.get_model_version(
            normalized_name,
            normalized_version,
        )

        status = str(model_version.status)

        if status != "READY":
            msg = (
                f"A versão {normalized_version} do modelo "
                f"{normalized_name!r} não está pronta. "
                f"Status atual: {status}"
            )
            raise RuntimeError(msg)

        registered_run_id = str(model_version.run_id)

        if registered_run_id != normalized_run_id:
            msg = (
                "O run_id informado não corresponde à versão "
                "registrada. "
                f"Esperado: {registered_run_id}; "
                f"recebido: {normalized_run_id}."
            )
            raise RuntimeError(msg)

        if target == "production" and require_staging_before_production:
            self._validate_staging_alias(
                model_name=normalized_name,
                version=normalized_version,
                staging_alias=staging_alias,
            )

        previous_alias_version = self._get_alias_version(
            model_name=normalized_name,
            alias=target_alias,
        )

        already_promoted = previous_alias_version == normalized_version

        self.client.set_registered_model_alias(
            normalized_name,
            target_alias,
            normalized_version,
        )

        promoted_at = datetime.now(UTC).isoformat()

        tags = {
            "lifecycle_stage": target,
            "promoted_alias": target_alias,
            f"{target}_promoted_at_utc": promoted_at,
            f"{target}_approved_by": normalized_approved_by,
            f"{target}_approval_notes": normalized_approval_notes,
        }

        for key, value in tags.items():
            self.client.set_model_version_tag(
                normalized_name,
                normalized_version,
                key,
                value,
            )

        legacy_stage = str(model_version.current_stage)

        if sync_legacy_stage:
            transitioned_version = self.client.transition_model_version_stage(
                name=normalized_name,
                version=normalized_version,
                stage=target.capitalize(),
                archive_existing_versions=True,
            )
            legacy_stage = str(transitioned_version.current_stage)

        alias_version = self.client.get_model_version_by_alias(
            normalized_name,
            target_alias,
        )

        if str(alias_version.version) != normalized_version:
            msg = (
                f"O alias {target_alias!r} não aponta para "
                f"a versão esperada {normalized_version}."
            )
            raise RuntimeError(msg)

        return PromotionResult(
            registered_model_name=normalized_name,
            version=normalized_version,
            run_id=normalized_run_id,
            target=target,
            alias=target_alias,
            alias_model_uri=(f"models:/{normalized_name}@{target_alias}"),
            status=status,
            legacy_stage=legacy_stage,
            tracking_uri=self.tracking_uri,
            promoted_at_utc=promoted_at,
            previous_alias_version=previous_alias_version,
            already_promoted=already_promoted,
            approved_by=normalized_approved_by,
            approval_notes=normalized_approval_notes,
        )

    def _validate_staging_alias(
        self,
        *,
        model_name: str,
        version: str,
        staging_alias: str,
    ) -> None:
        """Confirma que a versão passou por staging."""
        normalized_alias = self._require_text(
            staging_alias,
            "staging_alias",
        )

        try:
            staging_version = self.client.get_model_version_by_alias(
                model_name,
                normalized_alias,
            )
        except MlflowException as error:
            msg = (
                "A promoção para production exige que a versão "
                f"esteja primeiro no alias {normalized_alias!r}."
            )
            raise RuntimeError(msg) from error

        if str(staging_version.version) != version:
            msg = (
                f"O alias {normalized_alias!r} aponta para a "
                f"versão {staging_version.version}, e não para "
                f"a versão solicitada {version}."
            )
            raise RuntimeError(msg)

    def _get_alias_version(
        self,
        *,
        model_name: str,
        alias: str,
    ) -> str | None:
        """Retorna a versão atualmente associada ao alias."""
        try:
            model_version = self.client.get_model_version_by_alias(
                model_name,
                alias,
            )
        except MlflowException:
            return None

        return str(model_version.version)

    @staticmethod
    def _require_text(
        value: str,
        field_name: str,
    ) -> str:
        """Valida uma string obrigatória."""
        normalized = value.strip()

        if not normalized:
            msg = f"{field_name} não pode ser vazio."
            raise ValueError(msg)

        return normalized
