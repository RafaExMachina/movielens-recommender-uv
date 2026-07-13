"""Promove um modelo registrado para staging ou production."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, cast

from recommender.repositories.artifact_repository import (
    ArtifactRepository,
)
from recommender.tracking.model_promotion import (
    ModelPromotionService,
    PromotionTarget,
)
from recommender.utils.config import load_yaml
from recommender.utils.settings import get_settings

ROOT_DIR = Path(__file__).resolve().parents[1]
PARAMS_PATH = ROOT_DIR / "params.yaml"


def parse_args() -> argparse.Namespace:
    """Lê os argumentos da linha de comando.

    Returns:
        Argumentos informados pelo usuário.
    """
    parser = argparse.ArgumentParser(
        description=("Promove um modelo registrado para staging ou production.")
    )
    parser.add_argument(
        "target",
        choices=("staging", "production"),
        help="Destino da promoção.",
    )

    return parser.parse_args()


def get_section(
    params: dict[str, Any],
    section_name: str,
) -> dict[str, Any]:
    """Obtém uma seção obrigatória do params.yaml.

    Args:
        params: Parâmetros completos do projeto.
        section_name: Nome da seção solicitada.

    Returns:
        Seção validada como dicionário.

    Raises:
        ValueError: Se a seção estiver ausente ou for inválida.
    """
    section = params.get(section_name)

    if not isinstance(section, dict):
        msg = f"A seção {section_name!r} deve ser um objeto no arquivo params.yaml."
        raise ValueError(msg)

    return {str(key): value for key, value in section.items()}


def require_string(
    value: object,
    parameter_name: str,
) -> str:
    """Valida uma string obrigatória.

    Args:
        value: Valor que será validado.
        parameter_name: Nome do parâmetro.

    Returns:
        String sem espaços nas extremidades.

    Raises:
        ValueError: Se o valor não for uma string válida.
    """
    if not isinstance(value, str) or not value.strip():
        msg = f"O parâmetro {parameter_name!r} deve ser uma string."
        raise ValueError(msg)

    return value.strip()


def require_float(
    value: object,
    parameter_name: str,
) -> float:
    """Valida um número obrigatório.

    Args:
        value: Valor que será validado.
        parameter_name: Nome do parâmetro.

    Returns:
        Valor convertido para float.

    Raises:
        ValueError: Se o valor não for numérico.
    """
    if isinstance(value, bool) or not isinstance(value, int | float):
        msg = f"O parâmetro {parameter_name!r} deve ser numérico."
        raise ValueError(msg)

    return float(value)


def require_bool(
    value: object,
    parameter_name: str,
) -> bool:
    """Valida um booleano obrigatório.

    Args:
        value: Valor que será validado.
        parameter_name: Nome do parâmetro.

    Returns:
        Valor booleano validado.

    Raises:
        ValueError: Se o valor não for booleano.
    """
    if not isinstance(value, bool):
        msg = f"O parâmetro {parameter_name!r} deve ser booleano."
        raise ValueError(msg)

    return value


def resolve_project_path(
    raw_path: object,
    parameter_name: str,
) -> Path:
    """Resolve um caminho em relação à raiz do projeto.

    Args:
        raw_path: Caminho configurado.
        parameter_name: Nome do parâmetro.

    Returns:
        Caminho absoluto resolvido.
    """
    value = require_string(
        raw_path,
        parameter_name,
    )
    path = Path(value)

    if path.is_absolute():
        return path

    return ROOT_DIR / path


def read_json(
    path: Path,
) -> dict[str, Any]:
    """Carrega e valida um arquivo JSON.

    Args:
        path: Caminho do arquivo.

    Returns:
        Conteúdo JSON validado como dicionário.

    Raises:
        FileNotFoundError: Se o arquivo não existir.
        ValueError: Se o conteúdo não for um objeto JSON.
    """
    if not path.is_file():
        msg = f"Arquivo JSON não encontrado: {path}"
        raise FileNotFoundError(msg)

    payload: Any = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(payload, dict):
        msg = f"O arquivo {path} deve conter um objeto JSON."
        raise ValueError(msg)

    return {str(key): value for key, value in payload.items()}


def output_path_for_target(
    target: PromotionTarget,
    artifacts: dict[str, Any],
) -> Path:
    """Obtém o arquivo de saída da promoção.

    Args:
        target: Destino da promoção.
        artifacts: Configurações dos artefatos.

    Returns:
        Caminho do arquivo de saída.
    """
    key = (
        "staging_promotion_path" if target == "staging" else "production_promotion_path"
    )

    return resolve_project_path(
        artifacts.get(key),
        f"artifacts.{key}",
    )


def approval_for_target(
    target: PromotionTarget,
    promotion: dict[str, Any],
) -> tuple[str, str]:
    """Obtém responsável e justificativa da promoção.

    Args:
        target: Destino da promoção.
        promotion: Configurações da política de promoção.

    Returns:
        Responsável e justificativa correspondentes ao destino.
    """
    approved_by_key = f"{target}_approved_by"
    approval_notes_key = f"{target}_approval_notes"

    approved_by = require_string(
        promotion.get(approved_by_key),
        f"promotion.{approved_by_key}",
    )
    approval_notes = require_string(
        promotion.get(approval_notes_key),
        f"promotion.{approval_notes_key}",
    )

    return approved_by, approval_notes


def normalize_uri(
    value: str,
) -> str:
    """Normaliza um URI removendo a barra final.

    Args:
        value: URI original.

    Returns:
        URI normalizado.
    """
    return value.rstrip("/")


def main() -> None:
    """Executa a promoção solicitada."""
    args = parse_args()
    target = cast(PromotionTarget, args.target)

    params = load_yaml(PARAMS_PATH)

    model_config = get_section(
        params,
        "model",
    )
    artifacts = get_section(
        params,
        "artifacts",
    )
    quality_gate = get_section(
        params,
        "quality_gate",
    )
    promotion = get_section(
        params,
        "promotion",
    )

    registered_model_path = resolve_project_path(
        artifacts.get("registered_model_path"),
        "artifacts.registered_model_path",
    )
    evaluation_metrics_path = resolve_project_path(
        artifacts.get("evaluation_metrics_path"),
        "artifacts.evaluation_metrics_path",
    )
    comparison_path = resolve_project_path(
        artifacts.get("model_comparison_path"),
        "artifacts.model_comparison_path",
    )
    output_path = output_path_for_target(
        target,
        artifacts,
    )

    registered_model = read_json(registered_model_path)
    evaluation_metrics = read_json(evaluation_metrics_path)
    comparison = read_json(comparison_path)

    model_name = require_string(
        registered_model.get("registered_model_name"),
        "registered_model.registered_model_name",
    )
    version = require_string(
        registered_model.get("version"),
        "registered_model.version",
    )
    run_id = require_string(
        registered_model.get("run_id"),
        "registered_model.run_id",
    )
    registered_status = require_string(
        registered_model.get("status"),
        "registered_model.status",
    )
    registered_tracking_uri = require_string(
        registered_model.get("tracking_uri"),
        "registered_model.tracking_uri",
    )

    if registered_status != "READY":
        msg = f"O modelo registrado não está pronto. Status atual: {registered_status}"
        raise RuntimeError(msg)

    tracking_uri = str(get_settings().mlflow_tracking_uri)

    if normalize_uri(registered_tracking_uri) != normalize_uri(tracking_uri):
        msg = (
            "O tracking URI do modelo registrado é "
            "diferente do URI configurado no ambiente.\n"
            f"Modelo registrado: {registered_tracking_uri}\n"
            f"Ambiente: {tracking_uri}"
        )
        raise ValueError(msg)

    rmse = require_float(
        evaluation_metrics.get("rmse"),
        "evaluation_metrics.rmse",
    )
    maximum_rmse = require_float(
        quality_gate.get("maximum_rmse"),
        "quality_gate.maximum_rmse",
    )

    quality_gate_status = "passed" if rmse <= maximum_rmse else "failed"

    require_quality_gate = require_bool(
        promotion.get("require_quality_gate"),
        "promotion.require_quality_gate",
    )

    if require_quality_gate and quality_gate_status != "passed":
        msg = f"Quality gate reprovado. RMSE={rmse:.6f}; limite={maximum_rmse:.6f}."
        raise RuntimeError(msg)

    source_model_name = require_string(
        model_config.get("name"),
        "model.name",
    )
    selected_model = require_string(
        comparison.get("selected_model"),
        "model_comparison.selected_model",
    )

    require_comparison_winner = require_bool(
        promotion.get("require_comparison_winner"),
        "promotion.require_comparison_winner",
    )

    if require_comparison_winner and selected_model != source_model_name:
        msg = (
            "A política exige que o modelo seja "
            "o vencedor da comparação. "
            f"Vencedor atual: {selected_model}."
        )
        raise RuntimeError(msg)

    staging_alias = require_string(
        promotion.get("staging_alias"),
        "promotion.staging_alias",
    )
    production_alias = require_string(
        promotion.get("production_alias"),
        "promotion.production_alias",
    )
    require_staging = require_bool(
        promotion.get("require_staging_before_production"),
        ("promotion.require_staging_before_production"),
    )
    sync_legacy_stage = require_bool(
        promotion.get("sync_legacy_stage"),
        "promotion.sync_legacy_stage",
    )

    approved_by, approval_notes = approval_for_target(
        target,
        promotion,
    )

    result = ModelPromotionService(tracking_uri=tracking_uri).promote(
        model_name=model_name,
        version=version,
        run_id=run_id,
        target=target,
        staging_alias=staging_alias,
        production_alias=production_alias,
        approved_by=approved_by,
        approval_notes=approval_notes,
        require_staging_before_production=(require_staging),
        sync_legacy_stage=sync_legacy_stage,
    )

    payload = result.to_dict()

    payload["test_rmse"] = rmse
    payload["maximum_rmse"] = maximum_rmse
    payload["quality_gate_status"] = quality_gate_status
    payload["comparison_winner"] = selected_model
    payload["is_comparison_winner"] = selected_model == source_model_name

    ArtifactRepository().save_json(
        payload,
        output_path,
    )

    action = (
        "Alias já associado à versão"
        if result.already_promoted
        else "Promoção realizada"
    )

    print("Promoção concluída com sucesso.")
    print(f"Ação: {action}")
    print(f"Modelo: {result.registered_model_name}")
    print(f"Versão: {result.version}")
    print(f"Destino: {result.target}")
    print(f"Alias: {result.alias}")
    print(f"URI: {result.alias_model_uri}")
    print(f"Estágio legado: {result.legacy_stage}")
    print(f"RMSE: {rmse:.6f}")
    print(f"Limite: {maximum_rmse:.6f}")
    print(f"Quality gate: {quality_gate_status}")
    print(f"Aprovado por: {result.approved_by}")
    print(f"Vencedor da comparação: {selected_model}")
    print(f"Metadados salvos em: {output_path}")


if __name__ == "__main__":
    main()
