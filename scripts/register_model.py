"""Registra o modelo neural treinado no MLflow Model Registry."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from recommender.repositories.artifact_repository import ArtifactRepository
from recommender.tracking.model_registry import ModelRegistryService
from recommender.utils.config import load_yaml
from recommender.utils.settings import get_settings

ROOT_DIR = Path(__file__).resolve().parents[1]
PARAMS_PATH = ROOT_DIR / "params.yaml"


def get_section(
    params: dict[str, Any],
    section_name: str,
) -> dict[str, Any]:
    """Obtém uma seção obrigatória do arquivo de parâmetros.

    Args:
        params: Parâmetros completos do projeto.
        section_name: Nome da seção solicitada.

    Returns:
        Seção validada.

    Raises:
        ValueError: Se a seção estiver ausente ou não for um objeto.
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
        parameter_name: Nome do parâmetro para mensagens de erro.

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
    """Valida um valor numérico obrigatório.

    Args:
        value: Valor que será validado.
        parameter_name: Nome do parâmetro para mensagens de erro.

    Returns:
        Valor convertido para float.

    Raises:
        ValueError: Se o valor não for numérico.
    """
    if isinstance(value, bool) or not isinstance(value, int | float):
        msg = f"O parâmetro {parameter_name!r} deve ser numérico."
        raise ValueError(msg)

    return float(value)


def require_positive_int(
    value: object,
    parameter_name: str,
) -> int:
    """Valida um inteiro positivo.

    Args:
        value: Valor que será validado.
        parameter_name: Nome do parâmetro para mensagens de erro.

    Returns:
        Inteiro positivo validado.

    Raises:
        ValueError: Se o valor não for um inteiro positivo.
    """
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        msg = f"O parâmetro {parameter_name!r} deve ser um inteiro positivo."
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
    value = require_string(raw_path, parameter_name)
    path = Path(value)

    if path.is_absolute():
        return path

    return ROOT_DIR / path


def read_json(path: Path) -> dict[str, Any]:
    """Carrega e valida um arquivo JSON.

    Args:
        path: Caminho do arquivo.

    Returns:
        Objeto JSON convertido para dicionário.

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


def normalize_uri(value: str) -> str:
    """Remove a barra final de um URI.

    Args:
        value: URI original.

    Returns:
        URI normalizado.
    """
    return value.rstrip("/")


def main() -> None:
    """Registra o modelo neural no MLflow Model Registry."""
    params = load_yaml(PARAMS_PATH)

    artifacts = get_section(params, "artifacts")
    registry = get_section(params, "registry")
    quality_gate = get_section(params, "quality_gate")

    training_run_path = resolve_project_path(
        artifacts.get("training_run_path"),
        "artifacts.training_run_path",
    )
    evaluation_metrics_path = resolve_project_path(
        artifacts.get("evaluation_metrics_path"),
        "artifacts.evaluation_metrics_path",
    )
    comparison_path = resolve_project_path(
        artifacts.get("model_comparison_path"),
        "artifacts.model_comparison_path",
    )
    output_path = resolve_project_path(
        artifacts.get("registered_model_path"),
        "artifacts.registered_model_path",
    )

    training_run = read_json(training_run_path)
    evaluation_metrics = read_json(evaluation_metrics_path)
    comparison = read_json(comparison_path)

    run_id = require_string(
        training_run.get("run_id"),
        "training_run.run_id",
    )
    model_uri = require_string(
        training_run.get("model_uri"),
        "training_run.model_uri",
    )
    source_model_name = require_string(
        training_run.get("model_name"),
        "training_run.model_name",
    )
    model_flavor = require_string(
        training_run.get("model_flavor"),
        "training_run.model_flavor",
    )
    saved_tracking_uri = require_string(
        training_run.get("tracking_uri"),
        "training_run.tracking_uri",
    )

    settings = get_settings()
    configured_tracking_uri = str(settings.mlflow_tracking_uri)

    if normalize_uri(saved_tracking_uri) != normalize_uri(configured_tracking_uri):
        msg = (
            "O tracking URI salvo no treinamento é diferente "
            "do URI configurado no ambiente.\n"
            f"Treinamento: {saved_tracking_uri}\n"
            f"Ambiente: {configured_tracking_uri}"
        )
        raise ValueError(msg)

    registered_model_name = require_string(
        registry.get("model_name"),
        "registry.model_name",
    )
    description = require_string(
        registry.get("description"),
        "registry.description",
    )
    initial_alias = require_string(
        registry.get("initial_alias"),
        "registry.initial_alias",
    )
    await_seconds = require_positive_int(
        registry.get("await_registration_seconds", 300),
        "registry.await_registration_seconds",
    )

    rmse = require_float(
        evaluation_metrics.get("rmse"),
        "evaluation_metrics.rmse",
    )
    selected_model = require_string(
        comparison.get("selected_model"),
        "model_comparison.selected_model",
    )
    selection_metric = require_string(
        comparison.get("selection_metric"),
        "model_comparison.selection_metric",
    )
    maximum_rmse = require_float(
        quality_gate.get("maximum_rmse"),
        "quality_gate.maximum_rmse",
    )

    is_comparison_winner = selected_model == source_model_name

    quality_gate_status = "passed" if rmse <= maximum_rmse else "failed"

    service = ModelRegistryService(tracking_uri=configured_tracking_uri)

    result = service.register(
        model_name=registered_model_name,
        model_uri=model_uri,
        run_id=run_id,
        alias=initial_alias,
        description=description,
        await_registration_seconds=await_seconds,
        registered_model_tags={
            "task": "rating_prediction",
            "dataset": "MovieLens 100K",
            "framework": model_flavor,
            "model_type": "neural",
        },
        model_version_tags={
            "validation_status": "registered",
            "quality_gate_status": quality_gate_status,
            "selection_metric": selection_metric,
            "test_rmse": f"{rmse:.12f}",
            "maximum_rmse": f"{maximum_rmse:.12f}",
            "comparison_winner": selected_model,
            "is_comparison_winner": str(is_comparison_winner).lower(),
            "source_model_name": source_model_name,
        },
    )

    ArtifactRepository().save_json(
        result.to_dict(),
        output_path,
    )

    action = (
        "Nova versão criada"
        if result.created_new_version
        else "Versão existente reutilizada"
    )

    print("Registro concluído com sucesso.")
    print(f"Ação: {action}")
    print(f"Modelo: {result.registered_model_name}")
    print(f"Versão: {result.version}")
    print(f"Status: {result.status}")
    print(f"Run ID: {result.run_id}")
    print(f"Alias: {result.alias}")
    print(f"URI da versão: {result.registry_model_uri}")
    print(f"URI do alias: {result.alias_model_uri}")
    print(f"Vencedor da comparação: {selected_model}")
    print(f"Quality gate: {quality_gate_status}")
    print(f"Metadados salvos em: {output_path}")


if __name__ == "__main__":
    main()
