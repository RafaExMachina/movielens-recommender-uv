"""Compara o modelo neural com os baselines Scikit-Learn."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from recommender.evaluation.comparison import compare_from_files

ROOT_DIR = Path(__file__).resolve().parents[1]
PARAMS_PATH = ROOT_DIR / "params.yaml"


def load_params(params_path: Path = PARAMS_PATH) -> dict[str, Any]:
    """Carrega os parâmetros YAML do projeto.

    Args:
        params_path: Caminho do arquivo params.yaml.

    Returns:
        Parâmetros carregados como dicionário.

    Raises:
        FileNotFoundError: Se o arquivo não existir.
        ValueError: Se o conteúdo não for um objeto YAML.
    """
    if not params_path.is_file():
        message = f"Arquivo de parâmetros não encontrado: {params_path}"
        raise FileNotFoundError(message)

    with params_path.open(encoding="utf-8") as params_file:
        payload: Any = yaml.safe_load(params_file)

    if not isinstance(payload, dict):
        message = "O arquivo params.yaml deve conter um objeto YAML."
        raise ValueError(message)

    return {str(key): value for key, value in payload.items()}


def get_section(
    params: dict[str, Any],
    section_name: str,
) -> dict[str, Any]:
    """Obtém e valida uma seção do arquivo de parâmetros.

    Args:
        params: Parâmetros completos do projeto.
        section_name: Nome da seção desejada.

    Returns:
        Conteúdo da seção.

    Raises:
        ValueError: Se a seção não existir ou não for um objeto YAML.
    """
    section = params.get(section_name)

    if not isinstance(section, dict):
        message = f"A seção {section_name!r} deve ser um objeto no arquivo params.yaml."
        raise ValueError(message)

    return {str(key): value for key, value in section.items()}


def resolve_project_path(
    raw_path: object,
    parameter_name: str,
) -> Path:
    """Converte um caminho configurado em caminho absoluto.

    Args:
        raw_path: Caminho informado no params.yaml.
        parameter_name: Nome do parâmetro para mensagens de erro.

    Returns:
        Caminho absoluto dentro do projeto.

    Raises:
        ValueError: Se o caminho não for uma string válida.
    """
    if not isinstance(raw_path, str) or not raw_path.strip():
        message = f"O parâmetro {parameter_name!r} deve ser uma string."
        raise ValueError(message)

    path = Path(raw_path)

    if path.is_absolute():
        return path

    return ROOT_DIR / path


def get_selection_metric(
    quality_gate: dict[str, Any],
) -> str:
    """Obtém a métrica usada para selecionar o melhor modelo.

    Args:
        quality_gate: Configuração do quality gate.

    Returns:
        Nome da métrica de seleção.

    Raises:
        ValueError: Se a métrica não for uma string válida.
    """
    metric = quality_gate.get("selection_metric", "rmse")

    if not isinstance(metric, str) or not metric.strip():
        message = "quality_gate.selection_metric deve ser uma string válida."
        raise ValueError(message)

    return metric


def get_model_name(model_params: dict[str, Any]) -> str:
    """Obtém o nome configurado para o modelo neural.

    Args:
        model_params: Configuração do modelo.

    Returns:
        Nome do modelo neural.

    Raises:
        ValueError: Se o nome não for uma string válida.
    """
    model_name = model_params.get("name", "mlp")

    if not isinstance(model_name, str) or not model_name.strip():
        message = "model.name deve ser uma string válida."
        raise ValueError(message)

    return model_name


def print_ranking(comparison: dict[str, object]) -> None:
    """Exibe o ranking dos modelos no terminal.

    Args:
        comparison: Resultado consolidado da comparação.
    """
    ranking = comparison.get("ranking")

    if not isinstance(ranking, list):
        return

    print("\nRanking dos modelos:")

    for item in ranking:
        if not isinstance(item, dict):
            continue

        position = item.get("position")
        model_name = item.get("model")
        metric_name = item.get("metric")
        metric_value = item.get("value")

        if isinstance(metric_value, int | float):
            formatted_value = f"{float(metric_value):.6f}"
        else:
            formatted_value = str(metric_value)

        print(f"{position}. {model_name}: {metric_name}={formatted_value}")


def main() -> None:
    """Executa a comparação entre o modelo neural e os baselines."""
    params = load_params()

    artifacts = get_section(params, "artifacts")
    quality_gate = get_section(params, "quality_gate")
    model_params = get_section(params, "model")

    neural_metrics_path = resolve_project_path(
        artifacts.get("evaluation_metrics_path"),
        "artifacts.evaluation_metrics_path",
    )

    baseline_metrics_path = resolve_project_path(
        artifacts.get("baseline_metrics_path"),
        "artifacts.baseline_metrics_path",
    )

    output_path = resolve_project_path(
        artifacts.get("model_comparison_path"),
        "artifacts.model_comparison_path",
    )

    selection_metric = get_selection_metric(quality_gate)
    neural_model_name = get_model_name(model_params)

    comparison = compare_from_files(
        neural_metrics_path=neural_metrics_path,
        baseline_metrics_path=baseline_metrics_path,
        output_path=output_path,
        neural_model_name=neural_model_name,
        selection_metric=selection_metric,
    )

    selected_model = comparison["selected_model"]
    selected_value = float(comparison["selected_value"])
    selection_direction = comparison["selection_direction"]

    print("Comparação concluída com sucesso.")
    print(f"Métrica de seleção: {selection_metric}")
    print(f"Direção: {selection_direction}")
    print(f"Modelo selecionado: {selected_model}")
    print(f"Valor selecionado: {selected_value:.6f}")

    print_ranking(comparison)

    print(f"\nRelatório salvo em: {output_path}")


if __name__ == "__main__":
    main()
