"""Comparação entre o modelo neural e os baselines."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal, cast

MetricDirection = Literal["minimize", "maximize"]
Metrics = dict[str, float]
MetricsByModel = dict[str, Metrics]

_METRIC_DIRECTIONS: dict[str, MetricDirection] = {
    "rmse": "minimize",
    "mae": "minimize",
    "mse": "minimize",
    "median_absolute_error": "minimize",
    "rmse_rating_scale": "minimize",
    "mae_rating_scale": "minimize",
    "r2": "maximize",
}


def metric_direction(metric_name: str) -> MetricDirection:
    """Retorna a direção utilizada para ordenar uma métrica.

    Métricas de erro, como RMSE e MAE, devem ser minimizadas. Já o
    coeficiente R² deve ser maximizado.

    Args:
        metric_name: Nome da métrica.

    Returns:
        Direção da métrica: ``minimize`` ou ``maximize``.

    Raises:
        ValueError: Se a métrica não possuir uma direção conhecida.
    """
    try:
        return _METRIC_DIRECTIONS[metric_name]
    except KeyError as error:
        supported_metrics = ", ".join(sorted(_METRIC_DIRECTIONS))
        message = (
            f"Métrica de seleção não suportada: {metric_name!r}. "
            f"Métricas disponíveis: {supported_metrics}."
        )
        raise ValueError(message) from error


def _normalize_metrics(
    model_name: str,
    metrics: Mapping[str, float],
) -> Metrics:
    """Valida e normaliza as métricas de um modelo.

    Args:
        model_name: Nome do modelo.
        metrics: Métricas calculadas para o modelo.

    Returns:
        Métricas convertidas para valores ``float``.

    Raises:
        ValueError: Se o nome estiver vazio, não houver métricas ou algum
            valor não for finito.
        TypeError: Se uma métrica não for numérica.
    """
    if not model_name.strip():
        message = "O nome do modelo não pode estar vazio."
        raise ValueError(message)

    if not metrics:
        message = f"O modelo {model_name!r} não possui métricas."
        raise ValueError(message)

    normalized_metrics: Metrics = {}

    for metric_name, raw_value in metrics.items():
        if isinstance(raw_value, bool) or not isinstance(
            raw_value,
            int | float,
        ):
            message = (
                f"A métrica {metric_name!r} do modelo {model_name!r} "
                "deve possuir um valor numérico."
            )
            raise TypeError(message)

        value = float(raw_value)

        if not math.isfinite(value):
            message = (
                f"A métrica {metric_name!r} do modelo {model_name!r} "
                "contém um valor NaN ou infinito."
            )
            raise ValueError(message)

        normalized_metrics[metric_name] = value

    return normalized_metrics


def build_model_metrics(
    neural_metrics: Mapping[str, float],
    baseline_metrics: Mapping[str, Mapping[str, float]],
    neural_model_name: str = "mlp",
) -> MetricsByModel:
    """Combina as métricas do modelo neural com as dos baselines.

    Args:
        neural_metrics: Métricas do modelo neural.
        baseline_metrics: Métricas organizadas pelo nome de cada baseline.
        neural_model_name: Nome usado para identificar o modelo neural.

    Returns:
        Dicionário com as métricas de todos os modelos.

    Raises:
        ValueError: Se o nome do modelo neural também for utilizado por
            algum baseline.
    """
    if neural_model_name in baseline_metrics:
        message = (
            f"O nome {neural_model_name!r} está sendo utilizado tanto pelo "
            "modelo neural quanto por um baseline."
        )
        raise ValueError(message)

    models: MetricsByModel = {
        neural_model_name: _normalize_metrics(
            neural_model_name,
            neural_metrics,
        )
    }

    for baseline_name, metrics in baseline_metrics.items():
        models[baseline_name] = _normalize_metrics(
            baseline_name,
            metrics,
        )

    return models


def compare_model_metrics(
    metrics_by_model: Mapping[str, Mapping[str, float]],
    selection_metric: str = "rmse",
) -> dict[str, object]:
    """Compara modelos e cria um ranking com base em uma métrica.

    Args:
        metrics_by_model: Métricas organizadas pelo nome do modelo.
        selection_metric: Métrica utilizada para escolher o melhor modelo.

    Returns:
        Resultado da comparação contendo o modelo selecionado, ranking e
        métricas completas.

    Raises:
        ValueError: Se nenhum modelo for informado ou se algum modelo não
            possuir a métrica de seleção.
    """
    if not metrics_by_model:
        message = "É necessário informar pelo menos um modelo."
        raise ValueError(message)

    direction = metric_direction(selection_metric)

    normalized_models: MetricsByModel = {
        model_name: _normalize_metrics(model_name, metrics)
        for model_name, metrics in metrics_by_model.items()
    }

    models_without_metric = [
        model_name
        for model_name, metrics in normalized_models.items()
        if selection_metric not in metrics
    ]

    if models_without_metric:
        model_names = ", ".join(sorted(models_without_metric))
        message = (
            f"Os seguintes modelos não possuem a métrica "
            f"{selection_metric!r}: {model_names}."
        )
        raise ValueError(message)

    reverse_order = direction == "maximize"

    ordered_models = sorted(
        normalized_models,
        key=lambda model_name: normalized_models[model_name][selection_metric],
        reverse=reverse_order,
    )

    selected_model = ordered_models[0]
    selected_value = normalized_models[selected_model][selection_metric]

    ranking: list[dict[str, object]] = [
        {
            "position": position,
            "model": model_name,
            "metric": selection_metric,
            "value": normalized_models[model_name][selection_metric],
        }
        for position, model_name in enumerate(ordered_models, start=1)
    ]

    return {
        "selected_model": selected_model,
        "selection_metric": selection_metric,
        "selection_direction": direction,
        "selected_value": selected_value,
        "ranking": ranking,
        "models": normalized_models,
    }


def compare_models(
    neural_metrics: Mapping[str, float],
    baseline_metrics: Mapping[str, Mapping[str, float]],
    neural_model_name: str = "mlp",
    selection_metric: str = "rmse",
) -> dict[str, object]:
    """Compara um modelo neural com um conjunto de baselines.

    Args:
        neural_metrics: Métricas do modelo neural.
        baseline_metrics: Métricas dos modelos baseline.
        neural_model_name: Nome utilizado para o modelo neural.
        selection_metric: Métrica utilizada na seleção.

    Returns:
        Resultado consolidado da comparação.
    """
    metrics_by_model = build_model_metrics(
        neural_metrics=neural_metrics,
        baseline_metrics=baseline_metrics,
        neural_model_name=neural_model_name,
    )

    return compare_model_metrics(
        metrics_by_model=metrics_by_model,
        selection_metric=selection_metric,
    )


def _read_json(path: Path) -> Any:
    """Lê um arquivo JSON.

    Args:
        path: Caminho do arquivo.

    Returns:
        Conteúdo desserializado do arquivo JSON.

    Raises:
        FileNotFoundError: Se o arquivo não existir.
        ValueError: Se o conteúdo não for um JSON válido.
    """
    if not path.is_file():
        message = f"Arquivo de métricas não encontrado: {path}"
        raise FileNotFoundError(message)

    try:
        with path.open(encoding="utf-8") as json_file:
            return json.load(json_file)
    except json.JSONDecodeError as error:
        message = f"O arquivo não contém um JSON válido: {path}"
        raise ValueError(message) from error


def save_comparison(
    comparison: Mapping[str, object],
    output_path: str | Path,
) -> None:
    """Salva o resultado da comparação em formato JSON.

    Args:
        comparison: Resultado retornado pela comparação.
        output_path: Caminho do arquivo JSON de saída.
    """
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    with destination.open("w", encoding="utf-8") as json_file:
        json.dump(
            comparison,
            json_file,
            ensure_ascii=False,
            indent=2,
        )
        json_file.write("\n")


def compare_from_files(
    neural_metrics_path: str | Path,
    baseline_metrics_path: str | Path,
    output_path: str | Path | None = None,
    neural_model_name: str = "mlp",
    selection_metric: str = "rmse",
) -> dict[str, object]:
    """Compara modelos a partir dos arquivos JSON de métricas.

    O arquivo do modelo neural deve possuir uma estrutura plana:

    .. code-block:: json

        {
          "rmse": 0.247,
          "mae": 0.197,
          "mse": 0.061,
          "r2": 0.35
        }

    O arquivo dos baselines deve ser organizado pelo nome do modelo:

    .. code-block:: json

        {
          "dummy_mean": {
            "rmse": 0.28,
            "mae": 0.23
          },
          "ridge_one_hot": {
            "rmse": 0.25,
            "mae": 0.20
          }
        }

    Args:
        neural_metrics_path: Arquivo de métricas do modelo neural.
        baseline_metrics_path: Arquivo com métricas dos baselines.
        output_path: Caminho opcional para salvar a comparação.
        neural_model_name: Nome utilizado para identificar a rede neural.
        selection_metric: Métrica utilizada para selecionar o melhor modelo.

    Returns:
        Resultado consolidado da comparação.

    Raises:
        ValueError: Se algum arquivo não possuir um objeto JSON.
    """
    neural_payload = _read_json(Path(neural_metrics_path))
    baseline_payload = _read_json(Path(baseline_metrics_path))

    if not isinstance(neural_payload, dict):
        message = "O arquivo de métricas do modelo neural deve conter um objeto JSON."
        raise ValueError(message)

    if not isinstance(baseline_payload, dict):
        message = "O arquivo de métricas dos baselines deve conter um objeto JSON."
        raise ValueError(message)

    comparison = compare_models(
        neural_metrics=cast(Mapping[str, float], neural_payload),
        baseline_metrics=cast(
            Mapping[str, Mapping[str, float]],
            baseline_payload,
        ),
        neural_model_name=neural_model_name,
        selection_metric=selection_metric,
    )

    if output_path is not None:
        save_comparison(comparison, output_path)

    return comparison
