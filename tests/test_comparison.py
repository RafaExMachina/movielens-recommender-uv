"""Testes da comparação entre modelos."""

import json
from pathlib import Path

import pytest

from recommender.evaluation.comparison import (
    build_model_metrics,
    compare_from_files,
    compare_model_metrics,
    compare_models,
    metric_direction,
    save_comparison,
)


@pytest.fixture
def neural_metrics() -> dict[str, float]:
    """Retorna métricas de exemplo para o modelo neural."""
    return {
        "rmse": 0.24,
        "mae": 0.19,
        "mse": 0.0576,
        "r2": 0.40,
        "median_absolute_error": 0.15,
    }


@pytest.fixture
def baseline_metrics() -> dict[str, dict[str, float]]:
    """Retorna métricas de exemplo para os baselines."""
    return {
        "dummy_mean": {
            "rmse": 0.30,
            "mae": 0.24,
            "mse": 0.09,
            "r2": 0.00,
            "median_absolute_error": 0.25,
        },
        "ridge_one_hot": {
            "rmse": 0.26,
            "mae": 0.20,
            "mse": 0.0676,
            "r2": 0.30,
            "median_absolute_error": 0.18,
        },
    }


def test_metric_direction() -> None:
    """Verifica a direção das métricas."""
    assert metric_direction("rmse") == "minimize"
    assert metric_direction("mae") == "minimize"
    assert metric_direction("r2") == "maximize"


def test_metric_direction_rejects_unknown_metric() -> None:
    """Uma métrica desconhecida deve ser rejeitada."""
    with pytest.raises(ValueError, match="não suportada"):
        metric_direction("unknown_metric")


def test_build_model_metrics(
    neural_metrics: dict[str, float],
    baseline_metrics: dict[str, dict[str, float]],
) -> None:
    """As métricas neurais e dos baselines devem ser combinadas."""
    models = build_model_metrics(
        neural_metrics=neural_metrics,
        baseline_metrics=baseline_metrics,
    )

    assert set(models) == {
        "mlp",
        "dummy_mean",
        "ridge_one_hot",
    }


def test_compare_models_selects_lowest_rmse(
    neural_metrics: dict[str, float],
    baseline_metrics: dict[str, dict[str, float]],
) -> None:
    """O modelo com menor RMSE deve ser selecionado."""
    result = compare_models(
        neural_metrics=neural_metrics,
        baseline_metrics=baseline_metrics,
        selection_metric="rmse",
    )

    assert result["selected_model"] == "mlp"
    assert result["selected_value"] == pytest.approx(0.24)
    assert result["selection_direction"] == "minimize"


def test_compare_models_selects_highest_r2(
    neural_metrics: dict[str, float],
    baseline_metrics: dict[str, dict[str, float]],
) -> None:
    """O modelo com maior R² deve ser selecionado."""
    result = compare_models(
        neural_metrics=neural_metrics,
        baseline_metrics=baseline_metrics,
        selection_metric="r2",
    )

    assert result["selected_model"] == "mlp"
    assert result["selected_value"] == pytest.approx(0.40)
    assert result["selection_direction"] == "maximize"


def test_compare_model_metrics_creates_ranking() -> None:
    """A comparação deve produzir um ranking ordenado."""
    metrics_by_model = {
        "model_a": {"rmse": 0.30},
        "model_b": {"rmse": 0.20},
        "model_c": {"rmse": 0.25},
    }

    result = compare_model_metrics(
        metrics_by_model,
        selection_metric="rmse",
    )

    ranking = result["ranking"]

    assert isinstance(ranking, list)
    assert ranking[0]["model"] == "model_b"
    assert ranking[1]["model"] == "model_c"
    assert ranking[2]["model"] == "model_a"


def test_compare_rejects_missing_selection_metric() -> None:
    """Todos os modelos devem possuir a métrica de seleção."""
    metrics_by_model = {
        "model_a": {"rmse": 0.20},
        "model_b": {"mae": 0.15},
    }

    with pytest.raises(ValueError, match="não possuem"):
        compare_model_metrics(
            metrics_by_model,
            selection_metric="rmse",
        )


def test_save_comparison(tmp_path: Path) -> None:
    """O resultado deve ser salvo em JSON."""
    comparison = {
        "selected_model": "mlp",
        "selection_metric": "rmse",
    }

    output_path = tmp_path / "metrics" / "comparison.json"

    save_comparison(comparison, output_path)

    assert output_path.is_file()

    saved_content = json.loads(output_path.read_text(encoding="utf-8"))

    assert saved_content["selected_model"] == "mlp"


def test_compare_from_files(
    tmp_path: Path,
    neural_metrics: dict[str, float],
    baseline_metrics: dict[str, dict[str, float]],
) -> None:
    """A comparação deve funcionar a partir de arquivos JSON."""
    neural_path = tmp_path / "evaluation_metrics.json"
    baseline_path = tmp_path / "baseline_metrics.json"
    output_path = tmp_path / "model_comparison.json"

    neural_path.write_text(
        json.dumps(neural_metrics),
        encoding="utf-8",
    )
    baseline_path.write_text(
        json.dumps(baseline_metrics),
        encoding="utf-8",
    )

    result = compare_from_files(
        neural_metrics_path=neural_path,
        baseline_metrics_path=baseline_path,
        output_path=output_path,
    )

    assert result["selected_model"] == "mlp"
    assert output_path.is_file()
