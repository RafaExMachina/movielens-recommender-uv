"""Testes das métricas de avaliação."""

import numpy as np
import pytest

from recommender.evaluation.metrics import (
    add_rating_scale_metrics,
    compute_regression_metrics,
    mae,
    median_absolute_error,
    mse,
    r2,
    rmse,
)


def test_metrics_are_positive() -> None:
    """As métricas de erro devem ser não negativas."""
    y_true = np.array([1.0, 0.5, 0.0])
    y_pred = np.array([0.8, 0.5, 0.1])

    assert rmse(y_true, y_pred) >= 0.0
    assert mae(y_true, y_pred) >= 0.0
    assert mse(y_true, y_pred) >= 0.0
    assert median_absolute_error(y_true, y_pred) >= 0.0


def test_perfect_prediction() -> None:
    """Uma previsão perfeita deve produzir erros iguais a zero."""
    y_true = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    y_pred = y_true.copy()

    assert rmse(y_true, y_pred) == pytest.approx(0.0)
    assert mae(y_true, y_pred) == pytest.approx(0.0)
    assert mse(y_true, y_pred) == pytest.approx(0.0)
    assert median_absolute_error(y_true, y_pred) == pytest.approx(0.0)
    assert r2(y_true, y_pred) == pytest.approx(1.0)


def test_rmse_expected_value() -> None:
    """O RMSE deve corresponder ao valor calculado manualmente."""
    y_true = np.array([1.0, 0.5, 0.0])
    y_pred = np.array([0.8, 0.5, 0.1])

    expected_mse = ((0.2**2) + (0.0**2) + (0.1**2)) / 3
    expected_rmse = np.sqrt(expected_mse)

    assert rmse(y_true, y_pred) == pytest.approx(expected_rmse)


def test_mae_expected_value() -> None:
    """O MAE deve corresponder ao valor calculado manualmente."""
    y_true = np.array([1.0, 0.5, 0.0])
    y_pred = np.array([0.8, 0.5, 0.1])

    expected_mae = (0.2 + 0.0 + 0.1) / 3

    assert mae(y_true, y_pred) == pytest.approx(expected_mae)


def test_mse_expected_value() -> None:
    """O MSE deve corresponder ao valor calculado manualmente."""
    y_true = np.array([1.0, 0.5, 0.0])
    y_pred = np.array([0.8, 0.5, 0.1])

    expected_mse = ((0.2**2) + (0.0**2) + (0.1**2)) / 3

    assert mse(y_true, y_pred) == pytest.approx(expected_mse)


def test_median_absolute_error_expected_value() -> None:
    """A mediana dos erros absolutos deve ser calculada corretamente."""
    y_true = np.array([1.0, 0.5, 0.0])
    y_pred = np.array([0.8, 0.5, 0.1])

    # Erros absolutos: [0.2, 0.0, 0.1].
    assert median_absolute_error(y_true, y_pred) == pytest.approx(0.1)


def test_r2_perfect_prediction() -> None:
    """O R² deve ser igual a 1 para uma previsão perfeita."""
    y_true = np.array([0.0, 0.5, 1.0])
    y_pred = y_true.copy()

    assert r2(y_true, y_pred) == pytest.approx(1.0)


def test_r2_constant_prediction_equal_to_mean() -> None:
    """O R² deve ser zero ao prever sempre a média dos valores reais."""
    y_true = np.array([0.0, 0.5, 1.0])
    y_pred = np.array([0.5, 0.5, 0.5])

    assert r2(y_true, y_pred) == pytest.approx(0.0)


def test_compute_regression_metrics_returns_all_metrics() -> None:
    """O agregador deve retornar todas as métricas esperadas."""
    y_true = np.array([0.0, 0.5, 1.0])
    y_pred = np.array([0.1, 0.4, 0.9])

    metrics = compute_regression_metrics(y_true, y_pred)

    assert set(metrics) == {
        "rmse",
        "mae",
        "mse",
        "r2",
        "median_absolute_error",
    }

    assert all(isinstance(value, float) for value in metrics.values())


def test_rating_scale_metrics() -> None:
    """RMSE e MAE devem ser convertidos para a escala original de 1 a 5."""
    metrics = {
        "rmse": 0.25,
        "mae": 0.20,
        "mse": 0.0625,
        "r2": 0.5,
        "median_absolute_error": 0.15,
    }

    result = add_rating_scale_metrics(metrics)

    assert result["rmse_rating_scale"] == pytest.approx(1.0)
    assert result["mae_rating_scale"] == pytest.approx(0.8)

    assert result["rmse"] == pytest.approx(0.25)
    assert result["mae"] == pytest.approx(0.20)


def test_custom_rating_scale_metrics() -> None:
    """A conversão deve aceitar uma escala de ratings personalizada."""
    metrics = {
        "rmse": 0.10,
        "mae": 0.05,
    }

    result = add_rating_scale_metrics(
        metrics,
        min_rating=0.0,
        max_rating=10.0,
    )

    assert result["rmse_rating_scale"] == pytest.approx(1.0)
    assert result["mae_rating_scale"] == pytest.approx(0.5)


def test_metrics_accept_python_lists() -> None:
    """As métricas devem aceitar listas Python além de arrays NumPy."""
    y_true = [0.0, 0.5, 1.0]
    y_pred = [0.1, 0.4, 0.9]

    assert rmse(y_true, y_pred) >= 0.0
    assert mae(y_true, y_pred) >= 0.0


def test_metrics_reject_different_array_sizes() -> None:
    """As métricas devem rejeitar arrays com tamanhos diferentes."""
    y_true = np.array([0.0, 0.5, 1.0])
    y_pred = np.array([0.0, 0.5])

    with pytest.raises(ValueError, match="mesmo tamanho"):
        rmse(y_true, y_pred)


def test_metrics_reject_empty_arrays() -> None:
    """As métricas devem rejeitar arrays vazios."""
    y_true = np.array([])
    y_pred = np.array([])

    with pytest.raises(ValueError, match="não podem estar vazios"):
        mae(y_true, y_pred)


def test_metrics_reject_nan_values() -> None:
    """As métricas devem rejeitar valores NaN."""
    y_true = np.array([0.0, np.nan, 1.0])
    y_pred = np.array([0.0, 0.5, 1.0])

    with pytest.raises(ValueError, match="NaN ou infinitos"):
        mse(y_true, y_pred)


def test_metrics_reject_infinite_values() -> None:
    """As métricas devem rejeitar valores infinitos."""
    y_true = np.array([0.0, 0.5, 1.0])
    y_pred = np.array([0.0, np.inf, 1.0])

    with pytest.raises(ValueError, match="NaN ou infinitos"):
        rmse(y_true, y_pred)


def test_rating_scale_rejects_invalid_range() -> None:
    """A conversão deve rejeitar uma escala sem intervalo válido."""
    metrics = {
        "rmse": 0.25,
        "mae": 0.20,
    }

    with pytest.raises(
        ValueError,
        match="max_rating deve ser maior que min_rating",
    ):
        add_rating_scale_metrics(
            metrics,
            min_rating=5.0,
            max_rating=1.0,
        )


def test_rating_scale_requires_rmse_and_mae() -> None:
    """A conversão deve exigir RMSE e MAE no dicionário."""
    metrics = {
        "mse": 0.0625,
        "r2": 0.5,
    }

    with pytest.raises(KeyError, match="rmse.*mae"):
        add_rating_scale_metrics(metrics)
