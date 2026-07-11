"""Testes de métricas."""

import numpy as np

from recommender.evaluation.metrics import mae, rmse


def test_metrics_are_positive() -> None:
    """RMSE e MAE devem ser não negativos."""
    y_true = np.array([1.0, 0.5, 0.0])
    y_pred = np.array([0.8, 0.5, 0.1])

    assert rmse(y_true, y_pred) >= 0
    assert mae(y_true, y_pred) >= 0
