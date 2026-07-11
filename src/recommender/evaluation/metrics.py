"""Métricas de avaliação."""

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]


def rmse(y_true: FloatArray, y_pred: FloatArray) -> float:
    """Calcula a raiz do erro quadrático médio.

    Args:
        y_true: Valores reais.
        y_pred: Valores previstos pelo modelo.

    Returns:
        Valor da métrica RMSE.
    """
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: FloatArray, y_pred: FloatArray) -> float:
    """Calcula o erro absoluto médio.

    Args:
        y_true: Valores reais.
        y_pred: Valores previstos pelo modelo.

    Returns:
        Valor da métrica MAE.
    """
    return float(np.mean(np.abs(y_true - y_pred)))
