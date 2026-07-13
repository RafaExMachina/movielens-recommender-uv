"""Métricas de avaliação para problemas de regressão."""

from collections.abc import Mapping

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]


def _prepare_arrays(
    y_true: npt.ArrayLike,
    y_pred: npt.ArrayLike,
) -> tuple[FloatArray, FloatArray]:
    """Converte e valida os vetores usados nas métricas.

    Args:
        y_true: Valores reais.
        y_pred: Valores previstos pelo modelo.

    Returns:
        Tupla contendo os valores reais e previstos como arrays
        unidimensionais de ponto flutuante.

    Raises:
        ValueError: Se os arrays estiverem vazios, possuírem dimensões
            diferentes ou contiverem valores não finitos.
    """
    true_values = np.asarray(y_true, dtype=np.float64).reshape(-1)
    predicted_values = np.asarray(y_pred, dtype=np.float64).reshape(-1)

    if true_values.size == 0 or predicted_values.size == 0:
        msg = "Os arrays de valores reais e previstos não podem estar vazios."
        raise ValueError(msg)

    if true_values.shape != predicted_values.shape:
        msg = (
            "Os arrays y_true e y_pred devem possuir o mesmo tamanho. "
            f"Recebidos: {true_values.shape} e {predicted_values.shape}."
        )
        raise ValueError(msg)

    if not np.all(np.isfinite(true_values)):
        msg = "O array y_true contém valores NaN ou infinitos."
        raise ValueError(msg)

    if not np.all(np.isfinite(predicted_values)):
        msg = "O array y_pred contém valores NaN ou infinitos."
        raise ValueError(msg)

    return true_values, predicted_values


def mse(y_true: npt.ArrayLike, y_pred: npt.ArrayLike) -> float:
    """Calcula o erro quadrático médio.

    Args:
        y_true: Valores reais.
        y_pred: Valores previstos pelo modelo.

    Returns:
        Valor da métrica MSE.
    """
    true_values, predicted_values = _prepare_arrays(y_true, y_pred)
    squared_errors = (true_values - predicted_values) ** 2

    return float(np.mean(squared_errors))


def rmse(y_true: npt.ArrayLike, y_pred: npt.ArrayLike) -> float:
    """Calcula a raiz do erro quadrático médio.

    Args:
        y_true: Valores reais.
        y_pred: Valores previstos pelo modelo.

    Returns:
        Valor da métrica RMSE.
    """
    return float(np.sqrt(mse(y_true, y_pred)))


def mae(y_true: npt.ArrayLike, y_pred: npt.ArrayLike) -> float:
    """Calcula o erro absoluto médio.

    Args:
        y_true: Valores reais.
        y_pred: Valores previstos pelo modelo.

    Returns:
        Valor da métrica MAE.
    """
    true_values, predicted_values = _prepare_arrays(y_true, y_pred)
    absolute_errors = np.abs(true_values - predicted_values)

    return float(np.mean(absolute_errors))


def median_absolute_error(
    y_true: npt.ArrayLike,
    y_pred: npt.ArrayLike,
) -> float:
    """Calcula a mediana dos erros absolutos.

    Essa métrica é menos sensível a valores extremos do que o MAE.

    Args:
        y_true: Valores reais.
        y_pred: Valores previstos pelo modelo.

    Returns:
        Mediana dos erros absolutos.
    """
    true_values, predicted_values = _prepare_arrays(y_true, y_pred)
    absolute_errors = np.abs(true_values - predicted_values)

    return float(np.median(absolute_errors))


def r2(y_true: npt.ArrayLike, y_pred: npt.ArrayLike) -> float:
    """Calcula o coeficiente de determinação R².

    O R² indica a proporção da variabilidade dos valores reais explicada
    pelo modelo. O melhor valor possível é 1. Valores negativos indicam
    que o modelo é pior do que prever constantemente a média.

    Para um vetor real constante, retorna 1 quando a previsão é perfeita
    e 0 quando a previsão é diferente.

    Args:
        y_true: Valores reais.
        y_pred: Valores previstos pelo modelo.

    Returns:
        Valor do coeficiente de determinação R².
    """
    true_values, predicted_values = _prepare_arrays(y_true, y_pred)

    residual_sum = float(np.sum((true_values - predicted_values) ** 2))
    total_sum = float(np.sum((true_values - np.mean(true_values)) ** 2))

    if np.isclose(total_sum, 0.0):
        return 1.0 if np.isclose(residual_sum, 0.0) else 0.0

    return float(1.0 - residual_sum / total_sum)


def compute_regression_metrics(
    y_true: npt.ArrayLike,
    y_pred: npt.ArrayLike,
) -> dict[str, float]:
    """Calcula todas as métricas de regressão utilizadas pelo projeto.

    Args:
        y_true: Valores reais.
        y_pred: Valores previstos pelo modelo.

    Returns:
        Dicionário contendo RMSE, MAE, MSE, R² e erro absoluto mediano.
    """
    return {
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "mse": mse(y_true, y_pred),
        "r2": r2(y_true, y_pred),
        "median_absolute_error": median_absolute_error(y_true, y_pred),
    }


def add_rating_scale_metrics(
    metrics: Mapping[str, float],
    min_rating: float = 1.0,
    max_rating: float = 5.0,
) -> dict[str, float]:
    """Acrescenta RMSE e MAE convertidos para a escala original de ratings.

    O projeto normaliza ratings usando:

        rating_norm = (rating - min_rating) / (max_rating - min_rating)

    Portanto, erros como RMSE e MAE podem ser convertidos para a escala
    original multiplicando-os pelo intervalo dos ratings.

    Args:
        metrics: Métricas calculadas na escala normalizada.
        min_rating: Menor rating da escala original.
        max_rating: Maior rating da escala original.

    Returns:
        Novo dicionário contendo as métricas normalizadas e as métricas
        convertidas para a escala original.

    Raises:
        ValueError: Se max_rating for menor ou igual a min_rating.
        KeyError: Se RMSE ou MAE não estiverem presentes.
    """
    if max_rating <= min_rating:
        msg = "max_rating deve ser maior que min_rating."
        raise ValueError(msg)

    if "rmse" not in metrics or "mae" not in metrics:
        msg = "As métricas de entrada devem conter 'rmse' e 'mae'."
        raise KeyError(msg)

    rating_range = max_rating - min_rating
    result = dict(metrics)

    result["rmse_rating_scale"] = metrics["rmse"] * rating_range
    result["mae_rating_scale"] = metrics["mae"] * rating_range

    return result
