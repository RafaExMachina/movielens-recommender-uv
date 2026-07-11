"""Estratégias de métricas de avaliação."""

from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt

from recommender.evaluation.metrics import mae, rmse

FloatArray = npt.NDArray[np.float64]


class MetricStrategy(ABC):
    """Interface para estratégias de métricas."""

    @abstractmethod
    def calculate(self, y_true: FloatArray, y_pred: FloatArray) -> float:
        """Calcula uma métrica.

        Args:
            y_true: Valores reais.
            y_pred: Valores previstos pelo modelo.

        Returns:
            Valor calculado da métrica.
        """


class RMSEStrategy(MetricStrategy):
    """Estratégia para cálculo de RMSE."""

    def calculate(self, y_true: FloatArray, y_pred: FloatArray) -> float:
        """Calcula RMSE.

        Args:
            y_true: Valores reais.
            y_pred: Valores previstos pelo modelo.

        Returns:
            Valor da métrica RMSE.
        """
        return rmse(y_true, y_pred)


class MAEStrategy(MetricStrategy):
    """Estratégia para cálculo de MAE."""

    def calculate(self, y_true: FloatArray, y_pred: FloatArray) -> float:
        """Calcula MAE.

        Args:
            y_true: Valores reais.
            y_pred: Valores previstos pelo modelo.

        Returns:
            Valor da métrica MAE.
        """
        return mae(y_true, y_pred)
