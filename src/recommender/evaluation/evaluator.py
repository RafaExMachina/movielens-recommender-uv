"""Avaliação do modelo."""

from typing import cast

import numpy as np
import numpy.typing as npt
import torch
from torch.utils.data import DataLoader

from recommender.evaluation.metric_strategy import MetricStrategy

Batch = tuple[torch.Tensor, torch.Tensor, torch.Tensor]
FloatArray = npt.NDArray[np.float64]


class Evaluator:
    """Classe responsável pela avaliação do modelo."""

    def __init__(self, model: torch.nn.Module) -> None:
        """Inicializa o avaliador.

        Args:
            model: Modelo PyTorch que será avaliado.
        """
        self.model = model

    def predict(self, data_loader: DataLoader[Batch]) -> tuple[FloatArray, FloatArray]:
        """Gera predições e valores reais.

        Args:
            data_loader: DataLoader contendo usuários, itens e avaliações reais.

        Returns:
            Tupla contendo arrays NumPy com valores reais e valores previstos.
        """
        self.model.eval()

        predictions: list[float] = []
        targets: list[float] = []

        with torch.no_grad():
            for users, items, ratings in data_loader:
                batch_predictions = self.model(users, items)

                predictions.extend(batch_predictions.tolist())
                targets.extend(ratings.tolist())

        targets_array = cast(
            FloatArray,
            np.asarray(targets, dtype=np.float64),
        )
        predictions_array = cast(
            FloatArray,
            np.asarray(predictions, dtype=np.float64),
        )

        return targets_array, predictions_array

    def evaluate(
        self,
        data_loader: DataLoader[Batch],
        metric: MetricStrategy,
    ) -> float:
        """Avalia o modelo usando uma estratégia de métrica.

        Args:
            data_loader: DataLoader contendo os dados de avaliação.
            metric: Estratégia de métrica, como RMSE ou MAE.

        Returns:
            Valor calculado da métrica.
        """
        targets, predictions = self.predict(data_loader)

        return metric.calculate(targets, predictions)
