"""Avaliação de modelos PyTorch."""

from typing import cast

import numpy as np
import numpy.typing as npt
import torch
from torch.utils.data import DataLoader

from recommender.evaluation.metric_strategy import MetricStrategy
from recommender.evaluation.metrics import (
    add_rating_scale_metrics,
    compute_regression_metrics,
)

Batch = tuple[torch.Tensor, torch.Tensor, torch.Tensor]
FloatArray = npt.NDArray[np.float64]
EvaluationMetrics = dict[str, float]


class Evaluator:
    """Classe responsável pela avaliação de modelos PyTorch."""

    def __init__(self, model: torch.nn.Module) -> None:
        """Inicializa o avaliador.

        Args:
            model: Modelo PyTorch que será avaliado.
        """
        self.model = model

    def _model_device(self) -> torch.device:
        """Obtém o dispositivo no qual o modelo está armazenado.

        Returns:
            Dispositivo utilizado pelo modelo. Caso o modelo não possua
            parâmetros, retorna CPU.
        """
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def predict(
        self,
        data_loader: DataLoader[Batch],
    ) -> tuple[FloatArray, FloatArray]:
        """Gera predições e recupera os valores reais.

        Os tensores de usuários e itens são enviados para o mesmo dispositivo
        do modelo. As predições e os alvos são convertidos para arrays NumPy
        unidimensionais em CPU.

        Args:
            data_loader: DataLoader contendo usuários, itens e avaliações.

        Returns:
            Tupla contendo os valores reais e os valores previstos.

        Raises:
            ValueError: Se o DataLoader não produzir nenhuma observação.
        """
        self.model.eval()
        device = self._model_device()

        predictions: list[float] = []
        targets: list[float] = []

        with torch.no_grad():
            for users, items, ratings in data_loader:
                users = users.to(device)
                items = items.to(device)

                batch_predictions = self.model(users, items)

                prediction_values = (
                    batch_predictions.detach()
                    .cpu()
                    .reshape(-1)
                    .numpy()
                    .astype(np.float64)
                )

                target_values = (
                    ratings.detach().cpu().reshape(-1).numpy().astype(np.float64)
                )

                predictions.extend(float(value) for value in prediction_values)
                targets.extend(float(value) for value in target_values)

        if not targets:
            message = "O DataLoader de avaliação não possui observações."
            raise ValueError(message)

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

        Este método foi mantido para preservar compatibilidade com o código
        existente que utiliza RMSEStrategy, MAEStrategy ou outra implementação
        de MetricStrategy.

        Args:
            data_loader: DataLoader contendo os dados de avaliação.
            metric: Estratégia utilizada para calcular a métrica.

        Returns:
            Valor calculado pela estratégia informada.
        """
        targets, predictions = self.predict(data_loader)

        return metric.calculate(targets, predictions)

    def evaluate_all(
        self,
        data_loader: DataLoader[Batch],
        min_rating: float = 1.0,
        max_rating: float = 5.0,
        include_rating_scale: bool = True,
    ) -> EvaluationMetrics:
        """Calcula todas as métricas de regressão do projeto.

        As métricas principais são calculadas usando os ratings normalizados:

        - RMSE;
        - MAE;
        - MSE;
        - R²;
        - erro absoluto mediano.

        Opcionalmente, RMSE e MAE também são convertidos para a escala
        original de ratings.

        Args:
            data_loader: DataLoader contendo os dados de avaliação.
            min_rating: Menor valor da escala original de ratings.
            max_rating: Maior valor da escala original de ratings.
            include_rating_scale: Indica se RMSE e MAE na escala original
                devem ser acrescentados ao resultado.

        Returns:
            Dicionário contendo as métricas calculadas.
        """
        targets, predictions = self.predict(data_loader)

        metrics = compute_regression_metrics(
            y_true=targets,
            y_pred=predictions,
        )

        if not include_rating_scale:
            return metrics

        return add_rating_scale_metrics(
            metrics=metrics,
            min_rating=min_rating,
            max_rating=max_rating,
        )
