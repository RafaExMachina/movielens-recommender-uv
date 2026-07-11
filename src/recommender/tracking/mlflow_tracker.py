"""Integração do sistema de recomendação com o MLflow."""

from typing import Any

import mlflow
import mlflow.pytorch
import numpy as np
import torch
from mlflow.models import infer_signature
from numpy.typing import NDArray
from torch import nn

from recommender.tracking.mlflow_wrapper import MLflowRecommenderWrapper
from recommender.utils.settings import get_settings

InputExample = tuple[torch.Tensor, torch.Tensor] | NDArray[np.int64]


class MLflowTracker:
    """Gerencia o rastreamento de experimentos no MLflow."""

    def __init__(self, experiment_name: str) -> None:
        """Configura o servidor de tracking e o experimento ativo.

        Args:
            experiment_name: Nome do experimento registrado no MLflow.

        Raises:
            ValueError: Se o nome do experimento estiver vazio.
        """
        normalized_name = experiment_name.strip()

        if not normalized_name:
            raise ValueError("O nome do experimento não pode ser vazio.")

        settings = get_settings()

        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        mlflow.set_experiment(normalized_name)

    def log_params(self, params: dict[str, Any]) -> None:
        """Registra parâmetros na execução ativa.

        Args:
            params: Parâmetros do experimento.
        """
        mlflow.log_params(params)

    def log_metrics(self, metrics: dict[str, float]) -> None:
        """Registra métricas na execução ativa.

        Args:
            metrics: Métricas calculadas durante o experimento.
        """
        mlflow.log_metrics(metrics)

    def log_model(
        self,
        model: nn.Module,
        artifact_path: str,
        input_example: InputExample,
    ) -> None:
        """Registra um modelo PyTorch no MLflow.

        O modelo original recebe dois tensores separados:

        ``model(user_ids, item_ids)``

        Para o registro no MLflow, os dados são convertidos para uma
        matriz com duas colunas:

        ``[[user_id, item_id], ...]``

        Args:
            model: Modelo PyTorch treinado.
            artifact_path: Caminho do artefato dentro da execução do MLflow.
            input_example: Exemplo de entrada como uma tupla de tensores ou
                uma matriz NumPy com duas colunas.
        """
        mlflow_input_example = self._as_mlflow_input_example(input_example)

        wrapped_model = MLflowRecommenderWrapper(model)
        wrapped_model.eval()

        device = self._get_model_device(model)
        wrapped_model.to(device)

        example_tensor = torch.as_tensor(
            mlflow_input_example,
            dtype=torch.long,
            device=device,
        )

        with torch.no_grad():
            output_example = wrapped_model(example_tensor).detach().cpu().numpy()

        signature = infer_signature(
            mlflow_input_example,
            output_example,
        )

        mlflow.pytorch.log_model(
            pytorch_model=wrapped_model,
            artifact_path=artifact_path,
            input_example=mlflow_input_example,
            signature=signature,
        )

    @staticmethod
    def _as_mlflow_input_example(
        input_example: InputExample,
    ) -> NDArray[np.int64]:
        """Converte o exemplo de entrada para o formato tabular do MLflow.

        Args:
            input_example: Tupla ``(user_ids, item_ids)`` ou matriz NumPy
                com shape ``(batch_size, 2)``.

        Returns:
            Matriz NumPy com shape ``(batch_size, 2)`` e tipo ``int64``.

        Raises:
            ValueError: Se a entrada estiver vazia, tiver dimensões inválidas
                ou contiver quantidades diferentes de usuários e itens.
        """
        if isinstance(input_example, tuple):
            user_ids, item_ids = input_example

            user_ids_array = MLflowTracker._tensor_to_numpy_ids(user_ids)
            item_ids_array = MLflowTracker._tensor_to_numpy_ids(item_ids)

            if user_ids_array.shape != item_ids_array.shape:
                raise ValueError(
                    "user_ids e item_ids devem ter o mesmo tamanho. "
                    f"Recebido: {user_ids_array.shape} e "
                    f"{item_ids_array.shape}."
                )

            if user_ids_array.size == 0:
                raise ValueError("input_example não pode ser vazio.")

            stacked = np.column_stack(
                (user_ids_array, item_ids_array),
            )

            return stacked.astype(np.int64, copy=False)

        array = np.asarray(input_example, dtype=np.int64)

        if array.ndim != 2 or array.shape[1] != 2:
            raise ValueError(
                "input_example em NumPy deve ter shape "
                "(batch_size, 2), com colunas [user_id, item_id]."
            )

        if array.shape[0] == 0:
            raise ValueError("input_example não pode ser vazio.")

        return array

    @staticmethod
    def _tensor_to_numpy_ids(
        tensor: torch.Tensor,
    ) -> NDArray[np.int64]:
        """Converte um tensor de identificadores para NumPy.

        Args:
            tensor: Tensor com identificadores de usuários ou itens.

        Returns:
            Array NumPy unidimensional com tipo ``int64``.
        """
        array = tensor.detach().cpu().long().reshape(-1).numpy()
        return array.astype(np.int64, copy=False)

    @staticmethod
    def _get_model_device(model: nn.Module) -> torch.device:
        """Obtém o dispositivo no qual o modelo está armazenado.

        Args:
            model: Modelo PyTorch.

        Returns:
            Dispositivo do primeiro parâmetro do modelo ou CPU quando o
            modelo não possui parâmetros.
        """
        try:
            return next(model.parameters()).device
        except StopIteration:
            return torch.device("cpu")
