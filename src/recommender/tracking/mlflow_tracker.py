"""Integração com MLflow."""

from typing import Any, cast

import mlflow
import mlflow.pytorch
import numpy as np
import torch
from mlflow.models import infer_signature
from numpy.typing import NDArray
from torch import nn

InputExample = tuple[torch.Tensor, torch.Tensor] | NDArray[np.int64]


class MLflowRecommenderWrapper(nn.Module):
    """Adapta modelos de recomendação para o formato aceito pelo MLflow.

    O modelo original recebe dois tensores:

        model(user_ids, item_ids)

    O MLflow trabalha melhor com uma única entrada tabular.
    Este wrapper transforma uma matriz com duas colunas em dois tensores:

        x[:, 0] = user_id
        x[:, 1] = item_id
    """

    def __init__(self, model: nn.Module) -> None:
        """Inicializa o wrapper.

        Args:
            model: Modelo PyTorch original de recomendação.
        """
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Executa predição usando entrada no formato aceito pelo MLflow.

        Args:
            x: Tensor com shape ``(batch_size, 2)``.
               A primeira coluna contém user_id.
               A segunda coluna contém item_id.

        Returns:
            Tensor com as predições do modelo.
        """
        if x.ndim != 2 or x.shape[1] != 2:
            raise ValueError(
                "A entrada deve ter shape (batch_size, 2), "
                "com colunas [user_id, item_id]."
            )

        user_ids = x[:, 0].long()
        item_ids = x[:, 1].long()

        return cast(torch.Tensor, self.model(user_ids, item_ids))


class MLflowTracker:
    """Classe auxiliar para rastrear experimentos no MLflow."""

    def __init__(self, experiment_name: str) -> None:
        """Define o experimento ativo.

        Args:
            experiment_name: Nome do experimento no MLflow.
        """
        mlflow.set_experiment(experiment_name)

    def log_params(self, params: dict[str, Any]) -> None:
        """Registra parâmetros no MLflow.

        Args:
            params: Dicionário com parâmetros do experimento.
        """
        mlflow.log_params(params)

    def log_metrics(self, metrics: dict[str, float]) -> None:
        """Registra métricas no MLflow.

        Args:
            metrics: Dicionário com métricas do experimento.
        """
        mlflow.log_metrics(metrics)

    def log_model(
        self,
        model: nn.Module,
        artifact_path: str,
        input_example: InputExample,
    ) -> None:
        """Registra modelo PyTorch no MLflow.

        O ``input_example`` pode ser recebido no formato original do projeto:

            (user_ids, item_ids)

        Internamente ele é convertido para uma matriz NumPy:

            [[user_id, item_id],
             [user_id, item_id],
             ...]

        Isso evita o erro do MLflow com entrada do tipo ``tuple``.

        Args:
            model: Modelo PyTorch treinado.
            artifact_path: Caminho do artefato dentro do run do MLflow.
            input_example: Exemplo de entrada usado pelo MLflow.
        """
        wrapped_model = MLflowRecommenderWrapper(model)
        wrapped_model.eval()

        mlflow_input_example = self._as_mlflow_input_example(input_example)

        device = self._get_model_device(model)
        example_tensor = torch.as_tensor(
            mlflow_input_example,
            dtype=torch.long,
            device=device,
        )

        with torch.no_grad():
            output_example = wrapped_model(example_tensor).detach().cpu().numpy()

        signature = infer_signature(mlflow_input_example, output_example)

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
        """Converte exemplo de entrada para o formato aceito pelo MLflow.

        Args:
            input_example: Tupla ``(user_ids, item_ids)`` ou matriz NumPy.

        Returns:
            Matriz NumPy com shape ``(batch_size, 2)``.
        """
        if isinstance(input_example, tuple):
            user_ids, item_ids = input_example

            user_ids_array = MLflowTracker._tensor_to_numpy_ids(user_ids)
            item_ids_array = MLflowTracker._tensor_to_numpy_ids(item_ids)

            if user_ids_array.shape != item_ids_array.shape:
                raise ValueError(
                    "user_ids e item_ids devem ter o mesmo tamanho. "
                    f"Recebido: {user_ids_array.shape} e {item_ids_array.shape}."
                )

            if user_ids_array.size == 0:
                raise ValueError("input_example não pode ser vazio.")

            stacked = np.column_stack((user_ids_array, item_ids_array))
            return stacked.astype(np.int64, copy=False)

        array = np.asarray(input_example, dtype=np.int64)

        if array.ndim != 2 or array.shape[1] != 2:
            raise ValueError(
                "input_example em NumPy deve ter shape (batch_size, 2), "
                "com colunas [user_id, item_id]."
            )

        if array.shape[0] == 0:
            raise ValueError("input_example não pode ser vazio.")

        return array

    @staticmethod
    def _tensor_to_numpy_ids(tensor: torch.Tensor) -> NDArray[np.int64]:
        """Converte tensor de ids para NumPy int64 unidimensional.

        Args:
            tensor: Tensor com ids de usuários ou itens.

        Returns:
            Array NumPy unidimensional com dtype int64.
        """
        array = tensor.detach().cpu().long().reshape(-1).numpy()
        return array.astype(np.int64, copy=False)

    @staticmethod
    def _get_model_device(model: nn.Module) -> torch.device:
        """Obtém o device atual do modelo.

        Args:
            model: Modelo PyTorch.

        Returns:
            Device onde o modelo está alocado.
        """
        try:
            return next(model.parameters()).device
        except StopIteration:
            return torch.device("cpu")
