"""Pipeline principal de treinamento."""

import json
from pathlib import Path
from typing import Any, cast

import mlflow
import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray
from torch.utils.data import DataLoader

from recommender.data.dataset import MovieLensDataset
from recommender.data.movielens_loader import load_ratings
from recommender.data.preprocess import encode_ids, normalize_ratings
from recommender.data.split import split_data
from recommender.evaluation.evaluator import Evaluator
from recommender.evaluation.metric_strategy import MAEStrategy, RMSEStrategy
from recommender.models.base import BaseRecommender
from recommender.models.model_factory import create_model
from recommender.pipeline.base_pipeline import BasePipeline
from recommender.repositories.artifact_repository import ArtifactRepository
from recommender.tracking.mlflow_tracker import MLflowTracker
from recommender.training.loss import get_loss_function
from recommender.training.optimizer_factory import create_optimizer
from recommender.training.trainer import Trainer
from recommender.utils.config import load_yaml
from recommender.utils.seed import set_seed

Batch = tuple[torch.Tensor, torch.Tensor, torch.Tensor]
MLflowInputExample = NDArray[np.int64]


class TrainingPipeline(BasePipeline):
    """Pipeline de treino do sistema de recomendação.

    Attributes:
        params: Configurações carregadas do arquivo YAML.
        repository: Repositório para persistência de artefatos.
        processed_dir: Diretório onde os dados processados são salvos.
        checkpoint_path: Caminho para o checkpoint do modelo.
        metadata_path: Caminho para o JSON de metadados do modelo.
    """

    def __init__(self, params_path: str) -> None:
        """Inicializa pipeline.

        Args:
            params_path: Caminho para o arquivo de configuração YAML.
        """
        self.params = load_yaml(params_path)
        self.repository = ArtifactRepository()

        self.processed_dir = Path(self.params["data"]["processed_dir"])
        self.checkpoint_path = Path("models/checkpoints/model.pt")
        self.metadata_path = Path("models/checkpoints/model_metadata.json")

    def prepare(self) -> None:
        """Prepara dados brutos e salva CSVs processados."""
        data = load_ratings(self.params["data"]["raw_path"])
        data = normalize_ratings(encode_ids(data))

        metadata = self._build_model_metadata(data)

        train, valid, test = split_data(
            data,
            self.params["data"]["test_size"],
            self.params["data"]["valid_size"],
            self.params["seed"],
        )

        self._save_splits(train, valid, test)
        self._save_json(metadata, self.metadata_path)

    def train(self) -> None:
        """Treina modelo, salva checkpoint e registra experimento."""
        set_seed(self.params["seed"])

        train, valid, test = self._load_processed_splits()

        metadata = self._load_or_create_metadata(train, valid, test)
        model = self._create_model_from_metadata(metadata)

        train_loss = self._fit_model(model, train)

        self.repository.save_model(model, self.checkpoint_path)
        self.repository.save_metrics(
            {"train_loss": train_loss},
            self._train_metrics_path(),
        )

        self._log_experiment(model, train, train_loss, str(self.checkpoint_path))

    def evaluate(self) -> None:
        """Avalia modelo treinado em dados de teste.

        Raises:
            FileNotFoundError: Se o checkpoint do modelo não for encontrado.
        """
        if not self.checkpoint_path.exists():
            msg = (
                "Checkpoint não encontrado em models/checkpoints/model.pt. "
                "Execute primeiro: uv run python scripts/run_pipeline.py train"
            )
            raise FileNotFoundError(msg)

        _, _, test = self._load_processed_splits()

        metadata = self._load_model_metadata()
        model = self._create_model_from_metadata(metadata)

        state_dict = torch.load(self.checkpoint_path, map_location="cpu")
        model.load_state_dict(state_dict)

        metrics = self._calculate_metrics(model, test)

        self.repository.save_metrics(
            metrics,
            self._eval_metrics_path(),
        )

        print("Métricas de avaliação:")
        print(metrics)

    def _save_splits(
        self,
        train: pd.DataFrame,
        valid: pd.DataFrame,
        test: pd.DataFrame,
    ) -> None:
        """Salva divisões processadas.

        Args:
            train: Dados de treino.
            valid: Dados de validação.
            test: Dados de teste.
        """
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        train.to_csv(self.processed_dir / "train.csv", index=False)
        valid.to_csv(self.processed_dir / "valid.csv", index=False)
        test.to_csv(self.processed_dir / "test.csv", index=False)

    def _load_processed_splits(
        self,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Carrega treino, validação e teste.

        Returns:
            Tupla contendo os DataFrames de treino, validação e teste.

        Raises:
            FileNotFoundError: Se algum arquivo de split não for encontrado.
        """
        train_path = self.processed_dir / "train.csv"
        valid_path = self.processed_dir / "valid.csv"
        test_path = self.processed_dir / "test.csv"

        missing_files = [
            str(path)
            for path in [train_path, valid_path, test_path]
            if not path.exists()
        ]

        if missing_files:
            msg = (
                "Arquivos processados não encontrados: "
                f"{missing_files}. "
                "Execute primeiro: uv run python scripts/run_pipeline.py prepare"
            )
            raise FileNotFoundError(msg)

        train = pd.read_csv(train_path)
        valid = pd.read_csv(valid_path)
        test = pd.read_csv(test_path)

        return train, valid, test

    def _build_model_metadata(self, data: pd.DataFrame) -> dict[str, Any]:
        """Cria metadados consistentes para o modelo.

        Args:
            data: DataFrame usado para calcular usuários e itens únicos.

        Returns:
            Dicionário com configurações de arquitetura e dimensões.
        """
        return {
            "model_name": self.params["model"]["name"],
            "num_users": int(data["user_idx"].max()) + 1,
            "num_items": int(data["item_idx"].max()) + 1,
            "embedding_dim": int(self.params["model"]["embedding_dim"]),
            "hidden_dim": int(self.params["model"]["hidden_dim"]),
            "seed": int(self.params["seed"]),
        }

    def _load_or_create_metadata(
        self,
        train: pd.DataFrame,
        valid: pd.DataFrame,
        test: pd.DataFrame,
    ) -> dict[str, Any]:
        """Carrega metadados ou cria a partir de todos os splits.

        Args:
            train: Dados de treino.
            valid: Dados de validação.
            test: Dados de teste.

        Returns:
            Dicionário de metadados do modelo.
        """
        if self.metadata_path.exists():
            return self._load_model_metadata()

        full_data = pd.concat([train, valid, test], ignore_index=True)
        metadata = self._build_model_metadata(full_data)
        self._save_json(metadata, self.metadata_path)

        return metadata

    def _load_model_metadata(self) -> dict[str, Any]:
        """Carrega metadados do modelo.

        Returns:
            Dicionário de metadados recuperados ou criados.
        """
        if not self.metadata_path.exists():
            train, valid, test = self._load_processed_splits()
            full_data = pd.concat([train, valid, test], ignore_index=True)
            metadata = self._build_model_metadata(full_data)
            self._save_json(metadata, self.metadata_path)

            return metadata

        with self.metadata_path.open("r", encoding="utf-8") as file:
            metadata = json.load(file)

        return cast(dict[str, Any], metadata)

    def _save_json(self, data: dict[str, Any], path: Path) -> None:
        """Salva dicionário em JSON.

        Args:
            data: Dicionário com os dados a serem serializados.
            path: Caminho completo de destino do arquivo JSON.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def _create_model_from_metadata(
        self,
        metadata: dict[str, Any],
    ) -> BaseRecommender:
        """Cria modelo usando os mesmos metadados do treino.

        Args:
            metadata: Dicionário contendo hiperparâmetros e dimensões.

        Returns:
            Instância configurada do modelo de recomendação.
        """
        return create_model(
            model_name=str(metadata["model_name"]),
            num_users=int(metadata["num_users"]),
            num_items=int(metadata["num_items"]),
            embedding_dim=int(metadata["embedding_dim"]),
            hidden_dim=int(metadata["hidden_dim"]),
        )

    def _fit_model(self, model: BaseRecommender, train: pd.DataFrame) -> float:
        """Executa loop de treino.

        Args:
            model: Modelo a ser treinado.
            train: DataFrame contendo os dados de treino.

        Returns:
            Valor final da perda da última época.
        """
        data_loader = self._create_dataloader(train, shuffle=True)
        optimizer = create_optimizer(
            model,
            self.params["training"]["learning_rate"],
        )
        trainer = Trainer(model, get_loss_function(), optimizer)

        loss = 0.0

        for epoch in range(self.params["training"]["epochs"]):
            loss = trainer.train_epoch(data_loader)
            print(f"Epoch {epoch + 1} - loss: {loss:.6f}")

        return loss

    def _log_experiment(
        self,
        model: BaseRecommender,
        train: pd.DataFrame,
        train_loss: float,
        model_path: str,
    ) -> None:
        """Registra parâmetros, métricas e artefatos no MLflow.

        O registro no MLflow é tratado como etapa complementar. Caso o servidor
        MLflow esteja indisponível, o checkpoint salvo localmente continua válido
        e o treino não é interrompido por falha de tracking.

        Args:
            model: Modelo treinado.
            train: Dados de treino usados para criar o exemplo de entrada.
            train_loss: Valor final de perda obtido no treinamento.
            model_path: Caminho local onde o checkpoint foi salvo.
        """
        input_example = self._create_mlflow_input_example(train)

        try:
            tracker = MLflowTracker("movielens-recommender")

            with mlflow.start_run():
                tracker.log_params(self.params["model"] | self.params["training"])
                tracker.log_metrics({"train_loss": train_loss})

                mlflow.log_artifact(
                    model_path,
                    artifact_path="checkpoints",
                )
                mlflow.log_artifact(
                    str(self.metadata_path),
                    artifact_path="checkpoints",
                )

                try:
                    tracker.log_model(
                        model=model,
                        artifact_path="model",
                        input_example=input_example,
                    )
                    mlflow.set_tag("model_logging_status", "success")
                except Exception as error:  # noqa: BLE001
                    mlflow.set_tag("model_logging_status", "failed")
                    mlflow.set_tag("model_logging_error", str(error))
                    print(
                        "Aviso: checkpoint salvo, mas o registro do modelo "
                        f"no MLflow falhou: {error}"
                    )

        except Exception as error:  # noqa: BLE001
            print(f"Aviso: checkpoint salvo, mas o registro no MLflow falhou: {error}")

    def _create_mlflow_input_example(
        self,
        train: pd.DataFrame,
    ) -> MLflowInputExample:
        """Cria exemplo de entrada aceito pelo MLflow.

        O modelo original trabalha com dois tensores:

            model(user_ids, item_ids)

        Para o MLflow, criamos uma única matriz NumPy:

            [[user_id, item_id],
             [user_id, item_id],
             ...]

        Args:
            train: DataFrame de treino usado para montar o exemplo.

        Returns:
            Matriz NumPy com shape ``(batch_size, 2)``.
        """
        data_loader = self._create_dataloader(train, shuffle=False)

        try:
            users, items, _ = next(iter(data_loader))
        except StopIteration as error:
            msg = "Não é possível criar input_example com treino vazio."
            raise ValueError(msg) from error

        users_array = users[:5].detach().cpu().long().reshape(-1).numpy()
        items_array = items[:5].detach().cpu().long().reshape(-1).numpy()

        if users_array.shape != items_array.shape:
            msg = (
                "users e items devem ter o mesmo tamanho para o input_example. "
                f"Recebido: {users_array.shape} e {items_array.shape}."
            )
            raise ValueError(msg)

        if users_array.size == 0:
            msg = "input_example não pode ser vazio."
            raise ValueError(msg)

        input_example = np.column_stack((users_array, items_array)).astype(
            np.int64,
            copy=False,
        )

        return input_example

    def _calculate_metrics(
        self,
        model: BaseRecommender,
        test: pd.DataFrame,
    ) -> dict[str, float]:
        """Calcula métricas de avaliação.

        Args:
            model: Modelo a ser avaliado.
            test: DataFrame contendo os dados de teste.

        Returns:
            Dicionário associando o nome da métrica ao seu valor.
        """
        data_loader = self._create_dataloader(test, shuffle=False)
        evaluator = Evaluator(model)

        return {
            "rmse": evaluator.evaluate(data_loader, RMSEStrategy()),
            "mae": evaluator.evaluate(data_loader, MAEStrategy()),
        }

    def _create_dataloader(
        self,
        data: pd.DataFrame,
        shuffle: bool,
    ) -> DataLoader[Batch]:
        """Cria DataLoader PyTorch.

        Args:
            data: DataFrame usado para gerar o Dataset.
            shuffle: Se verdadeiro, embaralha as amostras.

        Returns:
            DataLoader preparado para iteração em lotes.
        """
        dataset = MovieLensDataset(data)

        return DataLoader(
            dataset,
            batch_size=self.params["training"]["batch_size"],
            shuffle=shuffle,
        )

    def _train_metrics_path(self) -> str:
        """Retorna caminho de métricas de treino.

        Returns:
            Caminho relativo para o JSON de métricas de treino.
        """
        return "reports/metrics/train_metrics.json"

    def _eval_metrics_path(self) -> str:
        """Retorna caminho de métricas de avaliação.

        Returns:
            Caminho relativo para o JSON de métricas de teste.
        """
        return "reports/metrics/evaluation_metrics.json"
