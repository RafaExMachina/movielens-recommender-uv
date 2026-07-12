"""Pipeline principal de treinamento."""

import copy
import json
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd
import torch
from torch.utils.data import DataLoader

from recommender.data.dataset import MovieLensDataset
from recommender.data.movielens_loader import load_ratings
from recommender.data.split import split_data
from recommender.evaluation.evaluator import Evaluator
from recommender.evaluation.metric_strategy import MAEStrategy, RMSEStrategy
from recommender.features.build_features import (
    build_model_features,
    build_model_metadata,
)
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


class TrainingPipeline(BasePipeline):
    """Pipeline de treino do sistema de recomendação."""

    def __init__(self, params_path: str | Path) -> None:
        """Inicializa o pipeline.

        Args:
            params_path: Caminho para o arquivo de parâmetros YAML.
        """
        self.params = load_yaml(params_path)
        self.repository = ArtifactRepository()

    def preprocess(self) -> None:
        """Carrega, valida, limpa e persiste os dados intermediários."""
        raw_data = load_ratings(self.params["data"]["raw_path"])
        clean_data = self._clean_ratings(raw_data)

        interim_path = Path(self.params["data"]["interim_path"])
        interim_path.parent.mkdir(parents=True, exist_ok=True)

        clean_data.to_csv(interim_path, index=False)

    def feature_engineering(self) -> None:
        """Constrói features, metadados e divisões processadas."""
        interim_path = Path(self.params["data"]["interim_path"])

        if not interim_path.exists():
            msg = (
                f"Arquivo intermediário não encontrado: {interim_path}. "
                "Execute primeiro o estágio preprocess."
            )
            raise FileNotFoundError(msg)

        clean_data = pd.read_csv(interim_path)
        features = build_model_features(clean_data)
        metadata = build_model_metadata(features)

        test_size = float(self.params["data"]["test_size"])
        valid_size = float(self.params["data"]["valid_size"])

        relative_valid_size = self._relative_valid_size(
            test_size,
            valid_size,
        )

        train, valid, test = split_data(
            features,
            test_size,
            relative_valid_size,
            int(self.params["seed"]),
        )

        self._save_splits(train, valid, test)
        self._save_metadata(metadata)

    def train(self) -> None:
        """Treina o modelo, seleciona o melhor estado e salva artefatos."""
        set_seed(int(self.params["seed"]))

        train_data = self._load_split("train.csv")
        valid_data = self._load_split("valid.csv")
        metadata = self._load_metadata()

        model = self._create_model(metadata)
        metrics = self._fit_model(
            model,
            train_data,
            valid_data,
        )

        self.repository.save_model(
            model,
            self.params["artifacts"]["model_path"],
        )
        self.repository.save_metrics(
            metrics,
            self.params["artifacts"]["train_metrics_path"],
        )

    def evaluate(self) -> None:
        """Avalia o melhor checkpoint nos dados de teste."""
        test_data = self._load_split("test.csv")
        metadata = self._load_metadata()
        model = self._create_model(metadata)

        checkpoint_path = Path(self.params["artifacts"]["model_path"])

        if not checkpoint_path.exists():
            msg = (
                f"Checkpoint não encontrado: {checkpoint_path}. "
                "Execute primeiro o estágio train."
            )
            raise FileNotFoundError(msg)

        state_dict = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )

        model.load_state_dict(state_dict)

        metrics = self._calculate_metrics(
            model,
            test_data,
        )

        self.repository.save_metrics(
            metrics,
            self.params["artifacts"]["evaluation_metrics_path"],
        )

    @staticmethod
    def _clean_ratings(data: pd.DataFrame) -> pd.DataFrame:
        """Remove registros inválidos e garante o domínio dos ratings.

        Args:
            data: DataFrame original com as avaliações.

        Returns:
            DataFrame limpo, ordenado e validado.

        Raises:
            ValueError: Se faltarem colunas, houver ratings inválidos ou se
                o conjunto ficar vazio após a limpeza.
        """
        required_columns = [
            "user_id",
            "item_id",
            "rating",
            "timestamp",
        ]

        missing_columns = set(required_columns).difference(data.columns)

        if missing_columns:
            columns = ", ".join(sorted(missing_columns))
            msg = f"Colunas obrigatórias ausentes: {columns}"
            raise ValueError(msg)

        clean_data = (
            data.dropna(subset=required_columns)
            .drop_duplicates()
            .sort_values(
                ["timestamp", "user_id", "item_id"],
            )
            .reset_index(drop=True)
        )

        invalid_ratings = ~clean_data["rating"].between(1, 5)

        if invalid_ratings.any():
            msg = "Foram encontrados ratings fora do intervalo [1, 5]."
            raise ValueError(msg)

        if clean_data.empty:
            msg = "O dataset ficou vazio após o pré-processamento."
            raise ValueError(msg)

        return clean_data

    @staticmethod
    def _relative_valid_size(
        test_size: float,
        valid_size: float,
    ) -> float:
        """Converte validação total em proporção de treino-validação.

        Args:
            test_size: Proporção total destinada ao teste.
            valid_size: Proporção total destinada à validação.

        Returns:
            Proporção relativa de validação após a retirada do teste.

        Raises:
            ValueError: Se as proporções forem inválidas.
        """
        if not 0.0 < test_size < 1.0:
            msg = "data.test_size deve estar no intervalo (0, 1)."
            raise ValueError(msg)

        if not 0.0 < valid_size < 1.0:
            msg = "data.valid_size deve estar no intervalo (0, 1)."
            raise ValueError(msg)

        if test_size + valid_size >= 1.0:
            msg = "A soma de test_size e valid_size deve ser menor que 1."
            raise ValueError(msg)

        return valid_size / (1.0 - test_size)

    def _save_splits(
        self,
        train: pd.DataFrame,
        valid: pd.DataFrame,
        test: pd.DataFrame,
    ) -> None:
        """Salva as divisões processadas.

        Args:
            train: Dados de treinamento.
            valid: Dados de validação.
            test: Dados de teste.
        """
        output_dir = Path(self.params["data"]["processed_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)

        train.to_csv(
            output_dir / "train.csv",
            index=False,
        )
        valid.to_csv(
            output_dir / "valid.csv",
            index=False,
        )
        test.to_csv(
            output_dir / "test.csv",
            index=False,
        )

    def _save_metadata(
        self,
        metadata: dict[str, int],
    ) -> None:
        """Persiste as dimensões globais usadas pelos embeddings.

        Args:
            metadata: Número global de usuários e itens.
        """
        metadata_path = Path(self.params["data"]["metadata_path"])
        metadata_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        metadata_path.write_text(
            json.dumps(metadata, indent=2),
            encoding="utf-8",
        )

    def _load_metadata(self) -> dict[str, int]:
        """Carrega e valida os metadados do modelo.

        Returns:
            Dicionário contendo número de usuários e itens.

        Raises:
            FileNotFoundError: Se os metadados ainda não existirem.
        """
        metadata_path = Path(self.params["data"]["metadata_path"])

        if not metadata_path.exists():
            msg = (
                f"Metadados não encontrados: {metadata_path}. "
                "Execute primeiro o estágio feature_eng."
            )
            raise FileNotFoundError(msg)

        raw_metadata: dict[str, Any] = json.loads(
            metadata_path.read_text(encoding="utf-8")
        )

        return {
            "num_users": int(raw_metadata["num_users"]),
            "num_items": int(raw_metadata["num_items"]),
        }

    def _load_split(
        self,
        filename: str,
    ) -> pd.DataFrame:
        """Carrega uma divisão processada.

        Args:
            filename: Nome do arquivo da divisão.

        Returns:
            DataFrame da divisão solicitada.

        Raises:
            FileNotFoundError: Se o arquivo não existir.
        """
        split_path = Path(self.params["data"]["processed_dir"]) / filename

        if not split_path.exists():
            msg = (
                f"Divisão processada não encontrada: {split_path}. "
                "Execute primeiro o estágio feature_eng."
            )
            raise FileNotFoundError(msg)

        return pd.read_csv(split_path)

    def _create_model(
        self,
        metadata: dict[str, int],
    ) -> BaseRecommender:
        """Cria o modelo por meio do Factory Pattern.

        Args:
            metadata: Dimensões globais do conjunto de dados.

        Returns:
            Modelo de recomendação configurado.
        """
        return create_model(
            self.params["model"]["name"],
            metadata["num_users"],
            metadata["num_items"],
            int(self.params["model"]["embedding_dim"]),
            int(self.params["model"]["hidden_dim"]),
        )

    def _fit_model(
        self,
        model: BaseRecommender,
        train_data: pd.DataFrame,
        valid_data: pd.DataFrame,
    ) -> dict[str, float]:
        """Treina e restaura o estado com menor RMSE de validação.

        Args:
            model: Modelo que será treinado.
            train_data: Dados de treinamento.
            valid_data: Dados de validação.

        Returns:
            Melhores métricas obtidas durante o treinamento.

        Raises:
            RuntimeError: Se nenhum estado válido for produzido.
        """
        train_loader = self._create_dataloader(
            train_data,
            shuffle=True,
        )
        valid_loader = self._create_dataloader(
            valid_data,
            shuffle=False,
        )

        optimizer = create_optimizer(
            model,
            float(self.params["training"]["learning_rate"]),
        )

        trainer = Trainer(
            model,
            get_loss_function(),
            optimizer,
        )
        evaluator = Evaluator(model)

        tracker = MLflowTracker(self.params["tracking"]["experiment_name"])

        best_valid_rmse = float("inf")
        best_train_loss = float("inf")
        best_epoch = 0
        best_state: dict[str, torch.Tensor] | None = None

        epochs = int(self.params["training"]["epochs"])

        with mlflow.start_run():
            tracker.log_params(self.params["model"] | self.params["training"])

            for epoch in range(1, epochs + 1):
                train_loss = trainer.train_epoch(train_loader)

                valid_rmse = evaluator.evaluate(
                    valid_loader,
                    RMSEStrategy(),
                )

                mlflow.log_metrics(
                    {
                        "train_loss": train_loss,
                        "validation_rmse": valid_rmse,
                    },
                    step=epoch,
                )

                if valid_rmse < best_valid_rmse:
                    best_valid_rmse = valid_rmse
                    best_train_loss = train_loss
                    best_epoch = epoch
                    best_state = copy.deepcopy(model.state_dict())

            if best_state is None:
                msg = "O treinamento não produziu um estado de modelo válido."
                raise RuntimeError(msg)

            model.load_state_dict(best_state)

            tracker.log_metrics(
                {
                    "best_train_loss": best_train_loss,
                    "best_validation_rmse": best_valid_rmse,
                    "best_epoch": float(best_epoch),
                }
            )

            input_example = (
                torch.as_tensor(
                    train_data["user_idx"].head(5).to_numpy(),
                    dtype=torch.long,
                ),
                torch.as_tensor(
                    train_data["item_idx"].head(5).to_numpy(),
                    dtype=torch.long,
                ),
            )

            tracker.log_model(
                model=model,
                artifact_path="model",
                input_example=input_example,
            )

        return {
            "best_train_loss": best_train_loss,
            "best_validation_rmse": best_valid_rmse,
            "best_epoch": float(best_epoch),
        }

    def _calculate_metrics(
        self,
        model: BaseRecommender,
        test_data: pd.DataFrame,
    ) -> dict[str, float]:
        """Calcula métricas finais no conjunto de teste.

        Args:
            model: Modelo treinado.
            test_data: Dados de teste.

        Returns:
            RMSE e MAE calculados no conjunto de teste.
        """
        dataloader = self._create_dataloader(
            test_data,
            shuffle=False,
        )
        evaluator = Evaluator(model)

        return {
            "rmse": evaluator.evaluate(
                dataloader,
                RMSEStrategy(),
            ),
            "mae": evaluator.evaluate(
                dataloader,
                MAEStrategy(),
            ),
        }

    def _create_dataloader(
        self,
        data: pd.DataFrame,
        shuffle: bool,
    ) -> DataLoader[Batch]:
        """Cria um DataLoader do PyTorch.

        Args:
            data: DataFrame usado para criar o dataset.
            shuffle: Se verdadeiro, embaralha os exemplos.

        Returns:
            DataLoader configurado para processamento em lotes.
        """
        dataset = MovieLensDataset(data)

        return DataLoader(
            dataset,
            batch_size=int(self.params["training"]["batch_size"]),
            shuffle=shuffle,
        )
