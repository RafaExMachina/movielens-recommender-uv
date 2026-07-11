"""Classe base para pipelines."""

from abc import ABC, abstractmethod


class BasePipeline(ABC):
    """Template Method para pipelines de Machine Learning."""

    def run(self) -> None:
        """Executa o pipeline completo na ordem definida."""
        self.preprocess()
        self.feature_engineering()
        self.train()
        self.evaluate()

    @abstractmethod
    def preprocess(self) -> None:
        """Limpa e persiste os dados intermediários."""

    @abstractmethod
    def feature_engineering(self) -> None:
        """Constrói features e divisões de treino, validação e teste."""

    @abstractmethod
    def train(self) -> None:
        """Treina o modelo."""

    @abstractmethod
    def evaluate(self) -> None:
        """Avalia o modelo."""
