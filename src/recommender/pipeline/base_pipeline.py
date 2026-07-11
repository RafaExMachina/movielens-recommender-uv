"""Classe base para pipelines."""

from abc import ABC, abstractmethod


class BasePipeline(ABC):
    """Template Method para pipelines de ML."""

    def run(self) -> None:
        """Executa o fluxo padrão completo."""
        self.prepare()
        self.train()
        self.evaluate()

    @abstractmethod
    def prepare(self) -> None:
        """Prepara os dados."""

    @abstractmethod
    def train(self) -> None:
        """Treina o modelo."""

    @abstractmethod
    def evaluate(self) -> None:
        """Avalia o modelo."""
