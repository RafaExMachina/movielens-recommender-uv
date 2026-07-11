"""Interface base para modelos de recomendação."""

from abc import ABC, abstractmethod

import torch


class BaseRecommender(torch.nn.Module, ABC):
    """Classe base para recomendadores."""

    @abstractmethod
    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        """Executa a predição do modelo."""
