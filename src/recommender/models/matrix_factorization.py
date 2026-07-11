"""Modelo de fatoração matricial neural."""

import torch

from recommender.models.base import BaseRecommender


class MatrixFactorization(BaseRecommender):
    """Modelo baseado em embeddings de usuários e itens."""

    def __init__(self, num_users: int, num_items: int, embedding_dim: int) -> None:
        """Inicializa embeddings."""
        super().__init__()
        self.user_embedding = torch.nn.Embedding(num_users, embedding_dim)
        self.item_embedding = torch.nn.Embedding(num_items, embedding_dim)

    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        """Prediz rating normalizado."""
        user_vector = self.user_embedding(user_ids)
        item_vector = self.item_embedding(item_ids)
        return torch.sigmoid((user_vector * item_vector).sum(dim=1))
