"""Modelo MLP para recomendação."""

from typing import cast

import torch

from recommender.models.base import BaseRecommender


class MLPRecommender(BaseRecommender):
    """Recomendador baseado em embeddings e MLP."""

    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()

        self.user_embedding = torch.nn.Embedding(num_users, embedding_dim)
        self.item_embedding = torch.nn.Embedding(num_items, embedding_dim)

        self.network = torch.nn.Sequential(
            torch.nn.Linear(embedding_dim * 2, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 1),
        )

    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        """Executa a predição do modelo."""
        user_vector = self.user_embedding(user_ids)
        item_vector = self.item_embedding(item_ids)

        features = torch.cat([user_vector, item_vector], dim=1)
        output = cast(torch.Tensor, self.network(features))

        return output.squeeze(dim=1)
