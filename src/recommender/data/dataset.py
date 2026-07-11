"""Dataset PyTorch para recomendação."""

import pandas as pd
import torch
from torch.utils.data import Dataset


class MovieLensDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    """Dataset de interações usuário-item.

    Attributes:
        users (torch.Tensor): Tensor contendo os índices mapeados dos usuários.
        items (torch.Tensor): Tensor contendo os índices mapeados dos itens (filmes).
        ratings (torch.Tensor): Tensor contendo as avaliações normalizadas.
    """

    def __init__(self, data: pd.DataFrame) -> None:
        """Inicializa o dataset.

        Args:
            data (pd.DataFrame): DataFrame com as colunas 'user_idx', 'item_idx'
                e 'rating_norm'.
        """
        self.users = torch.tensor(data["user_idx"].values, dtype=torch.long)
        self.items = torch.tensor(data["item_idx"].values, dtype=torch.long)
        self.ratings = torch.tensor(data["rating_norm"].values, dtype=torch.float32)

    def __len__(self) -> int:
        """Retorna o número total de interações no dataset.

        Returns:
            int: A quantidade de amostras (avaliações) disponíveis.
        """
        return len(self.ratings)

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Retorna uma interação específica com base no índice fornecido.

        Args:
            index (int): O índice do elemento a ser recuperado.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Uma tupla contendo três
            tensores unidimensionais correspondentes a (user_id, item_id, rating)
            na posição especificada.
        """
        return self.users[index], self.items[index], self.ratings[index]
