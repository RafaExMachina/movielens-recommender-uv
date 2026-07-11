from typing import cast

import torch
from torch import nn


class MLflowRecommenderWrapper(nn.Module):
    """Adapta modelos de recomendação para o formato aceito pelo MLflow.

    Entrada esperada:
        x[:, 0] = user_id
        x[:, 1] = item_id
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2 or x.shape[1] != 2:
            raise ValueError(
                "A entrada deve ter shape (batch_size, 2), "
                "com colunas [user_id, item_id]."
            )

        user_ids = x[:, 0].long()
        item_ids = x[:, 1].long()

        return cast(torch.Tensor, self.model(user_ids, item_ids))
