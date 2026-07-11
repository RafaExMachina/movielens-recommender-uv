"""Treinador do modelo."""

import torch
from torch.utils.data import DataLoader

Batch = tuple[torch.Tensor, torch.Tensor, torch.Tensor]


class Trainer:
    """Classe responsável pelo loop de treinamento.

    Attributes:
        model (torch.nn.Module): O modelo de rede neural a ser treinado.
        loss_fn (torch.nn.Module): A função de perda (loss function) utilizada.
        optimizer (torch.optim.Optimizer): O otimizador para atualização dos pesos.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        loss_fn: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer

    def train_epoch(self, dataloader: DataLoader[Batch]) -> float:
        """Treina o modelo por uma época.

        Args:
            dataloader (DataLoader[Batch]): O DataLoader contendo os dados de
                treinamento estruturados em lotes (batches) de (users, items, ratings).

        Returns:
            float: A perda (loss) média acumulada ao longo de todas as iterações
            (batches) desta época.
        """
        self.model.train()
        total_loss = 0.0

        for users, items, ratings in dataloader:
            predictions = self.model(users, items)
            loss = self.loss_fn(predictions, ratings)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(dataloader)
