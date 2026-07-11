"""Factory para otimizadores."""

import torch


def create_optimizer(
    model: torch.nn.Module,
    learning_rate: float,
) -> torch.optim.Optimizer:
    """Cria otimizador Adam."""
    return torch.optim.Adam(model.parameters(), lr=learning_rate)
