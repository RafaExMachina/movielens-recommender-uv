"""Funções de perda."""

import torch


def get_loss_function() -> torch.nn.Module:
    """Retorna a função de perda padrão."""
    return torch.nn.MSELoss()
