"""Configuração de logger."""

import logging

from recommender.utils.settings import get_settings


def get_logger(name: str) -> logging.Logger:
    """Retorna um logger configurado para o ambiente atual.

    Args:
        name: Nome do logger.

    Returns:
        Logger configurado.
    """
    settings = get_settings()

    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    return logging.getLogger(name)
