"""Configuração de logger."""

import logging


def get_logger(name: str) -> logging.Logger:
    """Retorna um logger padrão.

    Args:
        name: Nome do logger.

    Returns:
        Logger configurado.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    return logging.getLogger(name)
