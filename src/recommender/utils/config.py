"""Leitura de configurações YAML."""

from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Carrega um arquivo YAML.

    Args:
        path: Caminho do arquivo YAML.

    Returns:
        Dicionário com os parâmetros carregados.
    """
    with Path(path).open("r", encoding="utf-8") as file:
        content = yaml.safe_load(file)
    return dict(content)
