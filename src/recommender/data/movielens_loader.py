"""Carregamento do dataset MovieLens."""

from pathlib import Path

import pandas as pd

RATING_COLUMNS = ["user_id", "item_id", "rating", "timestamp"]


def load_ratings(path: str | Path) -> pd.DataFrame:
    """Carrega o arquivo u.data do MovieLens 100K.

    Args:
        path: Caminho para o arquivo u.data.

    Returns:
        DataFrame com user_id, item_id, rating e timestamp.
    """
    file_path = Path(path)
    if not file_path.exists():
        msg = f"Arquivo não encontrado: {file_path}. Execute scripts/download_data.py."
        raise FileNotFoundError(msg)

    return pd.read_csv(file_path, sep="\t", names=RATING_COLUMNS, engine="python")
