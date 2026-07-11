"""Pré-processamento dos dados MovieLens."""

import pandas as pd
from sklearn.preprocessing import LabelEncoder


def encode_ids(data: pd.DataFrame) -> pd.DataFrame:
    """Codifica IDs de usuário e item em índices sequenciais.

    Args:
        data: DataFrame bruto de avaliações.

    Returns:
        DataFrame com colunas user_idx e item_idx.
    """
    encoded = data.copy()
    encoded["user_idx"] = LabelEncoder().fit_transform(encoded["user_id"])
    encoded["item_idx"] = LabelEncoder().fit_transform(encoded["item_id"])
    return encoded


def normalize_ratings(data: pd.DataFrame) -> pd.DataFrame:
    """Normaliza ratings para o intervalo aproximado [0, 1]."""
    normalized = data.copy()
    normalized["rating_norm"] = (normalized["rating"] - 1.0) / 4.0
    return normalized
