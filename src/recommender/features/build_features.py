"""Construção de features para recomendação."""

import pandas as pd


def build_interaction_features(data: pd.DataFrame) -> pd.DataFrame:
    """Cria features iniciais de interação usuário-item.

    Args:
        data: DataFrame pré-processado.

    Returns:
        DataFrame com features de interação.
    """
    features = data.copy()
    features["interaction"] = 1
    return features
