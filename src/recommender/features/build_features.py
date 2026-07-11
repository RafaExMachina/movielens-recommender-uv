"""Construção de features para recomendação."""

import pandas as pd

from recommender.data.preprocess import encode_ids, normalize_ratings

RAW_REQUIRED_COLUMNS = {"user_id", "item_id", "rating"}
MODEL_REQUIRED_COLUMNS = {"user_idx", "item_idx"}


def build_model_features(data: pd.DataFrame) -> pd.DataFrame:
    """Constrói as features utilizadas pelo modelo.

    Args:
        data: DataFrame limpo com IDs e avaliações originais.

    Returns:
        DataFrame com IDs codificados, rating normalizado e indicador de interação.

    Raises:
        ValueError: Se alguma coluna obrigatória estiver ausente.
    """
    missing_columns = RAW_REQUIRED_COLUMNS.difference(data.columns)
    if missing_columns:
        columns = ", ".join(sorted(missing_columns))
        msg = f"Colunas obrigatórias ausentes: {columns}"
        raise ValueError(msg)

    features = encode_ids(data)
    features = normalize_ratings(features)
    features["interaction"] = 1
    return features


def build_model_metadata(data: pd.DataFrame) -> dict[str, int]:
    """Calcula dimensões estáveis para os embeddings do modelo.

    Args:
        data: DataFrame com user_idx e item_idx já construídos.

    Returns:
        Quantidade de usuários e itens usada em treino e avaliação.

    Raises:
        ValueError: Se os dados estiverem vazios ou sem as colunas necessárias.
    """
    missing_columns = MODEL_REQUIRED_COLUMNS.difference(data.columns)
    if missing_columns:
        columns = ", ".join(sorted(missing_columns))
        msg = f"Colunas obrigatórias ausentes: {columns}"
        raise ValueError(msg)

    if data.empty:
        msg = "Não é possível gerar metadados a partir de dados vazios."
        raise ValueError(msg)

    return {
        "num_users": int(data["user_idx"].max()) + 1,
        "num_items": int(data["item_idx"].max()) + 1,
    }
