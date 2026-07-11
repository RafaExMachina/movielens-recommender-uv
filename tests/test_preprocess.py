"""Testes de pré-processamento."""

import pandas as pd

from recommender.data.preprocess import encode_ids, normalize_ratings


def test_encode_ids_creates_columns() -> None:
    """Deve criar índices sequenciais."""
    data = pd.DataFrame({"user_id": [10, 20], "item_id": [100, 200], "rating": [5, 4]})

    result = encode_ids(data)

    assert "user_idx" in result.columns
    assert "item_idx" in result.columns


def test_normalize_ratings_creates_rating_norm() -> None:
    """Deve criar rating normalizado."""
    data = pd.DataFrame({"rating": [1, 3, 5]})

    result = normalize_ratings(data)

    assert result["rating_norm"].between(0, 1).all()
