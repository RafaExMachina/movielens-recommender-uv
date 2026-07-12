"""Testes da construção de features."""

import pandas as pd
import pytest

from recommender.features.build_features import (
    build_model_features,
    build_model_metadata,
)


def test_build_model_features_creates_expected_columns() -> None:
    """Deve criar índices, rating normalizado e indicador de interação."""
    data = pd.DataFrame(
        {
            "user_id": [10, 20, 10],
            "item_id": [100, 200, 200],
            "rating": [5, 3, 1],
        }
    )

    result = build_model_features(data)

    assert {"user_idx", "item_idx", "rating_norm", "interaction"}.issubset(
        result.columns
    )
    assert result["rating_norm"].between(0, 1).all()
    assert result["interaction"].eq(1).all()


def test_build_model_metadata_uses_global_dimensions() -> None:
    """Deve persistir dimensões compatíveis com os índices codificados."""
    features = pd.DataFrame(
        {
            "user_idx": [0, 2, 1],
            "item_idx": [3, 0, 1],
        }
    )

    metadata = build_model_metadata(features)

    assert metadata == {"num_users": 3, "num_items": 4}


def test_build_model_features_rejects_missing_columns() -> None:
    """Deve falhar quando os dados não possuem todas as colunas necessárias."""
    data = pd.DataFrame({"user_id": [1], "rating": [5]})

    with pytest.raises(ValueError, match="item_id"):
        build_model_features(data)
