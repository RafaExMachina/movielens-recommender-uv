"""Testes do carregador de dados."""

from pathlib import Path

import pandas as pd

from recommender.data.movielens_loader import load_ratings


def test_load_ratings(tmp_path: Path) -> None:
    """Deve carregar arquivo u.data."""
    file_path = tmp_path / "u.data"
    file_path.write_text("1\t2\t5\t123456\n", encoding="utf-8")

    data = load_ratings(file_path)

    assert isinstance(data, pd.DataFrame)
    assert list(data.columns) == ["user_id", "item_id", "rating", "timestamp"]
