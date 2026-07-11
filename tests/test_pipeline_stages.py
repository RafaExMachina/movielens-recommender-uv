"""Testes dos estágios de dados do pipeline."""

import json
from pathlib import Path

import pandas as pd
import yaml

from recommender.pipeline.training_pipeline import TrainingPipeline


def _write_params(tmp_path: Path, raw_path: Path) -> Path:
    """Cria parâmetros isolados para os testes."""
    params = {
        "seed": 42,
        "data": {
            "raw_path": str(raw_path),
            "interim_path": str(tmp_path / "interim" / "ratings_clean.csv"),
            "processed_dir": str(tmp_path / "processed"),
            "metadata_path": str(tmp_path / "processed" / "metadata.json"),
            "test_size": 0.2,
            "valid_size": 0.1,
        },
        "model": {
            "name": "mlp",
            "embedding_dim": 8,
            "hidden_dim": 16,
        },
        "training": {
            "epochs": 1,
            "batch_size": 16,
            "learning_rate": 0.001,
        },
        "tracking": {"experiment_name": "test-experiment"},
        "artifacts": {
            "model_path": str(tmp_path / "models" / "model.pt"),
            "train_metrics_path": str(tmp_path / "reports" / "train.json"),
            "evaluation_metrics_path": str(tmp_path / "reports" / "evaluation.json"),
        },
        "evaluation": {"top_k": 10},
    }
    params_path = tmp_path / "params.yaml"
    params_path.write_text(yaml.safe_dump(params), encoding="utf-8")
    return params_path


def _write_raw_data(path: Path, rows: int = 100) -> None:
    """Cria um pequeno arquivo compatível com u.data."""
    lines = [
        f"{index % 10 + 1}\t{index % 20 + 1}\t{index % 5 + 1}\t{1000 + index}"
        for index in range(rows)
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_preprocess_creates_clean_interim_file(tmp_path: Path) -> None:
    """Deve criar automaticamente o diretório e o arquivo intermediário."""
    raw_path = tmp_path / "raw" / "u.data"
    _write_raw_data(raw_path, rows=20)
    params_path = _write_params(tmp_path, raw_path)
    pipeline = TrainingPipeline(params_path)

    pipeline.preprocess()

    interim_path = tmp_path / "interim" / "ratings_clean.csv"
    assert interim_path.exists()
    interim = pd.read_csv(interim_path)
    assert len(interim) == 20
    assert list(interim.columns) == ["user_id", "item_id", "rating", "timestamp"]


def test_feature_engineering_creates_splits_and_metadata(tmp_path: Path) -> None:
    """Deve gerar divisão 70/10/20 e dimensões globais estáveis."""
    raw_path = tmp_path / "raw" / "u.data"
    _write_raw_data(raw_path, rows=100)
    params_path = _write_params(tmp_path, raw_path)
    pipeline = TrainingPipeline(params_path)

    pipeline.preprocess()
    pipeline.feature_engineering()

    processed_dir = tmp_path / "processed"
    train = pd.read_csv(processed_dir / "train.csv")
    valid = pd.read_csv(processed_dir / "valid.csv")
    test = pd.read_csv(processed_dir / "test.csv")
    metadata = json.loads((processed_dir / "metadata.json").read_text())

    assert (len(train), len(valid), len(test)) == (70, 10, 20)
    assert metadata == {"num_users": 10, "num_items": 20}
    assert {"user_idx", "item_idx", "rating_norm", "interaction"}.issubset(
        train.columns
    )
