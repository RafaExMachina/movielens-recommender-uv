"""Testes do pipeline de treinamento."""

from pathlib import Path

from recommender.pipeline.training_pipeline import TrainingPipeline


def test_training_pipeline_loads_params() -> None:
    """Deve inicializar pipeline com params.yaml."""
    params_path = Path("params.yaml")

    pipeline = TrainingPipeline(params_path=str(params_path))

    assert pipeline.params["seed"] == 42
