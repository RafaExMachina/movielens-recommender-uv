"""Executa treinamento."""

from recommender.pipeline.training_pipeline import TrainingPipeline


def main() -> None:
    """Treina o modelo."""
    TrainingPipeline(params_path="params.yaml").train()


if __name__ == "__main__":
    main()
