"""Executa avaliação."""

from recommender.pipeline.training_pipeline import TrainingPipeline


def main() -> None:
    """Avalia o modelo."""
    TrainingPipeline(params_path="params.yaml").evaluate()


if __name__ == "__main__":
    main()
