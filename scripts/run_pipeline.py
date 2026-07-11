"""CLI para executar etapas do pipeline."""

import argparse

from recommender.pipeline.training_pipeline import TrainingPipeline


def parse_args() -> argparse.Namespace:
    """Lê argumentos de linha de comando."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage",
        choices=["prepare", "train", "evaluate", "all"],
        help="Etapa do pipeline a ser executada.",
    )
    return parser.parse_args()


def main() -> None:
    """Executa a etapa selecionada."""
    args = parse_args()
    pipeline = TrainingPipeline(params_path="params.yaml")

    if args.stage == "prepare":
        pipeline.prepare()
    elif args.stage == "train":
        pipeline.train()
    elif args.stage == "evaluate":
        pipeline.evaluate()
    else:
        pipeline.run()


if __name__ == "__main__":
    main()
