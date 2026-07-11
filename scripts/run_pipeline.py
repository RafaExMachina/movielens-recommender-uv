"""CLI para executar etapas do pipeline."""

import argparse

from recommender.pipeline.training_pipeline import TrainingPipeline


def parse_args() -> argparse.Namespace:
    """Lê argumentos de linha de comando."""
    parser = argparse.ArgumentParser(
        description="Executa o pipeline reprodutível do MovieLens."
    )
    parser.add_argument(
        "stage",
        choices=["preprocess", "feature_eng", "train", "evaluate", "all"],
        help="Etapa do pipeline a ser executada.",
    )
    parser.add_argument(
        "--params",
        default="params.yaml",
        help="Caminho para o arquivo de parâmetros.",
    )
    return parser.parse_args()


def main() -> None:
    """Executa a etapa selecionada."""
    args = parse_args()
    pipeline = TrainingPipeline(params_path=args.params)

    if args.stage == "preprocess":
        pipeline.preprocess()
    elif args.stage == "feature_eng":
        pipeline.feature_engineering()
    elif args.stage == "train":
        pipeline.train()
    elif args.stage == "evaluate":
        pipeline.evaluate()
    else:
        pipeline.run()


if __name__ == "__main__":
    main()
