"""Testes do Template Method do pipeline."""

from recommender.pipeline.base_pipeline import BasePipeline


class DummyPipeline(BasePipeline):
    """Pipeline mínimo para verificar a ordem das etapas."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def preprocess(self) -> None:
        self.calls.append("preprocess")

    def feature_engineering(self) -> None:
        self.calls.append("feature_eng")

    def train(self) -> None:
        self.calls.append("train")

    def evaluate(self) -> None:
        self.calls.append("evaluate")


def test_run_executes_all_stages_in_order() -> None:
    """Deve executar as quatro etapas na ordem do pipeline."""
    pipeline = DummyPipeline()

    pipeline.run()

    assert pipeline.calls == ["preprocess", "feature_eng", "train", "evaluate"]
