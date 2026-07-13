"""Baselines Scikit-Learn para comparação com o modelo neural."""

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from recommender.evaluation.metrics import (
    add_rating_scale_metrics,
    compute_regression_metrics,
)

FEATURE_COLUMNS = ("user_idx", "item_idx")
TARGET_COLUMN = "rating_norm"

SUPPORTED_BASELINES = (
    "dummy_mean",
    "dummy_median",
    "ridge_one_hot",
)

MetricResults = dict[str, float]
BaselineResults = dict[str, MetricResults]


def _validate_dataset(
    dataset: pd.DataFrame,
    dataset_name: str,
) -> None:
    """Valida um conjunto de dados usado pelos baselines.

    Args:
        dataset: DataFrame a ser validado.
        dataset_name: Nome usado nas mensagens de erro.

    Raises:
        ValueError: Se o conjunto estiver vazio, tiver colunas ausentes
            ou possuir valores não finitos.
    """
    if dataset.empty:
        message = f"O conjunto {dataset_name!r} não pode estar vazio."
        raise ValueError(message)

    required_columns = {*FEATURE_COLUMNS, TARGET_COLUMN}
    missing_columns = required_columns.difference(dataset.columns)

    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        message = (
            f"O conjunto {dataset_name!r} não possui as colunas "
            f"obrigatórias: {missing}."
        )
        raise ValueError(message)

    values = dataset.loc[
        :,
        [*FEATURE_COLUMNS, TARGET_COLUMN],
    ].to_numpy(dtype=np.float64)

    if not np.all(np.isfinite(values)):
        message = f"O conjunto {dataset_name!r} contém valores NaN ou infinitos."
        raise ValueError(message)


def build_baseline_models(
    ridge_alpha: float = 1.0,
) -> dict[str, Any]:
    """Cria os modelos baseline utilizados pelo projeto.

    Args:
        ridge_alpha: Intensidade da regularização L2 do modelo Ridge.

    Returns:
        Dicionário contendo os modelos baseline.

    Raises:
        TypeError: Se ridge_alpha não for numérico.
        ValueError: Se ridge_alpha não for positivo.
    """
    if isinstance(ridge_alpha, bool) or not isinstance(
        ridge_alpha,
        int | float,
    ):
        message = "ridge_alpha deve ser um valor numérico."
        raise TypeError(message)

    if ridge_alpha <= 0.0:
        message = "ridge_alpha deve ser maior que zero."
        raise ValueError(message)

    ridge_pipeline = Pipeline(
        steps=[
            (
                "one_hot_encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=True,
                    dtype=np.float64,
                ),
            ),
            (
                "regressor",
                Ridge(
                    alpha=float(ridge_alpha),
                    solver="lsqr",
                ),
            ),
        ]
    )

    return {
        "dummy_mean": DummyRegressor(strategy="mean"),
        "dummy_median": DummyRegressor(strategy="median"),
        "ridge_one_hot": ridge_pipeline,
    }


def _validate_model_names(
    model_names: Sequence[str],
) -> tuple[str, ...]:
    """Valida os nomes dos baselines solicitados.

    Args:
        model_names: Nomes dos modelos que serão executados.

    Returns:
        Tupla contendo os nomes validados.

    Raises:
        ValueError: Se a sequência estiver vazia, possuir duplicatas
            ou contiver um modelo não suportado.
    """
    selected_models = tuple(model_names)

    if not selected_models:
        message = "É necessário selecionar pelo menos um baseline."
        raise ValueError(message)

    if len(set(selected_models)) != len(selected_models):
        message = "A lista de baselines contém nomes duplicados."
        raise ValueError(message)

    unknown_models = set(selected_models).difference(SUPPORTED_BASELINES)

    if unknown_models:
        unknown = ", ".join(sorted(unknown_models))
        supported = ", ".join(SUPPORTED_BASELINES)
        message = (
            f"Baselines não suportados: {unknown}. Modelos disponíveis: {supported}."
        )
        raise ValueError(message)

    return selected_models


def evaluate_baselines(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    model_names: Sequence[str] | None = None,
    ridge_alpha: float = 1.0,
) -> BaselineResults:
    """Treina e avalia os baselines Scikit-Learn.

    Os modelos são treinados com ``user_idx`` e ``item_idx`` como
    atributos e ``rating_norm`` como alvo.

    Args:
        train_data: Conjunto utilizado para treinamento.
        test_data: Conjunto utilizado para avaliação.
        model_names: Baselines que serão executados. Quando omitido,
            todos os modelos suportados são executados.
        ridge_alpha: Regularização L2 utilizada pelo Ridge.

    Returns:
        Métricas organizadas pelo nome de cada baseline.
    """
    _validate_dataset(train_data, "train")
    _validate_dataset(test_data, "test")

    selected_models = _validate_model_names(
        model_names if model_names is not None else SUPPORTED_BASELINES
    )

    available_models = build_baseline_models(
        ridge_alpha=ridge_alpha,
    )

    x_train = train_data.loc[:, list(FEATURE_COLUMNS)]
    y_train = train_data[TARGET_COLUMN].to_numpy(dtype=np.float64)

    x_test = test_data.loc[:, list(FEATURE_COLUMNS)]
    y_test = test_data[TARGET_COLUMN].to_numpy(dtype=np.float64)

    results: BaselineResults = {}

    for model_name in selected_models:
        model = available_models[model_name]

        model.fit(x_train, y_train)

        predictions = np.asarray(
            model.predict(x_test),
            dtype=np.float64,
        ).reshape(-1)

        normalized_metrics = compute_regression_metrics(
            y_true=y_test,
            y_pred=predictions,
        )

        results[model_name] = add_rating_scale_metrics(
            normalized_metrics,
        )

    return results
