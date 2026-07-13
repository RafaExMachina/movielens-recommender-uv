"""Testes dos baselines Scikit-Learn."""

import math

import pandas as pd
import pytest

from recommender.evaluation.baselines import (
    SUPPORTED_BASELINES,
    build_baseline_models,
    evaluate_baselines,
)


def _training_data() -> pd.DataFrame:
    """Cria um pequeno conjunto de treinamento."""
    return pd.DataFrame(
        {
            "user_idx": [0, 0, 1, 1, 2, 2],
            "item_idx": [0, 1, 0, 2, 1, 2],
            "rating_norm": [0.0, 0.25, 0.5, 0.75, 1.0, 0.5],
        }
    )


def _test_data() -> pd.DataFrame:
    """Cria um pequeno conjunto de teste."""
    return pd.DataFrame(
        {
            "user_idx": [0, 1, 2],
            "item_idx": [2, 1, 0],
            "rating_norm": [0.25, 0.75, 0.5],
        }
    )


def test_build_baseline_models_returns_expected_models() -> None:
    """A factory deve criar todos os baselines esperados."""
    models = build_baseline_models()

    assert set(models) == set(SUPPORTED_BASELINES)


def test_evaluate_baselines_returns_all_models() -> None:
    """A avaliação deve produzir resultados para todos os baselines."""
    results = evaluate_baselines(
        train_data=_training_data(),
        test_data=_test_data(),
    )

    assert set(results) == set(SUPPORTED_BASELINES)


def test_evaluate_baselines_returns_all_metrics() -> None:
    """Cada baseline deve possuir todas as métricas esperadas."""
    results = evaluate_baselines(
        train_data=_training_data(),
        test_data=_test_data(),
    )

    expected_metrics = {
        "rmse",
        "mae",
        "mse",
        "r2",
        "median_absolute_error",
        "rmse_rating_scale",
        "mae_rating_scale",
    }

    for metrics in results.values():
        assert set(metrics) == expected_metrics
        assert all(math.isfinite(value) for value in metrics.values())


def test_dummy_mean_uses_training_mean() -> None:
    """O baseline da média deve prever a média do conjunto de treino."""
    train_data = pd.DataFrame(
        {
            "user_idx": [0, 1, 2, 3],
            "item_idx": [0, 1, 2, 3],
            "rating_norm": [0.0, 0.0, 1.0, 1.0],
        }
    )

    test_data = pd.DataFrame(
        {
            "user_idx": [0, 1],
            "item_idx": [1, 0],
            "rating_norm": [0.25, 0.75],
        }
    )

    results = evaluate_baselines(
        train_data=train_data,
        test_data=test_data,
        model_names=["dummy_mean"],
    )

    assert results["dummy_mean"]["rmse"] == pytest.approx(0.25)
    assert results["dummy_mean"]["mae"] == pytest.approx(0.25)
    assert results["dummy_mean"]["rmse_rating_scale"] == pytest.approx(1.0)


def test_ridge_handles_unknown_users_and_items() -> None:
    """O Ridge deve aceitar categorias não observadas no treino."""
    test_data = pd.DataFrame(
        {
            "user_idx": [99],
            "item_idx": [99],
            "rating_norm": [0.5],
        }
    )

    results = evaluate_baselines(
        train_data=_training_data(),
        test_data=test_data,
        model_names=["ridge_one_hot"],
    )

    metrics = results["ridge_one_hot"]

    assert all(math.isfinite(value) for value in metrics.values())


def test_evaluate_baselines_accepts_model_subset() -> None:
    """A execução deve aceitar apenas os modelos selecionados."""
    results = evaluate_baselines(
        train_data=_training_data(),
        test_data=_test_data(),
        model_names=["dummy_mean", "ridge_one_hot"],
    )

    assert set(results) == {
        "dummy_mean",
        "ridge_one_hot",
    }


def test_evaluate_baselines_rejects_unknown_model() -> None:
    """Um modelo não suportado deve ser rejeitado."""
    with pytest.raises(ValueError, match="não suportados"):
        evaluate_baselines(
            train_data=_training_data(),
            test_data=_test_data(),
            model_names=["random_model"],
        )


def test_evaluate_baselines_rejects_missing_column() -> None:
    """Uma coluna obrigatória ausente deve gerar erro."""
    invalid_data = _training_data().drop(columns=["rating_norm"])

    with pytest.raises(ValueError, match="rating_norm"):
        evaluate_baselines(
            train_data=invalid_data,
            test_data=_test_data(),
        )


def test_evaluate_baselines_rejects_empty_data() -> None:
    """Um conjunto vazio deve ser rejeitado."""
    empty_data = pd.DataFrame(columns=["user_idx", "item_idx", "rating_norm"])

    with pytest.raises(ValueError, match="não pode estar vazio"):
        evaluate_baselines(
            train_data=empty_data,
            test_data=_test_data(),
        )


def test_build_baseline_models_rejects_invalid_alpha() -> None:
    """A regularização Ridge deve ser positiva."""
    with pytest.raises(
        ValueError,
        match="maior que zero",
    ):
        build_baseline_models(ridge_alpha=0.0)
