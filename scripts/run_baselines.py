"""Treina e avalia os baselines Scikit-Learn."""

from pathlib import Path
from typing import Any, cast

import pandas as pd
import yaml

from recommender.evaluation.baselines import (
    SUPPORTED_BASELINES,
    evaluate_baselines,
)
from recommender.repositories.artifact_repository import (
    ArtifactRepository,
)
from recommender.utils.paths import ROOT_DIR

PARAMS_PATH = ROOT_DIR / "params.yaml"


def load_params(
    params_path: Path = PARAMS_PATH,
) -> dict[str, Any]:
    """Carrega os parâmetros do projeto.

    Args:
        params_path: Caminho do arquivo params.yaml.

    Returns:
        Parâmetros carregados.

    Raises:
        FileNotFoundError: Se o arquivo não existir.
        ValueError: Se o conteúdo não for um objeto YAML.
    """
    if not params_path.is_file():
        message = f"Arquivo de parâmetros não encontrado: {params_path}"
        raise FileNotFoundError(message)

    with params_path.open(encoding="utf-8") as params_file:
        payload: Any = yaml.safe_load(params_file)

    if not isinstance(payload, dict):
        message = "O arquivo params.yaml deve conter um objeto YAML."
        raise ValueError(message)

    return cast(dict[str, Any], payload)


def _get_section(
    params: dict[str, Any],
    section_name: str,
) -> dict[str, Any]:
    """Obtém e valida uma seção dos parâmetros.

    Args:
        params: Parâmetros completos.
        section_name: Nome da seção desejada.

    Returns:
        Conteúdo da seção.

    Raises:
        ValueError: Se a seção não existir ou não for um objeto YAML.
    """
    section = params.get(section_name)

    if not isinstance(section, dict):
        message = f"A seção {section_name!r} deve ser um objeto no params.yaml."
        raise ValueError(message)

    return cast(dict[str, Any], section)


def _get_path(
    section: dict[str, Any],
    parameter_name: str,
) -> Path:
    """Obtém um caminho relativo ao diretório do projeto.

    Args:
        section: Seção contendo o parâmetro.
        parameter_name: Nome do parâmetro.

    Returns:
        Caminho absoluto correspondente.

    Raises:
        ValueError: Se o parâmetro não for uma string válida.
    """
    raw_path = section.get(parameter_name)

    if not isinstance(raw_path, str) or not raw_path.strip():
        message = f"O parâmetro {parameter_name!r} deve ser uma string."
        raise ValueError(message)

    return ROOT_DIR / raw_path


def _get_model_names(
    baseline_params: dict[str, Any],
) -> list[str]:
    """Obtém os nomes dos baselines configurados.

    Args:
        baseline_params: Configuração dos baselines.

    Returns:
        Lista de modelos configurados.

    Raises:
        ValueError: Se a configuração não for uma lista de strings.
    """
    raw_names = baseline_params.get(
        "models",
        list(SUPPORTED_BASELINES),
    )

    if not isinstance(raw_names, list) or not all(
        isinstance(name, str) for name in raw_names
    ):
        message = "baselines.models deve ser uma lista de strings."
        raise ValueError(message)

    return cast(list[str], raw_names)


def _get_ridge_alpha(
    baseline_params: dict[str, Any],
) -> float:
    """Obtém a regularização do Ridge.

    Args:
        baseline_params: Configuração dos baselines.

    Returns:
        Valor da regularização.

    Raises:
        ValueError: Se o valor não for numérico.
    """
    raw_alpha = baseline_params.get("ridge_alpha", 1.0)

    if isinstance(raw_alpha, bool) or not isinstance(
        raw_alpha,
        int | float,
    ):
        message = "baselines.ridge_alpha deve ser numérico."
        raise ValueError(message)

    return float(raw_alpha)


def main() -> None:
    """Executa todos os baselines configurados."""
    params = load_params()

    data_params = _get_section(params, "data")
    artifacts_params = _get_section(params, "artifacts")
    baseline_params = _get_section(params, "baselines")

    processed_dir = _get_path(
        data_params,
        "processed_dir",
    )

    output_path = _get_path(
        artifacts_params,
        "baseline_metrics_path",
    )

    train_path = processed_dir / "train.csv"
    test_path = processed_dir / "test.csv"

    if not train_path.is_file():
        message = f"Conjunto de treinamento não encontrado: {train_path}"
        raise FileNotFoundError(message)

    if not test_path.is_file():
        message = f"Conjunto de teste não encontrado: {test_path}"
        raise FileNotFoundError(message)

    model_names = _get_model_names(baseline_params)
    ridge_alpha = _get_ridge_alpha(baseline_params)

    print(f"Carregando treinamento: {train_path}")
    train_data = pd.read_csv(train_path)

    print(f"Carregando teste: {test_path}")
    test_data = pd.read_csv(test_path)

    print("Executando baselines:")
    for model_name in model_names:
        print(f"- {model_name}")

    results = evaluate_baselines(
        train_data=train_data,
        test_data=test_data,
        model_names=model_names,
        ridge_alpha=ridge_alpha,
    )

    metrics_payload: dict[str, Any] = {}
    metrics_payload.update(results)

    repository = ArtifactRepository()
    repository.save_metrics(
        metrics=metrics_payload,
        path=output_path,
    )

    print("\nResultados:")

    for model_name, metrics in results.items():
        print(
            f"{model_name}: "
            f"RMSE={metrics['rmse']:.5f}, "
            f"MAE={metrics['mae']:.5f}, "
            f"MSE={metrics['mse']:.5f}, "
            f"R²={metrics['r2']:.5f}"
        )

    print(f"\nMétricas salvas em: {output_path}")


if __name__ == "__main__":
    main()
