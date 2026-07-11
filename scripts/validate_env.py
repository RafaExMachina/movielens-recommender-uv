"""Validação do ambiente de execução do projeto."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from recommender.utils.settings import get_settings

EXPECTED_PYTHON = (3, 12)

REQUIRED_FILES = (
    Path("pyproject.toml"),
    Path("uv.lock"),
    Path(".python-version"),
    Path(".env.example"),
)

REQUIRED_MODULES = (
    "dvc",
    "mlflow",
    "numpy",
    "pandas",
    "pydantic_settings",
    "recommender",
    "sklearn",
    "torch",
    "yaml",
)


def validate_python() -> list[str]:
    """Valida a versão principal do Python.

    Returns:
        Lista de erros encontrados.
    """
    current_version = sys.version_info[:2]

    if current_version != EXPECTED_PYTHON:
        expected = ".".join(map(str, EXPECTED_PYTHON))
        current = ".".join(map(str, current_version))

        return [f"Python incompatível: esperado {expected}, encontrado {current}."]

    return []


def validate_files() -> list[str]:
    """Valida os arquivos necessários para reprodução do ambiente.

    Returns:
        Lista de erros encontrados.
    """
    return [
        f"Arquivo obrigatório não encontrado: {path}"
        for path in REQUIRED_FILES
        if not path.is_file()
    ]


def validate_modules() -> list[str]:
    """Valida a disponibilidade das dependências principais.

    Returns:
        Lista de erros encontrados.
    """
    errors: list[str] = []

    for module_name in REQUIRED_MODULES:
        if importlib.util.find_spec(module_name) is None:
            errors.append(f"Módulo não disponível: {module_name}")
        else:
            print(f"[OK] módulo: {module_name}")

    return errors


def validate_settings() -> list[str]:
    """Valida o carregamento das configurações de ambiente.

    Returns:
        Lista de erros encontrados.
    """
    try:
        settings = get_settings()
    except Exception as error:
        return [f"Falha ao carregar configurações: {error}"]

    print(f"[OK] app_env: {settings.app_env}")
    print(f"[OK] log_level: {settings.log_level}")
    print(f"[OK] mlflow_tracking_uri: {settings.mlflow_tracking_uri}")

    return []


def main() -> int:
    """Executa todas as validações do ambiente.

    Returns:
        Código zero em caso de sucesso e um em caso de erro.
    """
    print(f"Python: {sys.version.split()[0]}")
    print(f"Executável: {sys.executable}")
    print()

    errors = [
        *validate_python(),
        *validate_files(),
        *validate_modules(),
        *validate_settings(),
    ]

    if errors:
        print("\nValidação do ambiente falhou:")

        for error in errors:
            print(f"[ERRO] {error}")

        return 1

    print("\nAmbiente validado com sucesso.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
