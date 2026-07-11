"""Baixa e extrai o dataset MovieLens 100K."""

from pathlib import Path
from urllib.request import urlretrieve
from zipfile import ZipFile

DATA_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
ZIP_PATH = Path("data/raw/ml-100k.zip")
RAW_DIR = Path("data/raw")
TARGET_FILE = Path("data/raw/ml-100k/u.data")


def download_file() -> None:
    """Baixa o arquivo ZIP do MovieLens 100K."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Baixando dataset de {DATA_URL}")
    urlretrieve(DATA_URL, ZIP_PATH)  # noqa: S310


def extract_zip() -> None:
    """Extrai o arquivo ZIP baixado."""
    with ZipFile(ZIP_PATH) as zip_file:
        zip_file.extractall(RAW_DIR)


def main() -> None:
    """Executa download e extração."""
    if TARGET_FILE.exists():
        print(f"Dataset já existe em {TARGET_FILE}")
        return

    download_file()
    extract_zip()
    print(f"Dataset extraído em {TARGET_FILE.parent}")


if __name__ == "__main__":
    main()
