"""Repository para persistência de artefatos do projeto."""

import json
from pathlib import Path
from typing import Any

import torch


class ArtifactRepository:
    """Abstrai a leitura e a escrita de artefatos."""

    def save_model(
        self,
        model: torch.nn.Module,
        path: str | Path,
    ) -> None:
        """Salva os pesos de um modelo PyTorch.

        Args:
            model: Modelo cujos parâmetros serão persistidos.
            path: Caminho do checkpoint.
        """
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)

        torch.save(model.state_dict(), target)

    def save_json(
        self,
        payload: dict[str, Any],
        path: str | Path,
    ) -> None:
        """Salva um dicionário em formato JSON.

        Args:
            payload: Conteúdo que será persistido.
            path: Caminho do arquivo JSON.
        """
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)

        target.write_text(
            json.dumps(
                payload,
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )

    def save_metrics(
        self,
        metrics: dict[str, Any],
        path: str | Path,
    ) -> None:
        """Salva métricas em formato JSON.

        Args:
            metrics: Métricas que serão persistidas.
            path: Caminho do arquivo JSON.
        """
        self.save_json(metrics, path)
