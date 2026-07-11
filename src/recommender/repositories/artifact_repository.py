"""Repository para artefatos do projeto."""

import json
from pathlib import Path
from typing import Any

import torch


class ArtifactRepository:
    """Abstrai leitura e escrita de artefatos."""

    def save_model(self, model: torch.nn.Module, path: str | Path) -> None:
        """Salva pesos do modelo."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), target)

    def save_metrics(self, metrics: dict[str, Any], path: str | Path) -> None:
        """Salva métricas em JSON."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
