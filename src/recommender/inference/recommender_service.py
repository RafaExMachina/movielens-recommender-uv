"""Serviço de recomendação."""

import torch


class RecommenderService:
    """Serviço para gerar recomendações."""

    def __init__(self, model: torch.nn.Module) -> None:
        """Inicializa serviço."""
        self.model = model

    def recommend(self, user_id: int, item_ids: list[int]) -> list[tuple[int, float]]:
        """Recomenda itens para um usuário."""
        users = torch.tensor([user_id] * len(item_ids), dtype=torch.long)
        items = torch.tensor(item_ids, dtype=torch.long)

        with torch.no_grad():
            scores = self.model(users, items)

        pairs = zip(item_ids, scores.tolist(), strict=False)
        return sorted(pairs, key=lambda pair: pair[1], reverse=True)
