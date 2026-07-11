"""Callbacks de treinamento."""


class EarlyStopping:
    """Early stopping simples."""

    def __init__(self, patience: int) -> None:
        """Inicializa callback."""
        self.patience = patience
        self.best_loss = float("inf")
        self.counter = 0

    def should_stop(self, loss: float) -> bool:
        """Indica se o treinamento deve parar."""
        if loss < self.best_loss:
            self.best_loss = loss
            self.counter = 0
            return False

        self.counter += 1
        return self.counter >= self.patience
