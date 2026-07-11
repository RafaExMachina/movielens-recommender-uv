"""Testes da model factory."""

from recommender.models.model_factory import create_model


def test_create_mlp_model() -> None:
    """Deve criar modelo MLP."""
    model = create_model(
        "mlp", num_users=10, num_items=20, embedding_dim=8, hidden_dim=16
    )

    assert model is not None


def test_create_matrix_factorization_model() -> None:
    """Deve criar modelo de fatoração matricial."""
    model = create_model(
        "matrix_factorization",
        num_users=10,
        num_items=20,
        embedding_dim=8,
        hidden_dim=16,
    )

    assert model is not None
