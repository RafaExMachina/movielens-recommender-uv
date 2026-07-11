"""Factory para criação de modelos."""

from recommender.models.base import BaseRecommender
from recommender.models.matrix_factorization import MatrixFactorization
from recommender.models.mlp_recommender import MLPRecommender


def create_model(
    model_name: str,
    num_users: int,
    num_items: int,
    embedding_dim: int,
    hidden_dim: int,
) -> BaseRecommender:
    """Cria modelos de recomendação a partir de configuração."""
    if model_name == "matrix_factorization":
        return MatrixFactorization(num_users, num_items, embedding_dim)

    if model_name == "mlp":
        return MLPRecommender(num_users, num_items, embedding_dim, hidden_dim)

    msg = f"Modelo não suportado: {model_name}"
    raise ValueError(msg)
