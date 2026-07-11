"""Divisão dos dados em treino, validação e teste."""

import pandas as pd
from sklearn.model_selection import train_test_split


def split_data(
    data: pd.DataFrame,
    test_size: float,
    valid_size: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Divide os dados em treino, validação e teste.

    Args:
        data (pd.DataFrame): O DataFrame original contendo todos os dados.
        test_size (float): A proporção do conjunto de dados original a ser
            incluída na divisão de teste (ex: 0.2 para 20%).
        valid_size (float): A proporção do conjunto de treino/validação a ser
            separada para validação (ex: 0.25 para 25% do bloco restante).
        seed (int): O valor da semente (seed) para o gerador de números
            aleatórios, garantindo a reprodutibilidade.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Uma tupla contendo os
        DataFrames de treino, validação e teste, respectivamente (train, valid, test).
    """
    train_valid, test = train_test_split(data, test_size=test_size, random_state=seed)
    train, valid = train_test_split(
        train_valid, test_size=valid_size, random_state=seed
    )
    return train, valid, test
