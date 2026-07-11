# MovieLens Recommender System com uv

Sistema de recomendação baseado no dataset MovieLens 100K usando PyTorch, Scikit-Learn, MLflow, DVC, Docker e uv.

## Objetivo

Construir um pipeline profissional de Machine Learning para recomendação, com:

- código limpo e modular;
- type hints e docstrings;
- padrões de projeto: Factory, Strategy, Template Method e Repository;
- versionamento de dados e pipeline com DVC;
- rastreamento de experimentos com MLflow;
- ambiente reproduzível com uv e `uv.lock`;
- Docker e CI com GitHub Actions.

## Estrutura

```text
movielens-recommender-uv/
├── configs/              # configurações YAML
├── data/                 # dados brutos, intermediários e processados
├── models/               # checkpoints e modelos exportados
├── notebooks/            # exploração e análises
├── reports/              # métricas, figuras e predições
├── scripts/              # pontos de entrada CLI
├── src/recommender/      # pacote principal
├── tests/                # testes automatizados
├── dvc.yaml              # pipeline DVC
├── params.yaml           # parâmetros do experimento
├── pyproject.toml        # dependências e ferramentas do projeto
├── Makefile              # comandos automatizados do projeto
├── docker-compose.yml    # orquestração da aplicação e MLflow
└── Dockerfile            # imagem da aplicação
```

## Requisitos

Instale o `uv`:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Também é necessário ter Docker e Docker Compose instalados para executar o ambiente com MLflow em container.

## Instalação

```bash
uv sync
```

O comando acima cria o ambiente virtual e instala as dependências usando o `uv.lock`.

Depois, caso o lock file tenha sido criado ou atualizado:

```bash
git add uv.lock pyproject.toml
git commit -m "chore: update project dependencies"
```

## Baixar o MovieLens 100K

```bash
uv run python scripts/download_data.py
```

## MLflow

O projeto pode registrar experimentos no MLflow em dois cenários:

| Cenário | Endereço usado |
|---|---|
| Código rodando dentro do Docker | `http://mlflow:5000` |
| Código rodando localmente com `uv` | `http://localhost:5000` |
| Navegador | `http://localhost:5000` |

Para subir somente o servidor MLflow via Docker Compose:

```bash
make docker-mlflow
```

Acesse no navegador:

```text
http://localhost:5000
```

## Executar pipeline localmente com uv

Para rodar localmente com `uv` e registrar os experimentos no servidor MLflow, primeiro suba o MLflow:

```bash
make docker-mlflow
```

Depois execute o pipeline completo:

```bash
make local-pipeline
```

Ou execute cada etapa separadamente:

```bash
make local-prepare
make local-train
make local-evaluate
```

Esses comandos usam internamente:

```bash
MLFLOW_TRACKING_URI=http://localhost:5000
```

Isso evita o erro de tracking URI com esquema `file://` ao registrar modelos no MLflow.

## Executar pipeline dentro do Docker

Para executar todo o pipeline dentro do Docker:

```bash
make docker-pipeline
```

Ou execute cada etapa separadamente:

```bash
make docker-prepare
make docker-train
make docker-evaluate
```

Dentro do Docker, a aplicação usa:

```text
http://mlflow:5000
```

Esse endereço funciona porque `mlflow` é o nome do serviço definido no `docker-compose.yml`.

## Executar pipeline sem Makefile

Também é possível rodar diretamente com `uv`:

```bash
uv run python scripts/run_pipeline.py prepare
uv run python scripts/run_pipeline.py train
uv run python scripts/run_pipeline.py evaluate
```

Porém, para registrar corretamente no servidor MLflow, prefira:

```bash
MLFLOW_TRACKING_URI=http://localhost:5000 uv run python scripts/run_pipeline.py train
```

Ou use os comandos `make local-*`.

## DVC

Para executar o pipeline com DVC:

```bash
uv run dvc repro
```

Ou pelo Makefile:

```bash
make dvc
```

## Testes e qualidade

Execute as etapas individualmente:

```bash
make lint
make type
make test
```

Ou execute a checagem completa:

```bash
make check
```

O comando `make check` roda:

```bash
uv run ruff check src tests
uv run mypy src
uv run pytest
uv run pre-commit run --all-files
```

Para formatar o código:

```bash
make format
```

## Docker

Para subir o ambiente completo:

```bash
make docker-up
```

Para parar os containers:

```bash
make docker-down
```

Para acompanhar os logs:

```bash
make docker-logs
```

## Comandos principais do Makefile

| Comando | Descrição |
|---|---|
| `make install` | Instala dependências com `uv sync` |
| `make lint` | Executa `ruff check` |
| `make format` | Formata o código com `ruff format` |
| `make type` | Executa `mypy` |
| `make test` | Executa os testes com `pytest` |
| `make check` | Executa lint, type check, testes e pre-commit |
| `make docker-mlflow` | Sobe o servidor MLflow |
| `make local-pipeline` | Executa o pipeline local com `uv` usando MLflow em `localhost` |
| `make docker-pipeline` | Executa o pipeline dentro do Docker |
| `make dvc` | Executa o pipeline com DVC |
| `make clean` | Remove caches locais do Python e ferramentas |

## Resultados

Após o treinamento e avaliação, o projeto gera:

```text
models/checkpoints/model.pt
models/checkpoints/model_metadata.json
reports/metrics/train_metrics.json
reports/metrics/evaluation_metrics.json
```

Exemplo de métricas de avaliação:

```python
{
    "rmse": 0.24683344830482035,
    "mae": 0.19536370655996724
}
```

## Status esperado

Após executar o pipeline corretamente, espera-se que:

- o treinamento finalize sem erro;
- a avaliação exiba RMSE e MAE;
- a nova run apareça no MLflow com status `Finished`;
- os parâmetros, métricas e artefatos sejam registrados no MLflow;
- `make check` passe sem erros.
