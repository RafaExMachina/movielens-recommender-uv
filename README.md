# MovieLens Recommender System com uv

Sistema de recomendação baseado no dataset **MovieLens 100K**, estruturado como um projeto profissional de Machine Learning com código modular, ambiente reproduzível, versionamento de dados, rastreamento de experimentos e execução containerizada.

Versão atual:

```text
0.3.1
```

O repositório contém as seguintes entregas:

- **Etapa 1 — Clean Code e Estrutura**
- **Etapa 2 — Ambiente e Dependências**
- **Etapa 3 — DVC, Docker e MLflow**

## Objetivo

Construir um pipeline profissional de Machine Learning para recomendação, com:

- código limpo, modular e testável;
- gerenciamento reproduzível de dependências com `uv`;
- versionamento de dados e artefatos com DVC;
- pipeline composto por etapas independentes;
- rastreamento de experimentos com MLflow;
- treinamento e avaliação com PyTorch;
- execução reproduzível com Docker e Docker Compose;
- verificações automatizadas de qualidade;
- integração contínua com GitHub Actions.

## Etapas concluídas

### Etapa 1 — Clean Code e Estrutura

Foram implementados:

- estrutura modular com layout `src`;
- módulos com responsabilidades bem definidas;
- type hints nas funções públicas;
- docstrings no padrão Google;
- princípios SOLID;
- padrões Factory, Strategy, Template Method e Repository;
- testes automatizados;
- lint, formatação e verificação de tipos;
- hooks com pre-commit;
- integração contínua com GitHub Actions;
- histórico de commits semântico.

### Etapa 2 — Ambiente e Dependências

Foram implementados:

- gerenciamento de dependências com `uv`;
- separação entre dependências de produção e desenvolvimento;
- lockfile versionado com `uv.lock`;
- versão do Python declarada em `.python-version`;
- empacotamento com Hatchling;
- instalação reproduzível com `uv sync --locked`;
- build de distribuição em wheel e source distribution;
- PyTorch CPU como instalação padrão;
- configurações externas com Pydantic Settings;
- arquivo `.env.example`;
- integração das configurações com logging e MLflow;
- script automatizado de validação do ambiente;
- testes de configurações e validações;
- instalação validada em ambiente virtual recriado do zero.

### Etapa 3 — DVC, Docker e MLflow

Foram implementados:

- download reproduzível do dataset público MovieLens 100K;
- remote DVC local para dados derivados, modelos e demais artefatos cacheados;
- pipeline com cinco estágios:
  - `download`;
  - `preprocess`;
  - `feature_eng`;
  - `train`;
  - `evaluate`;
- dependências, parâmetros, métricas e outputs declarados em `dvc.yaml`;
- estado reproduzível registrado em `dvc.lock`;
- métricas de treinamento e avaliação rastreadas pelo DVC;
- execução incremental com `dvc repro`;
- execução completa com `dvc repro --force --no-run-cache`;
- rastreamento de experimentos com MLflow;
- servidor MLflow 2.14.0 com backend SQLite;
- persistência do MLflow em volume Docker;
- Dockerfile multi-stage;
- execução do pipeline com usuário não-root;
- orquestração com Docker Compose;
- healthcheck do servidor MLflow;
- execução de `dvc repro`, `dvc push`, métricas e status dentro do contêiner;
- validação do pipeline no Docker Compose;
- validação da persistência dos experimentos após reiniciar os serviços.

## Tecnologias

- Python 3.12
- uv
- PyTorch CPU
- Pandas
- NumPy
- Scikit-Learn
- MLflow 2.14.0
- DVC
- Docker
- Docker Compose
- PyYAML
- Pydantic Settings
- Pytest
- Ruff
- mypy
- pre-commit
- Hatchling
- GitHub Actions

## Estrutura do projeto

```text
movielens-recommender-uv/
├── .dvc/
│   └── config
├── .github/
│   └── workflows/
│       └── ci.yml
├── configs/
│   ├── config.yaml
│   ├── logging.yaml
│   └── model_config.yaml
├── data/
│   ├── features/
│   ├── interim/
│   ├── processed/
│   └── raw/
│       └── ml-100k/              # gerado pelo estágio download
├── models/
│   ├── checkpoints/
│   ├── exported/
│   └── registry/
├── notebooks/
├── reports/
│   ├── figures/
│   ├── metrics/
│   │   ├── evaluation_metrics.json
│   │   └── train_metrics.json
│   └── predictions/
├── scripts/
│   ├── download_data.py
│   ├── evaluate.py
│   ├── predict.py
│   ├── run_pipeline.py
│   ├── train.py
│   └── validate_env.py
├── src/
│   └── recommender/
│       ├── data/
│       ├── evaluation/
│       ├── features/
│       ├── inference/
│       ├── models/
│       ├── pipeline/
│       ├── repositories/
│       ├── tracking/
│       ├── training/
│       └── utils/
├── tests/
│   ├── test_base_pipeline.py
│   ├── test_data_loader.py
│   ├── test_features.py
│   ├── test_metrics.py
│   ├── test_model_factory.py
│   ├── test_pipeline_stages.py
│   ├── test_preprocess.py
│   ├── test_settings.py
│   └── test_training_pipeline.py
├── .dockerignore
├── .env.example
├── .gitignore
├── .pre-commit-config.yaml
├── .python-version
├── Dockerfile
├── Makefile
├── docker-compose.yml
├── dvc.lock
├── dvc.yaml
├── params.yaml
├── pyproject.toml
├── README.md
└── uv.lock
```

Os diretórios de dados, modelos e relatórios utilizam arquivos `.gitkeep` quando necessário para preservar a estrutura sem versionar datasets, checkpoints ou artefatos pesados diretamente no Git.

## Arquitetura

O código principal está localizado em:

```text
src/recommender/
```

A aplicação está dividida por responsabilidade:

| Diretório | Responsabilidade |
|---|---|
| `data/` | carregamento, preparação e divisão dos dados |
| `features/` | construção e transformação de atributos |
| `models/` | modelos e criação de instâncias |
| `training/` | treinamento, perdas, callbacks e otimizadores |
| `evaluation/` | métricas e estratégias de avaliação |
| `inference/` | predição e serviço de recomendação |
| `pipeline/` | organização dos fluxos de treinamento e inferência |
| `repositories/` | persistência e recuperação de artefatos |
| `tracking/` | integração e abstrações para MLflow |
| `utils/` | configurações, logging, caminhos e reprodutibilidade |

## Padrões de projeto

### Factory

Centraliza a criação dos modelos de recomendação:

```text
src/recommender/models/model_factory.py
```

A Factory reduz o acoplamento entre o pipeline e as implementações concretas dos modelos.

### Strategy

Permite diferentes estratégias de avaliação:

```text
src/recommender/evaluation/metric_strategy.py
```

Novas métricas podem ser adicionadas sem modificar o fluxo principal.

### Template Method

Define a estrutura geral dos pipelines:

```text
src/recommender/pipeline/base_pipeline.py
```

A classe base define a sequência de execução, enquanto as subclasses implementam etapas específicas.

### Repository

Abstrai o armazenamento e a recuperação de artefatos:

```text
src/recommender/repositories/artifact_repository.py
```

Essa separação evita que o código de domínio dependa diretamente do sistema de arquivos.

## Requisitos

### Execução local

É necessário ter:

- Git;
- Python 3.12;
- uv.

Para instalar o `uv` no Linux:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Confirme a instalação:

```bash
uv --version
```

### Execução containerizada

É necessário ter:

- Docker;
- Docker Compose;
- Git.

Confirme:

```bash
docker --version
docker compose version
```

## Clonar o repositório

```bash
git clone https://github.com/RafaExMachina/movielens-recommender-uv.git
cd movielens-recommender-uv
```

## Instalação reproduzível

Instale o projeto e as dependências registradas no lockfile:

```bash
uv sync --locked
```

O comando:

- cria o ambiente virtual em `.venv`;
- instala o pacote `movielens-recommender`;
- instala as dependências de produção e desenvolvimento;
- utiliza as versões registradas no `uv.lock`;
- falha caso `pyproject.toml` e `uv.lock` estejam dessincronizados.

Não é necessário ativar manualmente o ambiente virtual. Os comandos podem ser executados com `uv run`.

Para instalar apenas as dependências de produção:

```bash
uv sync --locked --no-dev
```

## PyTorch CPU

O projeto utiliza o build CPU do PyTorch como padrão:

```toml
[tool.uv.sources]
torch = { index = "pytorch-cpu" }

[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cpu"
explicit = true
```

Essa configuração evita a instalação de bibliotecas CUDA e NVIDIA desnecessárias para execução em CPU.

Verifique:

```bash
uv run python -c \
'import torch; print("PyTorch:", torch.__version__); print("CUDA:", torch.cuda.is_available())'
```

Em uma instalação CPU, espera-se:

```text
CUDA: False
```

## Configurações de ambiente

As configurações específicas de ambiente são carregadas por:

```text
src/recommender/utils/settings.py
```

O carregamento e a validação são realizados com Pydantic Settings.

| Variável | Descrição | Valor padrão |
|---|---|---|
| `APP_ENV` | ambiente da aplicação | `development` |
| `LOG_LEVEL` | nível de logging | `INFO` |
| `MLFLOW_TRACKING_URI` | endereço do MLflow | `http://localhost:5000` |

Valores aceitos por `APP_ENV`:

```text
development
test
production
```

Crie um arquivo `.env` local:

```bash
cp .env.example .env
```

Exemplo:

```dotenv
APP_ENV=development
LOG_LEVEL=INFO
MLFLOW_TRACKING_URI=http://localhost:5000
```

O arquivo `.env` não deve ser versionado. Apenas `.env.example` permanece no repositório.

Variáveis do sistema operacional possuem precedência sobre o arquivo `.env`.

## Parâmetros de treinamento

Os parâmetros de dados, treinamento e avaliação estão em:

```text
params.yaml
```

Entre eles:

- seed;
- caminhos dos dados;
- proporções de treino, validação e teste;
- dimensão dos embeddings;
- número de épocas;
- batch size;
- learning rate;
- top-k da avaliação;
- nome do experimento MLflow.

Essa separação evita misturar configurações de infraestrutura com hiperparâmetros do modelo.

## Versionamento de dados com DVC

O dataset MovieLens 100K é público e é obtido diretamente da fonte oficial pelo estágio `download` do pipeline DVC:

```text
scripts/download_data.py
```

O diretório gerado é:

```text
data/raw/ml-100k/
```

Esse diretório não é versionado pelo Git e não depende de um arquivo `data/raw/ml-100k.dvc`. Quando o dataset não existe, o comando abaixo executa o download automaticamente:

```bash
uv run dvc repro
```

Os dados derivados, modelos e demais outputs cacheados continuam sendo controlados pelo DVC.

Envie os artefatos disponíveis no cache para o remote:

```bash
uv run dvc push
```

Opcionalmente, recupere artefatos cacheados de uma execução anterior:

```bash
uv run dvc pull
```

O `dvc pull` não é necessário para obter o dataset bruto, pois ele é recriado pelo estágio `download`.

Verifique o estado:

```bash
uv run dvc status
```

Resultado esperado quando tudo estiver atualizado:

```text
Data and pipelines are up to date.
```

### Remote DVC local

O projeto utiliza um remote local localizado fora do repositório:

```text
../movielens-dvc-storage
```

Esse diretório não é enviado ao GitHub. Ele é utilizado para armazenar outputs cacheados do DVC, como dados processados e modelos.

Crie o diretório quando necessário:

```bash
mkdir -p ../movielens-dvc-storage
```

## Pipeline DVC

O pipeline é definido em:

```text
dvc.yaml
```

A sequência é:

```text
download
   ↓
preprocess
   ↓
feature_eng
   ↓
train
   ↓
evaluate
```

### Estágio `download`

Responsável por:

- baixar o MovieLens 100K da fonte oficial;
- extrair o conteúdo em `data/raw/ml-100k/`;
- permitir a reconstrução do dataset sem depender de um remote DVC;
- evitar download desnecessário quando o dataset já existe.

### Estágio `preprocess`

Responsável por:

- carregar os dados brutos;
- limpar e validar os registros;
- gerar os dados intermediários.

### Estágio `feature_eng`

Responsável por:

- construir os conjuntos de treino, validação e teste;
- gerar metadados utilizados pelo modelo.

### Estágio `train`

Responsável por:

- construir o modelo;
- treinar com PyTorch;
- registrar parâmetros, métricas e artefatos no MLflow;
- salvar o checkpoint;
- gerar métricas de treinamento.

### Estágio `evaluate`

Responsável por:

- carregar o checkpoint;
- avaliar o modelo no conjunto de teste;
- calcular RMSE e MAE;
- gerar as métricas finais.

Execute o pipeline:

```bash
uv run dvc repro
```

Quando dados, código, parâmetros e outputs não mudaram, o DVC pula os estágios atualizados.

Force a execução completa sem reutilizar o cache de execuções anteriores:

```bash
uv run dvc repro --force --no-run-cache
```

Visualize as métricas:

```bash
uv run dvc metrics show
```

Exemplo:

```text
Path                                     best_epoch    best_train_loss    best_validation_rmse    mae      rmse
reports/metrics/train_metrics.json       5.0           0.05233            0.24592                 -        -
reports/metrics/evaluation_metrics.json  -             -                  -                       0.19719  0.24712
```

Visualize o grafo:

```bash
uv run dvc dag
```

## MLflow

O treinamento registra experimentos no MLflow.

O nome padrão do experimento é:

```text
movielens-recommender
```

São registrados, entre outros:

- hiperparâmetros;
- perdas de treinamento;
- RMSE de validação;
- melhor época;
- modelo e artefatos relacionados.

Na execução com Docker Compose, a interface fica disponível em:

```text
http://localhost:5000
```

O servidor utiliza:

- MLflow 2.14.0;
- SQLite como backend;
- volume `mlflow_data` para persistência;
- healthcheck antes da inicialização do pipeline.

Cada execução real do estágio `train` cria um novo run. Quando o DVC identifica que o estágio está atualizado e o pula, nenhum novo run é criado.

## Execução com Docker

### 1. Definir UID e GID

No Linux:

```bash
export LOCAL_UID="$(id -u)"
export LOCAL_GID="$(id -g)"
```

Esses valores permitem que os arquivos produzidos no contêiner pertençam ao usuário local.

### 2. Construir a imagem

```bash
docker compose build pipeline
```

O Dockerfile utiliza build multi-stage e instala as dependências a partir de:

```text
pyproject.toml
uv.lock
```

### 3. Iniciar o MLflow

```bash
docker compose up -d mlflow
```

Verifique:

```bash
docker compose ps
```

O serviço deve aparecer como:

```text
healthy
```

A interface estará disponível em:

```text
http://localhost:5000
```

### 4. Executar o pipeline

```bash
docker compose run --rm pipeline
```

O serviço executa:

1. configuração do DVC em modo `no_scm`;
2. `dvc repro`;
3. `dvc push`;
4. `dvc metrics show`;
5. `dvc status`.

O estágio `download` obtém automaticamente o MovieLens 100K quando o dataset bruto ainda não existe.

A execução normal é incremental. Sem alterações, os estágios são pulados.

### 5. Forçar todos os estágios

```bash
docker compose run --rm pipeline \
  sh -eu -c '
    printf "%s\n" \
      "[core]" \
      "    no_scm = true" \
      > /app/.dvc/config.local

    uv run dvc repro --force --no-run-cache
    uv run dvc push
    uv run dvc metrics show
    uv run dvc status
  '
```

Esse comando executa novamente `download`, `preprocess`, `feature_eng`, `train` e `evaluate`, criando um novo run no MLflow.

### 6. Encerrar os serviços

```bash
docker compose down
```

Os experimentos permanecem disponíveis porque o volume `mlflow_data` é preservado.

Para apagar também os volumes:

```bash
docker compose down -v
```

> Esse último comando remove o banco e os artefatos persistidos pelo MLflow.

## Validação do ambiente

Execute:

```bash
uv run python scripts/validate_env.py
```

O script verifica:

- versão do Python;
- arquivos necessários;
- instalação das dependências;
- instalação do pacote;
- carregamento do Pydantic Settings;
- ambiente atual;
- nível de log;
- URI do MLflow.

Resultado esperado:

```text
Ambiente validado com sucesso.
```

## Executar os testes

```bash
uv run pytest
```

Ou:

```bash
make test
```

A versão `0.3.1` possui **16 testes automatizados**.

## Verificação de qualidade

### Ruff

```bash
uv run ruff check src tests scripts
```

Ou:

```bash
make lint
```

### mypy

```bash
uv run mypy src
```

Ou:

```bash
make type
```

### Pre-commit

```bash
uv run pre-commit run --all-files
```

### Verificação completa

```bash
make check
```

Também pode ser executada manualmente:

```bash
uv run ruff check src tests scripts
uv run mypy src
uv run pytest
uv run pre-commit run --all-files
```

## Formatação

```bash
uv run ruff format src tests scripts
```

Ou:

```bash
make format
```

## Build do pacote

Gere o source distribution e o wheel:

```bash
uv build
```

Arquivos esperados:

```text
dist/movielens_recommender-0.3.1.tar.gz
dist/movielens_recommender-0.3.1-py3-none-any.whl
```

Confira a versão instalada:

```bash
uv run python -c \
'from importlib.metadata import version; print(version("movielens-recommender"))'
```

Resultado esperado:

```text
0.3.1
```

## Integração contínua

O workflow está em:

```text
.github/workflows/ci.yml
```

A integração contínua é executada após pushes e pull requests.

As verificações incluem:

- instalação das dependências;
- lint com Ruff;
- verificação de tipos com mypy;
- testes com Pytest;
- hooks do pre-commit.

## Arquivos não versionados

O `.gitignore` impede o envio de arquivos locais ou gerados, incluindo:

```text
.venv/
.env
__pycache__/
.pytest_cache/
.mypy_cache/
.ruff_cache/
dist/
mlruns/
mlruns-local/
mlartifacts/
mlflow.db
.dvc/cache/
.dvc/tmp/
data/raw/ml-100k/
data/interim/*
data/processed/*
data/features/*
models/checkpoints/*
models/exported/*
models/registry/*
reports/figures/*
reports/predictions/*
```

São versionados:

- `dvc.yaml`;
- `dvc.lock`;
- `params.yaml`;
- `scripts/download_data.py`;
- métricas JSON em `reports/metrics/`;
- arquivos `.gitkeep` necessários.

## Comandos principais

| Comando | Descrição |
|---|---|
| `uv sync --locked` | instala o ambiente usando o lockfile |
| `uv sync --locked --no-dev` | instala apenas dependências de produção |
| `uv lock --check` | verifica se o lockfile está atualizado |
| `uv run python scripts/validate_env.py` | valida o ambiente |
| `uv run dvc pull` | recupera artefatos cacheados do remote, de forma opcional |
| `uv run dvc repro` | baixa o dataset quando necessário e executa o pipeline incremental |
| `uv run dvc repro --force --no-run-cache` | força todos os estágios sem reutilizar o cache de execução |
| `uv run dvc push` | envia dados e artefatos ao remote |
| `uv run dvc metrics show` | exibe métricas |
| `uv run dvc status` | verifica o estado do pipeline |
| `docker compose build pipeline` | constrói a imagem do pipeline |
| `docker compose up -d mlflow` | inicia o MLflow |
| `docker compose run --rm pipeline` | executa o pipeline no Docker |
| `docker compose down` | encerra os serviços |
| `uv run pytest` | executa os testes |
| `make lint` | executa Ruff |
| `make format` | formata o código |
| `make type` | executa mypy |
| `make test` | executa Pytest |
| `make check` | executa todas as verificações |
| `uv build` | gera wheel e source distribution |

## Reproduzir a entrega localmente

```bash
git clone https://github.com/RafaExMachina/movielens-recommender-uv.git
cd movielens-recommender-uv

uv sync --locked
uv run python scripts/validate_env.py
mkdir -p ../movielens-dvc-storage
uv run dvc repro
uv run dvc metrics show
make check
uv build
```

O dataset bruto é baixado automaticamente pelo estágio `download`. O remote DVC local é utilizado para armazenar e recuperar artefatos cacheados, mas não é necessário para obter o MovieLens 100K.

## Reproduzir a entrega com Docker

```bash
git clone https://github.com/RafaExMachina/movielens-recommender-uv.git
cd movielens-recommender-uv

export LOCAL_UID="$(id -u)"
export LOCAL_GID="$(id -g)"

mkdir -p ../movielens-dvc-storage

docker compose build pipeline
docker compose up -d mlflow
docker compose run --rm pipeline
```

Acesse:

```text
http://localhost:5000
```

Encerre os serviços:

```bash
docker compose down
```

## Critérios atendidos

### Etapa 1

- estrutura modular;
- Clean Code;
- princípios SOLID;
- padrões de projeto;
- type hints;
- docstrings;
- lint;
- testes;
- pre-commit;
- integração contínua.

### Etapa 2

- dependências de produção e desenvolvimento separadas;
- `pyproject.toml` configurado;
- `uv.lock` versionado;
- Python 3.12 definido;
- configurações externalizadas;
- validação com Pydantic Settings;
- script de validação do ambiente;
- instalação limpa com `uv sync --locked`;
- build de wheel e source distribution;
- PyTorch CPU.

### Etapa 3

- download reproduzível do dataset público MovieLens 100K;
- remote DVC configurado para artefatos cacheados;
- pipeline com cinco estágios;
- `dvc.yaml` e `dvc.lock`;
- métricas versionadas;
- execução incremental e forçada;
- rastreamento com MLflow;
- persistência com SQLite e volume Docker;
- Dockerfile multi-stage;
- usuário não-root;
- Docker Compose;
- healthcheck do MLflow;
- pipeline reproduzível dentro do contêiner;
- versão `0.3.1`.

## Próximas etapas

Possíveis evoluções:

- registro e promoção de modelos;
- API de inferência;
- testes de integração adicionais;
- automação de build e publicação da imagem;
- CI/CD;
- observabilidade;
- implantação em ambiente de nuvem ou Kubernetes.

## Status atual

A entrega está validada quando os comandos abaixo terminam sem erros:

```bash
uv lock --check
uv sync --locked
uv run python scripts/validate_env.py
uv run dvc repro
uv run dvc status
uv run ruff check src tests scripts
uv run mypy src
uv run pytest
uv run pre-commit run --all-files
docker compose config
```

Estado validado na versão:

```text
0.3.1
```

