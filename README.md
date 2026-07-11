# MovieLens Recommender System com uv

Sistema de recomendação baseado no dataset MovieLens 100K, estruturado como um projeto profissional de Machine Learning.

Versão atual:

```text
0.2.0
```

O repositório contém, até o momento, as seguintes entregas:

- **Etapa 1 — Clean Code e Estrutura**
- **Etapa 2 — Ambiente e Dependências**

O foco atual é fornecer uma base limpa, modular, testável, empacotável e reproduzível antes das etapas de versionamento de dados, rastreamento completo de experimentos e containerização.

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
- pre-commit;
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
- testes para valores padrão, sobrescritas e configurações inválidas;
- validação de instalação em ambiente virtual recriado do zero.

## Tecnologias

- Python 3.12
- uv
- PyTorch CPU
- Pandas
- NumPy
- Scikit-Learn
- MLflow
- DVC
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
├── models/
│   ├── checkpoints/
│   ├── exported/
│   └── registry/
├── notebooks/
├── reports/
│   ├── figures/
│   ├── metrics/
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
│           ├── config.py
│           ├── logger.py
│           ├── paths.py
│           ├── seed.py
│           └── settings.py
├── tests/
│   ├── test_data_loader.py
│   ├── test_metrics.py
│   ├── test_model_factory.py
│   ├── test_preprocess.py
│   ├── test_settings.py
│   └── test_training_pipeline.py
├── .dockerignore
├── .env.example
├── .gitignore
├── .pre-commit-config.yaml
├── .python-version
├── Makefile
├── params.yaml
├── pyproject.toml
├── README.md
└── uv.lock
```

Os diretórios de dados, modelos e relatórios utilizam arquivos `.gitkeep` para preservar a estrutura sem versionar datasets, checkpoints ou resultados gerados.

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
- utiliza as versões exatas registradas no `uv.lock`;
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

Essa configuração reduz o tamanho da instalação ao evitar bibliotecas CUDA e NVIDIA desnecessárias para testes e execução em CPU.

Verifique a instalação:

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

As configurações disponíveis são:

| Variável | Descrição | Valor padrão |
|---|---|---|
| `APP_ENV` | ambiente da aplicação | `development` |
| `LOG_LEVEL` | nível de logging | `INFO` |
| `MLFLOW_TRACKING_URI` | endereço do MLflow | `http://localhost:5000` |

Crie um arquivo `.env` local a partir do modelo:

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

Variáveis do sistema operacional possuem precedência sobre os valores do arquivo `.env`.

Exemplo:

```bash
APP_ENV=test \
LOG_LEVEL=DEBUG \
MLFLOW_TRACKING_URI=http://mlflow:5000 \
uv run python -c \
'from recommender.utils.settings import get_settings; print(get_settings())'
```

## Parâmetros de treinamento

Os parâmetros de experimento permanecem separados das configurações de ambiente.

Eles são armazenados em:

```text
params.yaml
```

Exemplos:

- seed;
- caminhos dos dados;
- proporções de treino, validação e teste;
- dimensão dos embeddings;
- número de épocas;
- batch size;
- learning rate;
- top-k da avaliação.

Essa separação evita misturar configurações de infraestrutura com hiperparâmetros do modelo.

## Validação do ambiente

Execute:

```bash
uv run python scripts/validate_env.py
```

O script verifica:

- versão do Python;
- arquivos necessários para reprodução;
- instalação das principais dependências;
- instalação do pacote `recommender`;
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

Ou pelo Makefile:

```bash
make test
```

Para mostrar mais detalhes:

```bash
uv run pytest -v
```

A versão `0.2.0` possui 10 testes automatizados, incluindo testes das configurações de ambiente.

## Verificação de qualidade

### Ruff

Executa análise estática:

```bash
make lint
```

Equivalente a:

```bash
uv run ruff check src tests
```

### mypy

Verifica type hints:

```bash
make type
```

Equivalente a:

```bash
uv run mypy src
```

### Verificação completa

```bash
make check
```

O comando executa:

```bash
uv run ruff check src tests
uv run mypy src
uv run pytest
uv run pre-commit run --all-files
```

## Formatação

Para formatar o código:

```bash
make format
```

Ou diretamente:

```bash
uv run ruff format src tests scripts
```

## Pre-commit

Instale os hooks:

```bash
uv run pre-commit install
```

Execute todas as validações manualmente:

```bash
uv run pre-commit run --all-files
```

## Build do pacote

Gere o source distribution e o wheel:

```bash
uv build
```

Os arquivos são criados em:

```text
dist/
```

Para a versão atual:

```text
dist/movielens_recommender-0.2.0.tar.gz
dist/movielens_recommender-0.2.0-py3-none-any.whl
```

A pasta `dist/` não é versionada.

Confira a versão instalada:

```bash
uv run python -c \
'from importlib.metadata import version; print(version("movielens-recommender"))'
```

Resultado esperado:

```text
0.2.0
```

## Integração contínua

O workflow está configurado em:

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

Um workflow aprovado indica que o projeto pode ser instalado e validado fora do ambiente local do desenvolvedor.

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
data/raw/*
data/interim/*
data/processed/*
data/features/*
models/checkpoints/*
models/exported/*
models/registry/*
reports/metrics/*
reports/figures/*
reports/predictions/*
```

Arquivos `.gitkeep` são mantidos para preservar a estrutura dos diretórios.

## Comandos principais

| Comando | Descrição |
|---|---|
| `uv sync --locked` | instala o ambiente usando o lockfile |
| `uv sync --locked --no-dev` | instala apenas dependências de produção |
| `uv lock --check` | verifica se o lockfile está atualizado |
| `uv run python scripts/validate_env.py` | valida o ambiente |
| `uv run pytest` | executa os testes |
| `make lint` | executa Ruff |
| `make format` | formata o código |
| `make type` | executa mypy |
| `make test` | executa Pytest |
| `make check` | executa todas as verificações |
| `uv build` | gera wheel e source distribution |
| `make clean` | remove caches locais |

## Reproduzir a validação das Etapas 1 e 2

Uma pessoa pode validar a entrega executando:

```bash
git clone https://github.com/RafaExMachina/movielens-recommender-uv.git
cd movielens-recommender-uv

uv sync --locked
uv run python scripts/validate_env.py
make check
uv build
```

Para simular uma instalação completamente limpa:

```bash
rm -rf .venv
uv sync --locked
uv run python scripts/validate_env.py
make check
uv build
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
- PyTorch CPU;
- versão `0.2.0`.

## Próximas etapas

Os seguintes recursos serão tratados em etapas específicas:

- download e versionamento do MovieLens 100K;
- pipeline de dados com DVC;
- rastreamento completo de experimentos com MLflow;
- treinamento e avaliação do modelo;
- geração e versionamento de métricas;
- containerização com Docker;
- orquestração com Docker Compose;
- execução reproduzível do pipeline completo.

## Status atual

As Etapas 1 e 2 são consideradas concluídas quando os comandos abaixo terminam sem erros:

```bash
uv lock --check
uv sync --locked
uv run python scripts/validate_env.py
make check
uv build
```

Estado validado na versão:

```text
0.2.0
```