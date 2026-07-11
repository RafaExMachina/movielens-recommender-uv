# MovieLens Recommender System com uv

Sistema de recomendação baseado no dataset MovieLens 100K, estruturado como um projeto profissional de Machine Learning.

Este repositório apresenta a implementação da:

> **Etapa 1 — Clean Code e Estrutura**

O foco desta etapa é estabelecer uma base limpa, modular, testável e reproduzível antes da implementação das etapas de versionamento de dados, rastreamento de experimentos e containerização.

## Objetivos da Etapa 1

Nesta etapa foram implementados:

- estrutura modular com `src/`, `tests/`, `configs/`, `scripts/`, `data/` e `models/`;
- nomes descritivos e módulos com responsabilidades bem definidas;
- type hints nas funções públicas;
- docstrings no padrão Google;
- aplicação de princípios SOLID;
- padrões de projeto Factory, Strategy, Template Method e Repository;
- gerenciamento de dependências com `uv`;
- separação entre dependências de produção e desenvolvimento;
- lock file versionado com `uv.lock`;
- testes automatizados com Pytest;
- análise estática com Ruff e mypy;
- hooks de qualidade com pre-commit;
- integração contínua com GitHub Actions;
- configuração de `.gitignore`, `.dockerignore` e `.env.example`;
- histórico de commits semântico.

## Tecnologias utilizadas nesta etapa

- Python 3.12
- uv
- PyTorch
- Scikit-Learn
- PyYAML
- Pytest
- Ruff
- mypy
- pre-commit
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
│   └── train.py
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
│   ├── test_data_loader.py
│   ├── test_metrics.py
│   ├── test_model_factory.py
│   ├── test_preprocess.py
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

Os diretórios de dados, modelos e relatórios utilizam arquivos `.gitkeep` para que a estrutura seja preservada no Git sem versionar datasets, checkpoints ou resultados gerados.

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
| `tracking/` | abstrações para rastreamento de experimentos |
| `utils/` | configurações, logging, caminhos e reprodutibilidade |

## Padrões de projeto

### Factory

Utilizado para centralizar a criação dos modelos de recomendação.

```text
src/recommender/models/model_factory.py
```

A Factory reduz o acoplamento entre o pipeline e as implementações concretas dos modelos.

### Strategy

Utilizado para permitir diferentes estratégias de avaliação.

```text
src/recommender/evaluation/metric_strategy.py
```

Novas métricas podem ser adicionadas sem modificar o fluxo principal de avaliação.

### Template Method

Utilizado na estrutura dos pipelines.

```text
src/recommender/pipeline/base_pipeline.py
```

A classe base define a sequência geral de execução, enquanto subclasses implementam etapas específicas.

### Repository

Utilizado para abstrair o armazenamento e a recuperação de artefatos.

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

## Instalar as dependências

```bash
uv sync --dev
```

Esse comando:

- cria o ambiente virtual em `.venv`;
- instala as dependências do projeto;
- instala as ferramentas de desenvolvimento;
- utiliza as versões registradas no `uv.lock`.

Não é necessário ativar manualmente o ambiente virtual para usar os comandos com `uv run`.

## Executar os testes

```bash
uv run pytest
```

Ou pelo Makefile:

```bash
make test
```

Para mostrar mais detalhes durante a execução:

```bash
uv run pytest -v
```

## Verificação de qualidade

### Ruff

Executa análise estática do código:

```bash
make lint
```

Equivalente a:

```bash
uv run ruff check src tests
```

### mypy

Verifica a consistência dos type hints:

```bash
make type
```

Equivalente a:

```bash
uv run mypy src
```

### Pytest

Executa os testes automatizados:

```bash
make test
```

### Verificação completa

```bash
make check
```

A verificação completa executa:

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
uv run ruff format src tests
```

## Pre-commit

Instale os hooks localmente:

```bash
uv run pre-commit install
```

Depois da instalação, as verificações configuradas serão executadas antes de cada commit.

Para executar manualmente em todos os arquivos:

```bash
uv run pre-commit run --all-files
```

## Integração contínua

O workflow está configurado em:

```text
.github/workflows/ci.yml
```

A integração contínua verifica automaticamente o projeto após pushes e pull requests.

As verificações incluem:

- instalação das dependências;
- lint com Ruff;
- verificação de tipos com mypy;
- testes com Pytest;
- validações configuradas no pre-commit.

Um resultado aprovado no GitHub Actions indica que o repositório pode ser instalado e testado em um ambiente limpo.

## Arquivos não versionados

O `.gitignore` impede o envio de arquivos locais ou gerados, incluindo:

```text
.venv/
__pycache__/
.pytest_cache/
.mypy_cache/
.ruff_cache/
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

O arquivo `.env` também não é versionado. Apenas o modelo abaixo é disponibilizado:

```text
.env.example
```

## Comandos principais

| Comando | Descrição |
|---|---|
| `uv sync --dev` | Instala dependências de produção e desenvolvimento |
| `make lint` | Executa análise estática com Ruff |
| `make format` | Formata o código |
| `make type` | Executa verificação de tipos com mypy |
| `make test` | Executa os testes com Pytest |
| `make check` | Executa todas as verificações de qualidade |
| `make clean` | Remove caches e arquivos temporários |

## Reproduzir a validação da Etapa 1

Uma pessoa que baixar o projeto pode validar a entrega com:

```bash
git clone https://github.com/RafaExMachina/movielens-recommender-uv.git
cd movielens-recommender-uv
uv sync --dev
make check
```

Também é possível executar apenas:

```bash
uv run pytest
```

## Escopo das próximas etapas

Os seguintes recursos fazem parte da evolução planejada do projeto, mas não são o foco desta entrega:

- download e versionamento do MovieLens 100K;
- pipeline de dados com DVC;
- rastreamento de experimentos com MLflow;
- treinamento completo dos modelos;
- geração de métricas e artefatos;
- containerização com Docker;
- orquestração com Docker Compose;
- execução reproduzível do pipeline completo.

Esses recursos serão adicionados em commits e etapas específicas.

## Status da Etapa 1

A Etapa 1 é considerada concluída quando:

- a estrutura do projeto está organizada;
- os módulos possuem responsabilidades bem definidas;
- os padrões de projeto estão implementados;
- as funções públicas possuem type hints e docstrings;
- Ruff não apresenta erros;
- mypy não apresenta erros;
- os testes passam;
- os hooks do pre-commit passam;
- o workflow do GitHub Actions finaliza com sucesso;
- outra pessoa consegue clonar, instalar e testar o projeto.