install:
	uv sync

lock:
	uv lock

lint:
	uv run ruff check src tests

format:
	uv run ruff format src tests

type:
	uv run mypy src

test:
	uv run pytest

precommit:
	uv run pre-commit run --all-files

check:
	uv run ruff check src tests
	uv run mypy src
	uv run pytest
	uv run pre-commit run --all-files

prepare:
	uv run python scripts/run_pipeline.py prepare

train:
	uv run python scripts/run_pipeline.py train

evaluate:
	uv run python scripts/run_pipeline.py evaluate

pipeline:
	uv run python scripts/run_pipeline.py prepare
	uv run python scripts/run_pipeline.py train
	uv run python scripts/run_pipeline.py evaluate

local-prepare:
	MLFLOW_TRACKING_URI=http://localhost:5000 uv run python scripts/run_pipeline.py prepare

local-train:
	MLFLOW_TRACKING_URI=http://localhost:5000 uv run python scripts/run_pipeline.py train

local-evaluate:
	MLFLOW_TRACKING_URI=http://localhost:5000 uv run python scripts/run_pipeline.py evaluate

local-pipeline:
	MLFLOW_TRACKING_URI=http://localhost:5000 uv run python scripts/run_pipeline.py prepare
	MLFLOW_TRACKING_URI=http://localhost:5000 uv run python scripts/run_pipeline.py train
	MLFLOW_TRACKING_URI=http://localhost:5000 uv run python scripts/run_pipeline.py evaluate

dvc:
	uv run dvc repro

mlflow:
	uv run mlflow ui

docker-mlflow:
	docker compose up -d mlflow

docker-prepare:
	docker compose run --rm app python scripts/run_pipeline.py prepare

docker-train:
	docker compose run --rm app python scripts/run_pipeline.py train

docker-evaluate:
	docker compose run --rm app python scripts/run_pipeline.py evaluate

docker-pipeline:
	docker compose run --rm app python scripts/run_pipeline.py prepare
	docker compose run --rm app python scripts/run_pipeline.py train
	docker compose run --rm app python scripts/run_pipeline.py evaluate

docker-up:
	docker compose up --build

docker-down:
	docker compose down

docker-logs:
	docker compose logs -f

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .mypy_cache .ruff_cache