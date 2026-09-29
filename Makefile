.PHONY: install dev test test-slow lint format samples smoke docker-build docker-test up up-gpu down logs

install:      ## Install dependencies (uv)
	uv sync

dev:          ## Run the app locally with auto-reload on http://localhost:8000
	uv run uvicorn resume_shortlister.api.app:create_app --factory --reload

test:         ## Fast test suite (no large downloads)
	uv run pytest

test-slow:    ## End-to-end ranking quality (~1 GB download on first run)
	uv run pytest -m slow

lint:         ## Lint and check formatting
	uv run ruff check src tests scripts
	uv run ruff format --check src tests scripts

format:       ## Auto-fix lint issues and format
	uv run ruff check --fix src tests scripts
	uv run ruff format src tests scripts

samples:      ## Regenerate the synthetic sample resumes in samples/
	uv run python scripts/make_samples.py

smoke:        ## Upload the samples to a running server and print the shortlist
	uv run python scripts/smoke_test.py

docker-build: ## Build the application image
	docker compose build

docker-test:  ## Run the test suite inside the production image (includes Tesseract)
	docker build --target test -t resume-shortlister:test .
	docker run --rm resume-shortlister:test

up:           ## Start the app (Tesseract fallback OCR)
	docker compose up -d --build

up-gpu:       ## Start the app plus a self-hosted OCR server (NVIDIA GPU required)
	docker compose --profile gpu up -d --build

down:         ## Stop everything
	docker compose --profile gpu down

logs:         ## Follow the app logs
	docker compose logs -f app
