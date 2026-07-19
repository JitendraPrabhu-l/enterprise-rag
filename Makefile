.PHONY: up down logs build lint typecheck test eval-gate clean

up:
	docker compose up -d --build

down:
	docker compose down

clean:
	docker compose down -v

logs:
	docker compose logs -f

build:
	docker compose build

lint:
	ruff check packages services deploy
	ruff format --check packages services deploy

typecheck:
	mypy packages/rag_core/rag_core
	mypy services/ingestion/rag_ingestion
	mypy services/retrieval/rag_retrieval
	mypy services/generation/rag_generation
	mypy services/eval/rag_eval

test:
	pytest packages/rag_core -v
	pytest services/ingestion/tests -v
	pytest services/retrieval/tests -v
	pytest services/generation/tests -v
	pytest services/eval/tests -v

eval-gate:
	python services/eval/scripts/run_eval_gate.py \
		--dataset services/eval/fixtures/synthetic_eval_set.json \
		--generation-url http://localhost:8003 \
		--retrieval-url http://localhost:8002 \
		--output eval-results.json
