.PHONY: up down rebuild logs bootstrap nb-sync nb-clean test lint shell

up:
	docker compose up -d

down:
	docker compose down

rebuild:
	docker compose build --no-cache

logs:
	docker compose logs -f --tail=200

bootstrap:
	docker compose exec jupyter python scripts/bootstrap_data.py --years 2020-2024 --uf SP

nb-sync:
	docker compose exec jupyter jupytext --sync notebooks/*.py

nb-clean:
	docker compose exec jupyter jupyter nbconvert --clear-output --inplace notebooks/*.ipynb

shell:
	docker compose exec jupyter bash

test:
	docker compose exec jupyter pytest -q

lint:
	docker compose exec jupyter ruff check .
	docker compose exec jupyter mypy src
