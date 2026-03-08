.PHONY: up down build logs dev

up:
	docker compose up -d --build

down:
	docker compose down

build:
	docker compose build

logs:
	docker compose logs -f

dev:
	uv run uvicorn main:app --host 0.0.0.0 --port 8000 --reload
