.PHONY: setup dev ingest test docker-up docker-down clean

# Copy .env.example to .env if it doesn't exist
setup:
	@test -f .env || cp .env.example .env
	pip install -r requirements.txt
	@echo "✅ Setup complete. Edit .env with your LLM endpoint and API keys."

# Run the agent in development mode
dev:
	uvicorn main:app --host 0.0.0.0 --port 8083 --reload

# Ingest sample documents into the RAG knowledge base
ingest:
	python3 ingest.py --path docs/ --source "Sample Docs"

# Run basic health check
test:
	@curl -s http://localhost:8083/api/v1/health | python3 -m json.tool

# Start all services with Docker Compose
docker-up:
	docker compose up -d --build
	@echo "✅ Services started. Agent available at http://localhost:8083"

# Stop all services
docker-down:
	docker compose down

# Stop and remove all data
clean:
	docker compose down -v
	@echo "🗑️  All containers and volumes removed."
