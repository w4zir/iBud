#!/usr/bin/env bash
set -eu

echo "Starting core services with Docker Compose (postgres, redis, ollama, backend)..."
docker compose up -d postgres redis ollama backend

echo "Running document ingestion inside backend container..."
docker compose exec -T backend python -m rag.ingest

echo "Seeding mock orders inside backend container..."
docker compose exec -T backend python /app/scripts/seed_mock_data.py

echo "Ingest + seed completed."

