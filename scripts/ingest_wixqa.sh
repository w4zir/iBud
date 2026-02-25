#!/usr/bin/env bash
set -eu

echo "Starting core services (postgres, redis, ollama, backend)..."
docker compose up -d postgres redis ollama backend

echo "Running WixQA KB ingestion inside backend container..."
docker compose exec -T backend python -m backend.rag.ingest_wixqa

echo "Seeding mock orders inside backend container..."
docker compose exec -T backend python /app/scripts/seed_mock_data.py

echo "Ingest (WixQA) + seed completed."
