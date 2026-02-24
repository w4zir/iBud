# backend/main.py — FastAPI entrypoint (Phase 0 placeholder)
from fastapi import FastAPI

app = FastAPI(title="E-Commerce Support RAG API")


@app.get("/health")
def health():
    return {"status": "ok"}
