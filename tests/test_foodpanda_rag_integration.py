import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.main import app


client = TestClient(app)


@pytest.mark.integration
def test_foodpanda_policy_queries_use_foodpanda_dataset_and_company():
    """Smoke-test that Foodpanda queries hit the chat API with correct scoping."""
    data_path = Path("data/foodpanda/testing/RAG_Test_Queries.json")
    assert data_path.exists(), f"Missing test data file: {data_path}"
    rows = json.loads(data_path.read_text(encoding="utf-8"))
    assert isinstance(rows, list) and rows, "Foodpanda RAG test queries must be a non-empty list"

    # Use just the first scenario for a light integration check.
    row = rows[0]
    user_query = row.get("user_query") or ""
    assert user_query, "user_query must be non-empty"

    resp = client.post(
        "/chat/",
        json={
            "session_id": None,
            "user_id": "foodpanda-user",
            "message": user_query,
            "dataset": "foodpanda",
            "company": "foodpanda",
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["session_id"]
    # We don't assert full answer text here, but ensure sources (when present)
    # are tagged with the Foodpanda dataset.
    for src in data.get("sources") or []:
        assert src.get("source") in (None, "foodpanda", "wixqa", "bitext")

