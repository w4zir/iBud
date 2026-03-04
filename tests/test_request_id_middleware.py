from __future__ import annotations

import re

from fastapi.testclient import TestClient

from backend.main import create_app


def test_request_id_is_echoed_when_provided():
    app = create_app()
    client = TestClient(app)

    response = client.get("/metrics", headers={"X-Request-ID": "req-abc-123"})

    assert response.status_code == 200
    assert response.headers.get("X-Request-ID") == "req-abc-123"


def test_request_id_is_generated_when_missing():
    app = create_app()
    client = TestClient(app)

    response = client.get("/metrics")
    request_id = response.headers.get("X-Request-ID")

    assert response.status_code == 200
    assert isinstance(request_id, str)
    assert bool(re.match(r"^[0-9a-fA-F-]{36}$", request_id or ""))
