from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DASHBOARD_DIR = ROOT / "infra" / "grafana" / "dashboards"


def _load_dashboard(name: str) -> dict:
    path = DASHBOARD_DIR / name
    assert path.exists(), f"Missing dashboard file: {name}"
    return json.loads(path.read_text(encoding="utf-8"))


def test_executive_dashboard_shape():
    data = _load_dashboard("executive-observability.json")
    assert data["uid"] == "executive-observability"
    assert isinstance(data.get("panels"), list) and data["panels"]
    titles = {panel.get("title") for panel in data["panels"]}
    assert "Automation Rate (Prometheus)" in titles
    assert "Escalation Rate (Prometheus)" in titles


def test_product_dashboard_shape():
    data = _load_dashboard("product-observability.json")
    assert data["uid"] == "product-observability"
    assert isinstance(data.get("panels"), list) and data["panels"]
    titles = {panel.get("title") for panel in data["panels"]}
    assert "Task Completion Rate" in titles
    assert "Tool Success Rate" in titles


def test_ai_quality_dashboard_shape():
    data = _load_dashboard("ai-quality-observability.json")
    assert data["uid"] == "ai-quality-observability"
    assert isinstance(data.get("panels"), list) and data["panels"]
    titles = {panel.get("title") for panel in data["panels"]}
    assert "Retrieval Latency (p95)" in titles
    assert "Tool Failure Rate" in titles
