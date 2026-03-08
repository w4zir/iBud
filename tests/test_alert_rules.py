from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ALERT_RULES_PATH = ROOT / "infra" / "prometheus" / "alert_rules.yml"


def test_alert_rules_file_exists():
    assert ALERT_RULES_PATH.exists(), "Expected alert_rules.yml to exist"


def test_alert_rules_contains_required_alerts_and_fields():
    text = ALERT_RULES_PATH.read_text(encoding="utf-8")

    expected_alerts = [
        "HighErrorRate",
        "P95LatencyRegression",
        "RetrievalFailureSpike",
        "EscalationSpike",
        "EvaluationQualityRegression",
    ]

    for alert_name in expected_alerts:
        assert f"- alert: {alert_name}" in text

    # Validate expected structural fields in each rule block.
    blocks = [b for b in text.split("- alert: ")[1:] if b.strip()]
    for block in blocks:
        assert "expr:" in block
        assert "for:" in block
        assert "severity:" in block
        assert "runbook:" in block
