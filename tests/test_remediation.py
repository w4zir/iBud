from __future__ import annotations

from dataclasses import dataclass

import pytest

from backend.remediation.engine import RemediationEngine
from backend.remediation.governance import GovernanceConfig
from backend.remediation.rules import RemediationRule, RuleEvaluation


@dataclass
class DummyRule(RemediationRule):
    should_trigger: bool = False

    async def evaluate(self) -> RuleEvaluation:
        return RuleEvaluation(
            name=self.name,
            triggered=self.should_trigger,
            action="dummy_action",
            reason="triggered" if self.should_trigger else "healthy",
            metrics={"x": 1},
            cooldown_seconds=self.cooldown_seconds,
        )

    async def remediate(self, metrics):
        return "done"


@pytest.mark.asyncio
async def test_remediation_engine_dry_run_reports_checks(monkeypatch):
    monkeypatch.setattr("backend.remediation.engine.recent_interventions", lambda hours=24: [])
    rules = [DummyRule(name="r1", cooldown_seconds=60, should_trigger=True)]
    engine = RemediationEngine(governance=GovernanceConfig(), rules=rules)
    report = await engine.run(dry_run=True)
    assert report["mode"] == "dry_run"
    assert report["triggered"] == 1
    assert report["executed"] == []


@pytest.mark.asyncio
async def test_remediation_engine_executes_triggered_rule(monkeypatch):
    async def fake_recent(hours=24):
        return []

    calls = []

    async def fake_record(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr("backend.remediation.engine.recent_interventions", fake_recent)
    monkeypatch.setattr("backend.remediation.engine.record_intervention", fake_record)
    rules = [DummyRule(name="r1", cooldown_seconds=60, should_trigger=True)]
    engine = RemediationEngine(governance=GovernanceConfig(), rules=rules)
    report = await engine.run(dry_run=False)
    assert report["mode"] == "active"
    assert len(report["executed"]) == 1
    assert report["executed"][0]["status"] == "executed"
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_remediation_engine_respects_manual_override(monkeypatch):
    async def fake_recent(hours=24):
        return []

    monkeypatch.setattr("backend.remediation.engine.recent_interventions", fake_recent)
    rules = [DummyRule(name="r1", cooldown_seconds=60, should_trigger=True)]
    cfg = GovernanceConfig(global_enabled=True, manual_override=True)
    engine = RemediationEngine(governance=cfg, rules=rules)
    report = await engine.run(dry_run=True)
    assert report["triggered"] == 0
    assert report["checks"][0]["enabled"] is False
