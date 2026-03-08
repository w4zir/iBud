from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

from .events import recent_interventions, record_intervention
from .governance import GovernanceConfig
from .rules import RemediationRule, RuleEvaluation, default_rules


class RemediationEngine:
    def __init__(
        self,
        *,
        governance: GovernanceConfig | None = None,
        rules: List[RemediationRule] | None = None,
    ) -> None:
        self.governance = governance or GovernanceConfig.from_env()
        self.rules = rules or default_rules()

    async def _is_in_cooldown(self, evaluation: RuleEvaluation) -> bool:
        history = await recent_interventions(hours=24)
        now = datetime.now(timezone.utc)
        for entry in history:
            if entry.get("rule_name") != evaluation.name:
                continue
            ts_raw = entry.get("created_at")
            if not ts_raw:
                continue
            try:
                created = datetime.fromisoformat(ts_raw.replace("Z", "+00:00"))
            except Exception:
                continue
            effective_cooldown = max(
                evaluation.cooldown_seconds,
                self.governance.min_cooldown_seconds,
            )
            if created + timedelta(seconds=effective_cooldown) > now:
                return True
        return False

    async def _within_action_budget(self) -> bool:
        history = await recent_interventions(hours=1)
        return len(history) < self.governance.max_actions_per_hour

    async def evaluate_rules(self) -> List[Dict[str, Any]]:
        checks: List[Dict[str, Any]] = []
        for rule in self.rules:
            if not self.governance.is_rule_enabled(rule.name, default=rule.enabled):
                checks.append(
                    {
                        "name": rule.name,
                        "enabled": False,
                        "triggered": False,
                        "reason": "disabled_by_governance",
                    }
                )
                continue
            evaluation = await rule.evaluate()
            checks.append(
                {
                    "name": evaluation.name,
                    "enabled": True,
                    "triggered": evaluation.triggered,
                    "reason": evaluation.reason,
                    "action": evaluation.action,
                    "metrics": evaluation.metrics,
                    "cooldown_seconds": evaluation.cooldown_seconds,
                }
            )
        return checks

    async def run(self, *, dry_run: bool = True) -> Dict[str, Any]:
        checks = await self.evaluate_rules()
        triggered = [c for c in checks if c.get("triggered")]
        executed: List[Dict[str, Any]] = []

        if dry_run:
            return {
                "mode": "dry_run",
                "governance": asdict(self.governance),
                "checks": checks,
                "triggered": len(triggered),
                "executed": executed,
            }

        if not await self._within_action_budget():
            return {
                "mode": "active",
                "governance": asdict(self.governance),
                "checks": checks,
                "triggered": len(triggered),
                "executed": executed,
                "blocked": "max_actions_per_hour_reached",
            }

        for rule in self.rules:
            check = next((c for c in checks if c.get("name") == rule.name), None)
            if not check or not check.get("triggered"):
                continue

            evaluation = RuleEvaluation(
                name=str(check["name"]),
                triggered=bool(check["triggered"]),
                action=str(check["action"]),
                reason=str(check["reason"]),
                metrics=dict(check.get("metrics") or {}),
                cooldown_seconds=int(check.get("cooldown_seconds") or 0),
            )
            if await self._is_in_cooldown(evaluation):
                executed.append(
                    {
                        "rule_name": rule.name,
                        "status": "skipped",
                        "reason": "cooldown_active",
                    }
                )
                continue

            action_result = await rule.remediate(evaluation.metrics)
            await record_intervention(
                session_id=None,
                trace_id=None,
                rule_name=rule.name,
                action_taken=evaluation.action,
                trigger_metrics=evaluation.metrics,
                outcome=action_result,
            )
            executed.append(
                {
                    "rule_name": rule.name,
                    "status": "executed",
                    "action": evaluation.action,
                    "outcome": action_result,
                }
            )

        return {
            "mode": "active",
            "governance": asdict(self.governance),
            "checks": checks,
            "triggered": len(triggered),
            "executed": executed,
        }
