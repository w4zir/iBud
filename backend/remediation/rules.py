from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Protocol

from sqlalchemy import Float, cast, func, select

from ..db.models import AgentSpan, EvaluationScore
from ..db.postgres import async_session_factory


class RemediationAction(Protocol):
    async def __call__(self) -> str: ...


@dataclass
class RuleEvaluation:
    name: str
    triggered: bool
    action: str
    reason: str
    metrics: Dict[str, Any]
    cooldown_seconds: int


@dataclass
class RemediationRule:
    name: str
    cooldown_seconds: int
    enabled: bool = True

    async def evaluate(self) -> RuleEvaluation:
        raise NotImplementedError

    async def remediate(self, metrics: Dict[str, Any]) -> str:
        return "noop"


class RetrievalQualityDropRule(RemediationRule):
    def __init__(self) -> None:
        super().__init__(name="retrieval_quality_drop", cooldown_seconds=1800)

    async def evaluate(self) -> RuleEvaluation:
        cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
        async with async_session_factory() as db:
            result = await db.execute(
                select(func.coalesce(func.avg(EvaluationScore.groundedness), 0.0)).where(
                    EvaluationScore.evaluated_at >= cutoff
                )
            )
            groundedness = float(result.scalar_one() or 0.0)
        triggered = groundedness < 0.55
        return RuleEvaluation(
            name=self.name,
            triggered=triggered,
            action="trigger_reingestion_reindex",
            reason="groundedness_below_threshold" if triggered else "healthy",
            metrics={"avg_groundedness_24h": groundedness, "threshold": 0.55},
            cooldown_seconds=self.cooldown_seconds,
        )

    async def remediate(self, metrics: Dict[str, Any]) -> str:
        return "reindex_trigger_requested"


class ToolFailureSpikeRule(RemediationRule):
    def __init__(self) -> None:
        super().__init__(name="tool_failure_spike", cooldown_seconds=900)

    async def evaluate(self) -> RuleEvaluation:
        cutoff = datetime.now(timezone.utc) - timedelta(hours=1)
        async with async_session_factory() as db:
            result = await db.execute(
                select(
                    func.coalesce(func.sum(cast((AgentSpan.attributes["failure_count"]).astext, Float)), 0.0),
                    func.coalesce(func.sum(cast((AgentSpan.attributes["success_count"]).astext, Float)), 0.0),
                ).where(
                    AgentSpan.span_name == "execute_tool",
                    AgentSpan.created_at >= cutoff,
                )
            )
            failures, successes = result.one()

        failures_f = float(failures or 0.0)
        successes_f = float(successes or 0.0)
        total = failures_f + successes_f
        failure_rate = failures_f / total if total > 0 else 0.0
        triggered = total >= 5 and failure_rate > 0.30
        return RuleEvaluation(
            name=self.name,
            triggered=triggered,
            action="activate_tool_circuit_breaker",
            reason="tool_failure_rate_high" if triggered else "healthy",
            metrics={"failure_rate_1h": failure_rate, "samples": total, "threshold": 0.30},
            cooldown_seconds=self.cooldown_seconds,
        )

    async def remediate(self, metrics: Dict[str, Any]) -> str:
        return "tool_fallback_mode_enabled"


class HallucinationIncreaseRule(RemediationRule):
    def __init__(self) -> None:
        super().__init__(name="hallucination_increase", cooldown_seconds=1800)

    async def evaluate(self) -> RuleEvaluation:
        cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
        async with async_session_factory() as db:
            result = await db.execute(
                select(
                    func.coalesce(
                        func.avg(cast(EvaluationScore.hallucination, Float)),
                        0.0,
                    )
                ).where(EvaluationScore.evaluated_at >= cutoff)
            )
            hallucination_rate = float(result.scalar_one() or 0.0)

        triggered = hallucination_rate > 0.10
        return RuleEvaluation(
            name=self.name,
            triggered=triggered,
            action="enable_stricter_grounding_policy",
            reason="hallucination_rate_high" if triggered else "healthy",
            metrics={
                "hallucination_rate_24h": hallucination_rate,
                "threshold": 0.10,
            },
            cooldown_seconds=self.cooldown_seconds,
        )

    async def remediate(self, metrics: Dict[str, Any]) -> str:
        return "strict_grounding_policy_enabled"


def default_rules() -> list[RemediationRule]:
    return [
        RetrievalQualityDropRule(),
        ToolFailureSpikeRule(),
        HallucinationIncreaseRule(),
    ]
