from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict

from sqlalchemy import Float, cast, func, select

from ..db.models import EvaluationScore, Session
from ..db.postgres import async_session_factory


@dataclass
class DriftReport:
    groundedness_recent: float
    groundedness_baseline: float
    groundedness_drop: float
    hallucination_recent: float
    hallucination_baseline: float
    hallucination_rise: float
    intent_shift_ratio: Dict[str, float]
    is_drifted: bool


async def detect_model_data_drift(*, recent_hours: int = 24, baseline_days: int = 7) -> DriftReport:
    now = datetime.now(timezone.utc)
    recent_cutoff = now - timedelta(hours=recent_hours)
    baseline_cutoff = now - timedelta(days=baseline_days)

    async with async_session_factory() as db:
        recent_eval = await db.execute(
            select(
                func.coalesce(func.avg(EvaluationScore.groundedness), 0.0),
                func.coalesce(
                    func.avg(cast(EvaluationScore.hallucination, Float)),
                    0.0,
                ),
            ).where(EvaluationScore.evaluated_at >= recent_cutoff)
        )
        recent_groundedness, recent_hallucination = recent_eval.one()

        baseline_eval = await db.execute(
            select(
                func.coalesce(func.avg(EvaluationScore.groundedness), 0.0),
                func.coalesce(
                    func.avg(cast(EvaluationScore.hallucination, Float)),
                    0.0,
                ),
            ).where(
                EvaluationScore.evaluated_at >= baseline_cutoff,
                EvaluationScore.evaluated_at < recent_cutoff,
            )
        )
        baseline_groundedness, baseline_hallucination = baseline_eval.one()

        intent_rows = await db.execute(
            select(
                Session.intent,
                func.count(Session.id),
            )
            .where(Session.created_at >= recent_cutoff)
            .group_by(Session.intent)
        )
        recent_intent = {str(intent or "unknown"): int(count) for intent, count in intent_rows.all()}

        baseline_intent_rows = await db.execute(
            select(
                Session.intent,
                func.count(Session.id),
            )
            .where(
                Session.created_at >= baseline_cutoff,
                Session.created_at < recent_cutoff,
            )
            .group_by(Session.intent)
        )
        baseline_intent = {
            str(intent or "unknown"): int(count)
            for intent, count in baseline_intent_rows.all()
        }

    intent_shift: Dict[str, float] = {}
    all_intents = set(recent_intent.keys()) | set(baseline_intent.keys())
    for intent in sorted(all_intents):
        recent = float(recent_intent.get(intent, 0))
        baseline = float(baseline_intent.get(intent, 0))
        ratio = recent / baseline if baseline > 0 else (1.0 if recent == 0 else 999.0)
        intent_shift[intent] = ratio

    groundedness_drop = float(baseline_groundedness or 0.0) - float(recent_groundedness or 0.0)
    hallucination_rise = float(recent_hallucination or 0.0) - float(baseline_hallucination or 0.0)

    drifted = groundedness_drop > 0.15 or hallucination_rise > 0.10 or any(
        ratio > 2.0 for ratio in intent_shift.values()
    )

    return DriftReport(
        groundedness_recent=float(recent_groundedness or 0.0),
        groundedness_baseline=float(baseline_groundedness or 0.0),
        groundedness_drop=groundedness_drop,
        hallucination_recent=float(recent_hallucination or 0.0),
        hallucination_baseline=float(baseline_hallucination or 0.0),
        hallucination_rise=hallucination_rise,
        intent_shift_ratio=intent_shift,
        is_drifted=drifted,
    )
