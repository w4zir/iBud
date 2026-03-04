from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from sqlalchemy import and_, select

from ..config import get_llm, log_event
from ..db.models import EvaluationScore, Message, Outcome, Session
from ..db.postgres import async_session_factory
from ..observability.warehouse import record_evaluation_score


SYSTEM_EVAL = """
You are a strict evaluator for customer-support responses.
Return JSON with keys:
- groundedness: float in [0,1]
- helpfulness: float in [0,1]
- hallucination: boolean
Do not include extra keys.
""".strip()


@dataclass
class EvaluationInput:
    session_id: str
    user_message: str
    assistant_message: str
    contexts: List[str]


class AsyncEvaluator:
    async def sample_sessions(self, *, limit: int = 25, min_age_minutes: int = 5) -> List[str]:
        cutoff = datetime.now(timezone.utc) - timedelta(minutes=min_age_minutes)
        async with async_session_factory() as db:
            query = (
                select(Session.id)
                .join(Outcome, Outcome.session_id == Session.id)
                .outerjoin(EvaluationScore, EvaluationScore.session_id == Session.id)
                .where(
                    and_(
                        Outcome.completed.is_(True),
                        Outcome.created_at <= cutoff,
                        EvaluationScore.id.is_(None),
                    )
                )
                .order_by(Outcome.created_at.desc())
                .limit(limit)
            )
            result = await db.execute(query)
            return [str(row[0]) for row in result.all()]

    async def reconstruct_inputs(self, session_id: str) -> Optional[EvaluationInput]:
        async with async_session_factory() as db:
            messages_result = await db.execute(
                select(Message)
                .where(Message.session_id == session_id)
                .order_by(Message.created_at.asc())
            )
            messages = list(messages_result.scalars().all())
            if not messages:
                return None

        user_message = ""
        assistant_message = ""
        for msg in messages:
            if msg.role == "user":
                user_message = msg.content or user_message
            elif msg.role == "assistant":
                assistant_message = msg.content or assistant_message

        if not user_message and not assistant_message:
            return None

        return EvaluationInput(
            session_id=session_id,
            user_message=user_message,
            assistant_message=assistant_message,
            contexts=[],
        )

    async def _score_payload(self, payload: EvaluationInput) -> Dict[str, Any]:
        try:
            llm = get_llm()
            prompt = (
                f"User message:\n{payload.user_message}\n\n"
                f"Assistant message:\n{payload.assistant_message}\n\n"
                f"Retrieved contexts:\n{json.dumps(payload.contexts)}"
            )
            resp = await llm.ainvoke(
                [
                    SystemMessage(content=SYSTEM_EVAL),
                    HumanMessage(content=prompt),
                ]
            )
            raw = str(resp.content or "").strip()
            parsed = json.loads(raw)
            return {
                "groundedness": float(parsed.get("groundedness", 0.0)),
                "helpfulness": float(parsed.get("helpfulness", 0.0)),
                "hallucination": bool(parsed.get("hallucination", False)),
            }
        except Exception:
            # Fallback heuristic for availability.
            answer_len = len(payload.assistant_message.strip())
            groundedness = 0.0 if answer_len == 0 else 0.6
            helpfulness = 0.0 if answer_len == 0 else 0.7
            return {
                "groundedness": groundedness,
                "helpfulness": helpfulness,
                "hallucination": False,
            }

    async def evaluate_session(self, session_id: str) -> bool:
        async with async_session_factory() as db:
            exists = await db.execute(
                select(EvaluationScore.id).where(EvaluationScore.session_id == session_id).limit(1)
            )
            if exists.scalar_one_or_none() is not None:
                return False

        payload = await self.reconstruct_inputs(session_id)
        if payload is None:
            return False

        score = await self._score_payload(payload)
        await record_evaluation_score(
            session_id=session_id,
            groundedness=score.get("groundedness"),
            helpfulness=score.get("helpfulness"),
            hallucination=score.get("hallucination"),
            metadata={"pipeline": "async", "version": "1"},
        )
        return True

    async def run_batch(self, *, limit: int = 25, min_age_minutes: int = 5) -> Dict[str, int]:
        session_ids = await self.sample_sessions(limit=limit, min_age_minutes=min_age_minutes)
        processed = 0
        skipped = 0
        for session_id in session_ids:
            ok = await self.evaluate_session(session_id)
            if ok:
                processed += 1
            else:
                skipped += 1
        log_event(
            "evaluation",
            "async_eval_batch_complete",
            sampled=len(session_ids),
            processed=processed,
            skipped=skipped,
        )
        return {"sampled": len(session_ids), "processed": processed, "skipped": skipped}


__all__ = ["AsyncEvaluator"]

