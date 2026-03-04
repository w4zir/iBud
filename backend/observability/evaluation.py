from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Optional

from langchain_core.messages import HumanMessage, SystemMessage

from ..config import get_llm

logger = logging.getLogger(__name__)

_GROUNDEDNESS_SYSTEM = (
    "You are an evaluator.  Given a CONTEXT and a RESPONSE, score how well "
    "the response is grounded in the provided context.  Return ONLY a JSON "
    'object: {"score": <float 0.0-1.0>}.  1.0 means fully grounded, '
    "0.0 means completely unsupported."
)

_HALLUCINATION_SYSTEM = (
    "You are a hallucination detector.  Given a CONTEXT and a RESPONSE, "
    "determine whether the response contains claims not supported by the "
    'context.  Return ONLY a JSON object: {"hallucinated": <true|false>}.'
)

_HELPFULNESS_SYSTEM = (
    "You are a helpfulness evaluator.  Given a USER QUERY and a RESPONSE, "
    "score how helpful the response is for the user.  Return ONLY a JSON "
    'object: {"score": <float 0.0-1.0>}.  1.0 means maximally helpful.'
)

_POLICY_SYSTEM = (
    "You are a policy compliance evaluator for a customer-support agent.  "
    "The agent must: (1) never promise what it cannot deliver, (2) escalate "
    "angry or complex issues, (3) stay on-topic, (4) protect user data.  "
    "Given a RESPONSE, score policy compliance.  Return ONLY a JSON object: "
    '{"score": <float 0.0-1.0>}.  1.0 means fully compliant.'
)


@dataclass
class EvaluationResult:
    session_id: str
    groundedness: Optional[float] = None
    hallucination: Optional[bool] = None
    helpfulness: Optional[float] = None
    policy_compliance: Optional[float] = None


def _parse_score(raw: str, key: str = "score") -> Optional[float]:
    try:
        data = json.loads(raw.strip())
        return float(data[key])
    except Exception:
        return None


def _parse_bool(raw: str, key: str = "hallucinated") -> Optional[bool]:
    try:
        data = json.loads(raw.strip())
        return bool(data[key])
    except Exception:
        return None


async def evaluate_groundedness(
    response: str,
    context: str,
    *,
    llm: object | None = None,
) -> Optional[float]:
    """Score how well *response* is grounded in *context* (0-1)."""
    try:
        llm = llm or get_llm()
        human = f"CONTEXT:\n{context}\n\nRESPONSE:\n{response}"
        result = await llm.ainvoke(  # type: ignore[union-attr]
            [SystemMessage(content=_GROUNDEDNESS_SYSTEM), HumanMessage(content=human)]
        )
        return _parse_score(result.content or "")
    except Exception:
        logger.warning("groundedness evaluation failed", exc_info=True)
        return None


async def evaluate_hallucination(
    response: str,
    context: str,
    *,
    llm: object | None = None,
) -> Optional[bool]:
    """Detect whether *response* contains hallucinated claims."""
    try:
        llm = llm or get_llm()
        human = f"CONTEXT:\n{context}\n\nRESPONSE:\n{response}"
        result = await llm.ainvoke(  # type: ignore[union-attr]
            [SystemMessage(content=_HALLUCINATION_SYSTEM), HumanMessage(content=human)]
        )
        return _parse_bool(result.content or "")
    except Exception:
        logger.warning("hallucination evaluation failed", exc_info=True)
        return None


async def evaluate_helpfulness(
    response: str,
    query: str,
    *,
    llm: object | None = None,
) -> Optional[float]:
    """Score how helpful *response* is for the user's *query* (0-1)."""
    try:
        llm = llm or get_llm()
        human = f"USER QUERY:\n{query}\n\nRESPONSE:\n{response}"
        result = await llm.ainvoke(  # type: ignore[union-attr]
            [SystemMessage(content=_HELPFULNESS_SYSTEM), HumanMessage(content=human)]
        )
        return _parse_score(result.content or "")
    except Exception:
        logger.warning("helpfulness evaluation failed", exc_info=True)
        return None


async def evaluate_policy_compliance(
    response: str,
    *,
    llm: object | None = None,
) -> Optional[float]:
    """Score policy compliance of *response* (0-1)."""
    try:
        llm = llm or get_llm()
        human = f"RESPONSE:\n{response}"
        result = await llm.ainvoke(  # type: ignore[union-attr]
            [SystemMessage(content=_POLICY_SYSTEM), HumanMessage(content=human)]
        )
        return _parse_score(result.content or "")
    except Exception:
        logger.warning("policy compliance evaluation failed", exc_info=True)
        return None


async def evaluate_session(
    session_id: str,
    query: str,
    response: str,
    context: str,
    *,
    llm: object | None = None,
) -> EvaluationResult:
    """
    Run the full evaluation suite for a completed session.

    Each evaluator is independent; failures in one do not block the others.
    """
    groundedness = await evaluate_groundedness(response, context, llm=llm)
    hallucination = await evaluate_hallucination(response, context, llm=llm)
    helpfulness = await evaluate_helpfulness(response, query, llm=llm)
    policy = await evaluate_policy_compliance(response, llm=llm)

    return EvaluationResult(
        session_id=session_id,
        groundedness=groundedness,
        hallucination=hallucination,
        helpfulness=helpfulness,
        policy_compliance=policy,
    )


__all__ = [
    "EvaluationResult",
    "evaluate_groundedness",
    "evaluate_hallucination",
    "evaluate_helpfulness",
    "evaluate_policy_compliance",
    "evaluate_session",
]
