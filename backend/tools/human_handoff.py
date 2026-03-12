from __future__ import annotations

import os
from typing import Any, Dict, Optional

import httpx


async def human_handoff_tool(
    *,
    session_id: Optional[str],
    user_id: Optional[str],
    reason: str,
    summary: str,
    plan: Dict[str, Any],
    result: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Best-effort external human handoff integration.

    Controlled by env vars:
    - HUMAN_HANDOFF_URL (required to enable)
    - HUMAN_HANDOFF_API_KEY (optional)
    - HUMAN_HANDOFF_TIMEOUT_SECONDS (default 5)
    """
    url = (os.getenv("HUMAN_HANDOFF_URL") or "").strip()
    if not url:
        return {"success": False, "disabled": True, "error": "HUMAN_HANDOFF_URL not set"}

    timeout_s = float(os.getenv("HUMAN_HANDOFF_TIMEOUT_SECONDS", "5"))
    api_key = (os.getenv("HUMAN_HANDOFF_API_KEY") or "").strip()

    headers: Dict[str, str] = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload: Dict[str, Any] = {
        "session_id": session_id,
        "user_id": user_id,
        "reason": reason,
        "summary": summary,
        "plan": plan,
        "result": result,
    }

    async with httpx.AsyncClient(timeout=timeout_s) as client:
        resp = await client.post(url, json=payload, headers=headers)
        ok = 200 <= resp.status_code < 300
        data: Any
        try:
            data = resp.json()
        except Exception:
            data = resp.text
        return {
            "success": ok,
            "status_code": resp.status_code,
            "response": data,
        }


__all__ = ["human_handoff_tool"]

