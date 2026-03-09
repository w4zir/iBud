from __future__ import annotations

import os

from backend.agent import intent_prompts


def test_get_intent_prompt_profile_default(monkeypatch) -> None:
    monkeypatch.delenv("INTENT_PROMPT_PROFILE", raising=False)
    system_prompt, allowed, name = intent_prompts.get_intent_prompt_profile(None)

    assert "order_status" in system_prompt
    assert allowed == intent_prompts.DEFAULT_INTENTS
    assert name == "default"


def test_get_intent_prompt_profile_bitext_by_name(monkeypatch) -> None:
    monkeypatch.delenv("INTENT_PROMPT_PROFILE", raising=False)
    system_prompt, allowed, name = intent_prompts.get_intent_prompt_profile("bitext")

    assert "cancel_order" in system_prompt
    assert allowed == intent_prompts.BITEXT_INTENTS
    assert name == "bitext"


def test_get_intent_prompt_profile_env_fallback(monkeypatch) -> None:
    monkeypatch.setenv("INTENT_PROMPT_PROFILE", "bitext")
    system_prompt, allowed, name = intent_prompts.get_intent_prompt_profile(None)

    assert "cancel_order" in system_prompt
    assert allowed == intent_prompts.BITEXT_INTENTS
    assert name == "bitext"


def test_get_intent_prompt_profile_unknown_uses_default(monkeypatch) -> None:
    monkeypatch.delenv("INTENT_PROMPT_PROFILE", raising=False)
    system_prompt, allowed, name = intent_prompts.get_intent_prompt_profile("does-not-exist")

    assert "order_status" in system_prompt
    assert allowed == intent_prompts.DEFAULT_INTENTS
    assert name == "default"

