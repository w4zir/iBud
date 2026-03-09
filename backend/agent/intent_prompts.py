from __future__ import annotations

import os
from typing import Dict, List, Tuple, TypedDict

from .prompts import SYSTEM_INTENT_CLASSIFIER


DEFAULT_INTENTS: List[str] = [
    "order_status",
    "return_request",
    "product_qa",
    "account_issue",
    "complaint",
    "other",
]


BITEXT_INTENTS: List[str] = [
    "cancel_order",
    "change_order",
    "change_shipping_address",
    "check_cancellation_fee",
    "check_invoice",
    "check_payment_methods",
    "check_refund_policy",
    "complaint",
    "contact_customer_service",
    "contact_human_agent",
    "create_account",
    "delete_account",
    "delivery_options",
    "delivery_period",
    "edit_account",
    "get_invoice",
    "get_refund",
    "newsletter_subscription",
    "payment_issue",
    "place_order",
    "recover_password",
    "registration_problems",
    "review",
    "set_up_shipping_address",
    "switch_account",
    "track_order",
    "track_refund",
]


SYSTEM_INTENT_CLASSIFIER_BITEXT = (
    "You are an e-commerce customer support triage assistant.\n"
    "Classify the user's latest message into ONE of the following intents.\n"
    "Respond with ONLY the intent name (lowercase, snake_case):\n\n"
    + "\n".join(f"- {name}" for name in BITEXT_INTENTS)
).strip()


class IntentPromptProfile(TypedDict):
    system_prompt: str
    allowed_intents: List[str]


_PROFILES: Dict[str, IntentPromptProfile] = {
    "default": {
        "system_prompt": SYSTEM_INTENT_CLASSIFIER,
        "allowed_intents": DEFAULT_INTENTS,
    },
    "bitext": {
        "system_prompt": SYSTEM_INTENT_CLASSIFIER_BITEXT,
        "allowed_intents": BITEXT_INTENTS,
    },
}


def get_intent_prompt_profile(
    name: str | None = None,
) -> Tuple[str, List[str], str]:
    """
    Resolve an intent-classification prompt profile.

    Precedence:
    - Explicit `name` argument (when provided and recognised)
    - `INTENT_PROMPT_PROFILE` environment variable
    - Built-in \"default\" profile

    Returns a tuple of:
    - system_prompt (str)
    - allowed_intents (List[str])
    - resolved_profile_name (str)
    """
    raw = (name or "").strip().lower()
    if not raw:
        raw = (os.getenv("INTENT_PROMPT_PROFILE") or "default").strip().lower()

    profile = _PROFILES.get(raw)
    if profile is None:
        # Fall back to the default profile if an unknown name is provided.
        raw = "default"
        profile = _PROFILES["default"]

    return profile["system_prompt"], list(profile["allowed_intents"]), raw


__all__ = [
    "DEFAULT_INTENTS",
    "BITEXT_INTENTS",
    "SYSTEM_INTENT_CLASSIFIER_BITEXT",
    "get_intent_prompt_profile",
]

