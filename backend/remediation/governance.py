from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class GovernanceConfig:
    global_enabled: bool = True
    manual_override: bool = False
    min_cooldown_seconds: int = 300
    max_actions_per_hour: int = 5
    rule_enabled_overrides: Dict[str, bool] | None = None

    @classmethod
    def from_env(cls) -> "GovernanceConfig":
        return cls(
            global_enabled=_env_bool("REMEDIATION_ENABLED", True),
            manual_override=_env_bool("REMEDIATION_MANUAL_OVERRIDE", False),
            min_cooldown_seconds=int(os.getenv("REMEDIATION_MIN_COOLDOWN_SECONDS", "300")),
            max_actions_per_hour=int(os.getenv("REMEDIATION_MAX_ACTIONS_PER_HOUR", "5")),
            rule_enabled_overrides={},
        )

    def is_rule_enabled(self, rule_name: str, default: bool = True) -> bool:
        if not self.global_enabled or self.manual_override:
            return False
        if not self.rule_enabled_overrides:
            return default
        return self.rule_enabled_overrides.get(rule_name, default)
