"""Legacy tracing helpers kept only for model labels."""

from __future__ import annotations

from typing import Any


def agent_model_label(config: Any) -> str:
    """Return a readable model label for agent-based runs."""
    main_model = getattr(config, "main_model", "")
    fallback_model = getattr(config, "fallback_model", "")
    if main_model and fallback_model:
        return f"{main_model} -> {fallback_model}"
    return main_model or fallback_model or "<unknown>"
