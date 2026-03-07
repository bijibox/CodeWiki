"""
Tracing helpers for verbose CLI execution.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from codewiki.cli.utils.logging import configure_logging

TRACE_LOGGER = logging.getLogger("codewiki.trace")


def agent_model_label(config: Any) -> str:
    """Return a readable model label for agent-based runs."""
    main_model = getattr(config, "main_model", "")
    fallback_model = getattr(config, "fallback_model", "")
    if main_model and fallback_model:
        return f"{main_model} -> {fallback_model}"
    return main_model or fallback_model or "<unknown>"


def emit_trace_block(
    config: Any,
    title: str,
    content: str,
    *,
    model: str | None = None,
    label: str | None = None,
    context: str | None = None,
) -> None:
    """Emit a structured trace record."""
    configure_logging(int(getattr(config, "verbosity", 0)))
    TRACE_LOGGER.debug(
        content,
        extra={
            "event_type": "trace",
            "verbosity_gate": 3,
            "trace_title": title,
            "trace_model": model,
            "trace_label": label,
            "trace_context": context,
        },
    )


def emit_json_trace_block(
    config: Any,
    title: str,
    payload: bytes | str | Any,
    *,
    model: str | None = None,
    label: str | None = None,
    context: str | None = None,
) -> None:
    """Emit JSON payloads as pretty JSON trace records."""
    if isinstance(payload, bytes):
        decoded = payload.decode("utf-8")
    elif isinstance(payload, str):
        decoded = payload
    else:
        decoded = json.dumps(payload)

    try:
        parsed = json.loads(decoded)
        formatted = json.dumps(parsed, indent=2, ensure_ascii=False)
    except json.JSONDecodeError:
        formatted = decoded

    emit_trace_block(
        config,
        title,
        formatted,
        model=model,
        label=label,
        context=context,
    )
