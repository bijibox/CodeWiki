"""Helpers for LLM logging and artifact persistence."""

from __future__ import annotations

from datetime import datetime
import json
import logging
from pathlib import Path
import re
from typing import Any, Iterable, Mapping

from codewiki.src.config import Config
from codewiki.src.utils import file_manager

_DEFAULT_PROMPT_TYPE = "llm"


def log_llm_summary(
    logger: logging.Logger,
    phase: str,
    *,
    prompt_type: str | None,
    duration_seconds: float | None = None,
    request_tokens: int | None = None,
    response_tokens: int | None = None,
    response_tokens_per_second: float | None = None,
) -> None:
    """Log a compact LLM event for verbosity level 3."""
    logger.debug(
        "",
        extra={
            "event_type": "llm_summary",
            "verbosity_gate": 3,
            "llm_phase": phase,
            "llm_prompt_type": prompt_type or _DEFAULT_PROMPT_TYPE,
            "llm_duration_seconds": duration_seconds,
            "llm_request_tokens": request_tokens,
            "llm_response_tokens": response_tokens,
            "llm_response_tokens_per_second": response_tokens_per_second,
        },
    )


def log_llm_content(
    logger: logging.Logger,
    title: str,
    content: str,
    *,
    prompt_type: str | None,
    model: str | None = None,
    context: str | None = None,
) -> None:
    """Log detailed LLM content for verbosity level 4."""
    logger.debug(
        content,
        extra={
            "event_type": "llm_content",
            "verbosity_gate": 4,
            "llm_title": title,
            "llm_prompt_type": prompt_type or _DEFAULT_PROMPT_TYPE,
            "llm_model": model,
            "llm_context": context,
        },
    )


def format_payload(payload: bytes | str | Any) -> tuple[str, str]:
    """Format payload for logs and markdown artifacts."""
    if isinstance(payload, bytes):
        decoded = payload.decode("utf-8")
    elif isinstance(payload, str):
        decoded = payload
    else:
        decoded = json.dumps(_normalize_payload(payload), ensure_ascii=False, default=str)

    try:
        parsed = json.loads(decoded)
    except json.JSONDecodeError:
        return decoded, "text"

    return json.dumps(parsed, indent=2, ensure_ascii=False), "json"


def write_llm_markdown_artifact(
    config: Config,
    *,
    prompt_type: str | None,
    sections: Iterable[tuple[str, str, str | None]],
    model: str | None = None,
    context: str | None = None,
    duration_seconds: float | None = None,
    request_tokens: int | None = None,
    response_tokens: int | None = None,
    response_tokens_per_second: float | None = None,
    extra_metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Persist a markdown artifact for a single LLM interaction."""
    normalized_prompt_type = _sanitize_prompt_type(prompt_type)
    now = datetime.now()
    target_dir = Path(config.docs_dir) / "temp" / "llm"
    file_manager.ensure_directory(target_dir)

    target_path = _build_unique_artifact_path(target_dir, now, normalized_prompt_type)

    metadata_lines = [
        f"- Timestamp: `{now.isoformat(timespec='milliseconds')}`",
        f"- Prompt type: `{normalized_prompt_type}`",
    ]
    if model:
        metadata_lines.append(f"- Model: `{model}`")
    if context:
        metadata_lines.append(f"- Context: `{context}`")
    if duration_seconds is not None:
        metadata_lines.append(f"- Duration: `{duration_seconds:.3f} s`")
    if request_tokens is not None:
        metadata_lines.append(f"- Request tokens: `{request_tokens}`")
    if response_tokens is not None:
        metadata_lines.append(f"- Response tokens: `{response_tokens}`")
    if response_tokens_per_second is not None:
        metadata_lines.append(f"- Response speed: `{response_tokens_per_second:.3f} tokens/s`")
    if extra_metadata:
        for key, value in extra_metadata.items():
            metadata_lines.append(f"- {key}: {_format_metadata_value(value)}")

    content_parts = ["# LLM Interaction", "", "## Metadata", *metadata_lines]
    for heading, body, language in sections:
        content_parts.extend(
            [
                "",
                f"## {heading}",
                f"```{language or 'text'}",
                body,
                "```",
            ]
        )

    target_path.write_text("\n".join(content_parts) + "\n", encoding="utf-8")
    return target_path


def _normalize_payload(payload: Any) -> Any:
    """Best-effort conversion of SDK objects to JSON-serializable structures."""
    if hasattr(payload, "to_dict") and callable(payload.to_dict):
        return _normalize_payload(payload.to_dict())
    if hasattr(payload, "model_dump") and callable(payload.model_dump):
        return _normalize_payload(payload.model_dump())
    if isinstance(payload, Mapping):
        return {str(key): _normalize_payload(value) for key, value in payload.items()}
    if isinstance(payload, (list, tuple, set)):
        return [_normalize_payload(item) for item in payload]
    if hasattr(payload, "__dict__"):
        return {
            key: _normalize_payload(value)
            for key, value in vars(payload).items()
            if not key.startswith("_")
        }
    return payload


def _format_metadata_value(value: Any) -> str:
    """Render metadata values consistently in markdown artifacts."""
    if value is None:
        return "`null`"
    if isinstance(value, bool):
        return f"`{str(value).lower()}`"
    if isinstance(value, (int, float)):
        return f"`{value}`"
    if isinstance(value, str):
        return f"`{value}`"
    normalized = _normalize_payload(value)
    return f"`{json.dumps(normalized, ensure_ascii=False, default=str)}`"


def _sanitize_prompt_type(prompt_type: str | None) -> str:
    raw_value = (prompt_type or _DEFAULT_PROMPT_TYPE).strip().lower()
    sanitized = re.sub(r"[^a-z0-9._-]+", "-", raw_value).strip("-")
    return sanitized or _DEFAULT_PROMPT_TYPE


def _build_unique_artifact_path(target_dir: Path, now: datetime, prompt_type: str) -> Path:
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S.") + f"{int(now.microsecond / 1000):03d}"
    base_name = f"{timestamp}-{prompt_type}"
    candidate = target_dir / f"{base_name}.md"
    suffix = 1

    while candidate.exists():
        suffix += 1
        candidate = target_dir / f"{base_name}-{suffix}.md"

    return candidate
