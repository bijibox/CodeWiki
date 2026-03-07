"""
LLM service factory for creating configured LLM clients.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.models.openai import OpenAIModelSettings
from pydantic_ai.models.fallback import FallbackModel
from openai import OpenAI

from codewiki.cli.utils.logging import configure_logging
from codewiki.src.be.llm_logging import (
    format_payload,
    log_llm_content,
    log_llm_summary,
    write_llm_markdown_artifact,
)
from codewiki.src.be.utils import count_tokens
from codewiki.src.config import Config

logger = logging.getLogger(__name__)


def create_main_model(config: Config) -> OpenAIModel:
    """Create the main LLM model from configuration."""
    return OpenAIModel(
        model_name=config.main_model,
        provider=OpenAIProvider(base_url=config.llm_base_url, api_key=config.llm_api_key),
        settings=OpenAIModelSettings(temperature=0.0, max_tokens=config.max_tokens),
    )


def create_fallback_model(config: Config) -> OpenAIModel:
    """Create the fallback LLM model from configuration."""
    return OpenAIModel(
        model_name=config.fallback_model,
        provider=OpenAIProvider(base_url=config.llm_base_url, api_key=config.llm_api_key),
        settings=OpenAIModelSettings(temperature=0.0, max_tokens=config.max_tokens),
    )


def create_fallback_models(config: Config) -> FallbackModel:
    """Create fallback models chain from configuration."""
    main = create_main_model(config)
    fallback = create_fallback_model(config)
    return FallbackModel(main, fallback)


def create_openai_client(config: Config) -> OpenAI:
    """Create OpenAI client from configuration."""
    return OpenAI(base_url=config.llm_base_url, api_key=config.llm_api_key)


def call_llm(
    prompt: str,
    config: Config,
    model: str | None = None,
    temperature: float = 0.0,
    prompt_type: str | None = None,
    context: str | None = None,
) -> str:
    """
    Call LLM with the given prompt.

    Args:
        prompt: The prompt to send
        config: Configuration containing LLM settings
        model: Model name (defaults to config.main_model)
        temperature: Temperature setting
        prompt_type: Optional prompt type for logging and artifacts
        context: Optional prompt context for logging and artifacts

    Returns:
        LLM response text
    """
    if model is None:
        model = config.main_model

    configure_logging(int(getattr(config, "verbosity", 0)))
    request_tokens = count_tokens(prompt)
    log_llm_summary(logger, "request", prompt_type=prompt_type, request_tokens=request_tokens)
    log_llm_content(
        logger,
        "LLM REQUEST",
        prompt,
        prompt_type=prompt_type,
        model=model,
        context=context,
    )

    client = create_openai_client(config)
    started_at = time.perf_counter()
    response: Any | None = None
    content: str | None = None
    reasoning = ""
    finish_reason: str | None = None
    usage: Any = None
    duration_seconds: float | None = None

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=config.max_tokens,
        )
        duration_ms = round((time.perf_counter() - started_at) * 1000)
        duration_seconds = duration_ms / 1000

        choice = response.choices[0]
        message = choice.message
        content = message.content
        reasoning = _extract_reasoning(message)
        finish_reason = getattr(choice, "finish_reason", None)
        usage = getattr(response, "usage", None)

        if content is None:
            raise RuntimeError("LLM response did not include message content")

        response_tokens = count_tokens(content)
        response_tokens_per_second = (
            response_tokens / duration_seconds if duration_seconds > 0 else None
        )

        log_llm_summary(
            logger,
            "response",
            prompt_type=prompt_type,
            duration_seconds=duration_seconds,
            response_tokens=response_tokens,
            response_tokens_per_second=response_tokens_per_second,
        )
        log_llm_content(
            logger,
            "LLM RESPONSE",
            content,
            prompt_type=prompt_type,
            model=model,
            context=context,
        )

        sections: list[tuple[str, str, str | None]] = [
            ("Request", prompt, "text"),
            ("Response", content, "markdown"),
        ]
        if reasoning:
            sections.append(("Reasoning", reasoning, "text"))

        write_llm_markdown_artifact(
            config,
            prompt_type=prompt_type,
            model=model,
            context=context,
            duration_seconds=duration_seconds,
            request_tokens=request_tokens,
            response_tokens=response_tokens,
            response_tokens_per_second=response_tokens_per_second,
            extra_metadata=_build_artifact_metadata(
                finish_reason=finish_reason,
                content_missing=False,
                usage=usage,
                reasoning=reasoning,
            ),
            sections=sections,
        )
        return content
    except Exception as exc:
        if duration_seconds is None:
            duration_ms = round((time.perf_counter() - started_at) * 1000)
            duration_seconds = duration_ms / 1000
        _write_failure_artifact(
            prompt=prompt,
            config=config,
            prompt_type=prompt_type,
            context=context,
            model=model,
            request_tokens=request_tokens,
            duration_seconds=duration_seconds,
            response=response,
            reasoning=reasoning,
            finish_reason=finish_reason,
            usage=usage,
            content_missing=content is None,
            error=exc,
        )
        raise


def _extract_reasoning(message: Any) -> str:
    """Read provider-specific reasoning text when present."""
    reasoning = getattr(message, "reasoning", None)
    if isinstance(reasoning, str):
        return reasoning
    return ""


def _build_artifact_metadata(
    *,
    finish_reason: str | None,
    content_missing: bool,
    usage: Any,
    reasoning: str,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {"Content missing": content_missing}
    if finish_reason is not None:
        metadata["Finish reason"] = finish_reason
    if usage is not None:
        metadata["Usage"] = _build_usage_metadata(usage, reasoning)
    return metadata


def _write_failure_artifact(
    *,
    prompt: str,
    config: Config,
    prompt_type: str | None,
    context: str | None,
    model: str | None,
    request_tokens: int,
    duration_seconds: float,
    response: Any,
    reasoning: str,
    finish_reason: str | None,
    usage: Any,
    content_missing: bool,
    error: Exception,
) -> None:
    """Persist request/response diagnostics for failed LLM calls."""
    sections: list[tuple[str, str, str | None]] = [("Request", prompt, "text")]
    if reasoning:
        sections.append(("Reasoning", reasoning, "text"))
    if response is not None:
        raw_response, raw_response_language = format_payload(response)
        sections.append(("Raw Response", raw_response, raw_response_language))
    sections.append(("Error", str(error), "text"))

    write_llm_markdown_artifact(
        config,
        prompt_type=prompt_type,
        model=model,
        context=context,
        duration_seconds=duration_seconds,
        request_tokens=request_tokens,
        extra_metadata=_build_artifact_metadata(
            finish_reason=finish_reason,
            content_missing=content_missing,
            usage=usage,
            reasoning=reasoning,
        ),
        sections=sections,
    )


def _build_usage_metadata(usage: Any, reasoning: str) -> Any:
    """Attach reasoning token details to usage metadata when available."""
    serialized_usage = _serialize_usage(usage)
    if not isinstance(serialized_usage, dict):
        return serialized_usage

    reasoning_tokens = _extract_reasoning_tokens(usage)
    if reasoning_tokens is not None:
        serialized_usage["reasoning_tokens"] = reasoning_tokens
        return serialized_usage

    if reasoning:
        serialized_usage["reasoning_tokens_estimated"] = count_tokens(reasoning)

    return serialized_usage


def _serialize_usage(usage: Any) -> Any:
    """Convert usage objects to plain data for artifact metadata."""
    if hasattr(usage, "to_dict") and callable(usage.to_dict):
        return usage.to_dict()
    if hasattr(usage, "model_dump") and callable(usage.model_dump):
        return usage.model_dump()
    if hasattr(usage, "__dict__"):
        return {key: value for key, value in vars(usage).items() if not key.startswith("_")}
    return usage


def _extract_reasoning_tokens(usage: Any) -> int | None:
    """Read provider-reported reasoning token counts when exposed."""
    completion_details = getattr(usage, "completion_tokens_details", None)
    if completion_details is None:
        return None

    reasoning_tokens = getattr(completion_details, "reasoning_tokens", None)
    if isinstance(reasoning_tokens, int):
        return reasoning_tokens

    if hasattr(completion_details, "to_dict") and callable(completion_details.to_dict):
        details_dict = completion_details.to_dict()
        value = details_dict.get("reasoning_tokens")
        if isinstance(value, int):
            return value

    return None
