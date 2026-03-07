"""
LLM service factory for creating configured LLM clients.
"""

from __future__ import annotations

import logging
import time

from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.models.openai import OpenAIModelSettings
from pydantic_ai.models.fallback import FallbackModel
from openai import OpenAI

from codewiki.cli.utils.logging import configure_logging
from codewiki.src.be.llm_logging import (
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
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=config.max_tokens,
    )
    duration_ms = round((time.perf_counter() - started_at) * 1000)
    duration_seconds = duration_ms / 1000
    content = response.choices[0].message.content
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
    write_llm_markdown_artifact(
        config,
        prompt_type=prompt_type,
        model=model,
        context=context,
        duration_seconds=duration_seconds,
        request_tokens=request_tokens,
        response_tokens=response_tokens,
        response_tokens_per_second=response_tokens_per_second,
        sections=(
            ("Request", prompt, "text"),
            ("Response", content, "markdown"),
        ),
    )
    return content
