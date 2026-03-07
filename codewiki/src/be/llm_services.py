"""
LLM service factory for creating configured LLM clients.
"""

from codewiki.src.be.tracing import emit_trace_block
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.models.openai import OpenAIModelSettings
from pydantic_ai.models.fallback import FallbackModel
from openai import OpenAI

from codewiki.src.config import Config


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
    trace_label: str | None = None,
    trace_context: str | None = None,
) -> str:
    """
    Call LLM with the given prompt.

    Args:
        prompt: The prompt to send
        config: Configuration containing LLM settings
        model: Model name (defaults to config.main_model)
        temperature: Temperature setting
        trace_label: Optional trace label for verbose output
        trace_context: Optional trace context for verbose output

    Returns:
        LLM response text
    """
    if model is None:
        model = config.main_model

    emit_trace_block(
        config,
        "LLM REQUEST",
        prompt,
        model=model,
        label=trace_label,
        context=trace_context,
    )

    client = create_openai_client(config)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=config.max_tokens,
    )
    content = response.choices[0].message.content
    if content is None:
        raise RuntimeError("LLM response did not include message content")
    emit_trace_block(
        config,
        "LLM RESPONSE",
        content,
        model=model,
        label=trace_label,
        context=trace_context,
    )
    return content
