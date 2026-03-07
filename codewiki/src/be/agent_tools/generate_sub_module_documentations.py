from pydantic_ai import RunContext, Tool, Agent
import time

from codewiki.cli.utils.logging import configure_logging, log_module_event
from codewiki.src.be.agent_tools.deps import CodeWikiDeps
from codewiki.src.be.agent_tools.read_code_components import read_code_components_tool
from codewiki.src.be.agent_tools.str_replace_editor import str_replace_editor_tool
from codewiki.src.be.llm_logging import (
    format_payload,
    log_llm_content,
    log_llm_summary,
    write_llm_markdown_artifact,
)
from codewiki.src.be.llm_services import create_fallback_models
from codewiki.src.be.tracing import agent_model_label
from codewiki.src.be.utils import is_complex_module, count_tokens
from codewiki.src.be.cluster_modules import format_potential_core_components

import logging

logger = logging.getLogger(__name__)


async def generate_sub_module_documentation(
    ctx: RunContext[CodeWikiDeps], sub_module_specs: dict[str, list[str]]
) -> str:
    """Generate detailed description of a given sub-module specs to the sub-agents

    Args:
        sub_module_specs: The specs of the sub-modules to generate documentation for. E.g. {"sub_module_1": ["core_component_1.1", "core_component_1.2"], "sub_module_2": ["core_component_2.1", "core_component_2.2"], ...}
    """

    deps = ctx.deps
    configure_logging(int(getattr(deps.config, "verbosity", 0)))
    previous_module_name = deps.current_module_name

    # Create fallback models from config
    fallback_models = create_fallback_models(deps.config)

    # add the sub-module to the module tree
    value = deps.module_tree
    for key in deps.path_to_current_module:
        value = value[key]["children"]
    for sub_module_name, core_component_ids in sub_module_specs.items():
        value[sub_module_name] = {"components": core_component_ids, "children": {}}

    total_submodules = len(sub_module_specs)
    for index, (sub_module_name, core_component_ids) in enumerate(
        sub_module_specs.items(), start=1
    ):
        submodule_depth = deps.current_depth + 1
        submodule_path = (
            "/".join([*deps.path_to_current_module, sub_module_name]) or sub_module_name
        )
        log_module_event(
            logger,
            current=index,
            total=total_submodules,
            module_kind="submodule",
            module_path=submodule_path,
            status="start",
            depth=submodule_depth,
        )

        num_tokens = count_tokens(
            format_potential_core_components(core_component_ids, ctx.deps.components)[-1]
        )

        if (
            is_complex_module(ctx.deps.components, core_component_ids)
            and ctx.deps.current_depth < ctx.deps.max_depth
            and num_tokens >= ctx.deps.config.max_token_per_leaf_module
        ):
            system_prompt = ctx.deps.config.prompts.build_system_prompt(
                sub_module_name, ctx.deps.custom_instructions
            )
            log_llm_content(
                logger,
                "AGENT SYSTEM PROMPT",
                system_prompt,
                prompt_type="sub_module_generation",
                model=agent_model_label(ctx.deps.config),
                context=sub_module_name,
            )
            sub_agent: Agent[CodeWikiDeps, str] = Agent(
                model=fallback_models,
                name=sub_module_name,
                deps_type=CodeWikiDeps,
                system_prompt=system_prompt,
                tools=[
                    read_code_components_tool,
                    str_replace_editor_tool,
                    generate_sub_module_documentation_tool,
                ],
            )
        else:
            system_prompt = ctx.deps.config.prompts.build_leaf_system_prompt(
                sub_module_name, ctx.deps.custom_instructions
            )
            log_llm_content(
                logger,
                "AGENT SYSTEM PROMPT",
                system_prompt,
                prompt_type="sub_module_generation",
                model=agent_model_label(ctx.deps.config),
                context=sub_module_name,
            )
            sub_agent = Agent[CodeWikiDeps, str](
                model=fallback_models,
                name=sub_module_name,
                deps_type=CodeWikiDeps,
                system_prompt=system_prompt,
                tools=[read_code_components_tool, str_replace_editor_tool],
            )

        deps.current_module_name = sub_module_name
        deps.path_to_current_module.append(sub_module_name)
        deps.current_depth += 1
        # log the current module tree
        # print(f"Current module tree: {json.dumps(deps.module_tree, indent=4)}")

        try:
            user_prompt = ctx.deps.config.prompts.build_user_prompt(
                module_name=deps.current_module_name,
                core_component_ids=core_component_ids,
                components=ctx.deps.components,
                module_tree=ctx.deps.module_tree,
            )
            model_label = agent_model_label(ctx.deps.config)
            log_llm_content(
                logger,
                "AGENT USER PROMPT",
                user_prompt,
                prompt_type="sub_module_generation",
                model=model_label,
                context=sub_module_name,
            )
            request_tokens = count_tokens(system_prompt) + count_tokens(user_prompt)
            log_llm_summary(
                logger,
                "request",
                prompt_type="sub_module_generation",
                request_tokens=request_tokens,
            )
            started_at = time.perf_counter()
            result = await sub_agent.run(
                user_prompt,
                deps=ctx.deps,
            )
            duration_ms = round((time.perf_counter() - started_at) * 1000)
            duration_seconds = duration_ms / 1000
            response_tokens = count_tokens(result.output)
            response_tokens_per_second = (
                response_tokens / duration_seconds if duration_seconds > 0 else None
            )
            log_llm_summary(
                logger,
                "response",
                prompt_type="sub_module_generation",
                duration_seconds=duration_seconds,
                response_tokens=response_tokens,
                response_tokens_per_second=response_tokens_per_second,
            )
            log_llm_content(
                logger,
                "AGENT RESULT",
                result.output,
                prompt_type="sub_module_generation",
                model=model_label,
                context=sub_module_name,
            )
            message_history, message_history_language = format_payload(result.new_messages_json())
            log_llm_content(
                logger,
                "AGENT MESSAGE HISTORY",
                message_history,
                prompt_type="sub_module_generation",
                model=model_label,
                context=sub_module_name,
            )
            write_llm_markdown_artifact(
                ctx.deps.config,
                prompt_type="sub_module_generation",
                model=model_label,
                context=sub_module_name,
                duration_seconds=duration_seconds,
                request_tokens=request_tokens,
                response_tokens=response_tokens,
                response_tokens_per_second=response_tokens_per_second,
                sections=(
                    ("System Prompt", system_prompt, "text"),
                    ("User Prompt", user_prompt, "text"),
                    ("Result", result.output, "markdown"),
                    ("Message History", message_history, message_history_language),
                ),
            )
            log_module_event(
                logger,
                current=index,
                total=total_submodules,
                module_kind="submodule",
                module_path=submodule_path,
                status="done",
                duration_seconds=duration_seconds,
                depth=submodule_depth,
            )
        except Exception:
            log_module_event(
                logger,
                current=index,
                total=total_submodules,
                module_kind="submodule",
                module_path=submodule_path,
                status="failed",
                depth=submodule_depth,
            )
            raise
        finally:
            # Restore traversal state even if a nested agent run fails.
            deps.path_to_current_module.pop()
            deps.current_depth -= 1

    # restore the previous module name
    deps.current_module_name = previous_module_name

    return f"Generate successfully. Documentations: {', '.join([key + '.md' for key in sub_module_specs.keys()])} are saved in the working directory."


generate_sub_module_documentation_tool = Tool(
    function=generate_sub_module_documentation,
    name="generate_sub_module_documentation",
    description="Generate detailed description of a given sub-module specs to the sub-agents",
    takes_ctx=True,
)
