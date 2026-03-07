from pydantic_ai import RunContext, Tool, Agent

from codewiki.src.be.agent_tools.deps import CodeWikiDeps
from codewiki.src.be.agent_tools.read_code_components import read_code_components_tool
from codewiki.src.be.agent_tools.str_replace_editor import str_replace_editor_tool
from codewiki.src.be.llm_services import create_fallback_models
from codewiki.src.be.tracing import agent_model_label, emit_json_trace_block, emit_trace_block
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
    previous_module_name = deps.current_module_name

    # Create fallback models from config
    fallback_models = create_fallback_models(deps.config)

    # add the sub-module to the module tree
    value = deps.module_tree
    for key in deps.path_to_current_module:
        value = value[key]["children"]
    for sub_module_name, core_component_ids in sub_module_specs.items():
        value[sub_module_name] = {"components": core_component_ids, "children": {}}

    for sub_module_name, core_component_ids in sub_module_specs.items():

        # Create visual indentation for nested modules
        indent = "  " * deps.current_depth
        arrow = "└─" if deps.current_depth > 0 else "→"

        logger.info(f"{indent}{arrow} Generating documentation for sub-module: {sub_module_name}")

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
            emit_trace_block(
                ctx.deps.config,
                "AGENT SYSTEM PROMPT",
                system_prompt,
                model=agent_model_label(ctx.deps.config),
                label="sub_module_generation",
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
            emit_trace_block(
                ctx.deps.config,
                "AGENT SYSTEM PROMPT",
                system_prompt,
                model=agent_model_label(ctx.deps.config),
                label="sub_module_generation",
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

        user_prompt = ctx.deps.config.prompts.build_user_prompt(
            module_name=deps.current_module_name,
            core_component_ids=core_component_ids,
            components=ctx.deps.components,
            module_tree=ctx.deps.module_tree,
        )
        emit_trace_block(
            ctx.deps.config,
            "AGENT USER PROMPT",
            user_prompt,
            model=agent_model_label(ctx.deps.config),
            label="sub_module_generation",
            context=sub_module_name,
        )
        result = await sub_agent.run(
            user_prompt,
            deps=ctx.deps,
        )
        emit_trace_block(
            ctx.deps.config,
            "AGENT RESULT",
            result.output,
            model=agent_model_label(ctx.deps.config),
            label="sub_module_generation",
            context=sub_module_name,
        )
        emit_json_trace_block(
            ctx.deps.config,
            "AGENT MESSAGE HISTORY",
            result.new_messages_json(),
            model=agent_model_label(ctx.deps.config),
            label="sub_module_generation",
            context=sub_module_name,
        )

        # remove the sub-module name from the path to current module and the module tree
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
