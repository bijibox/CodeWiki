from pydantic_ai import Agent

# import logfire
import logging
import os
import time
import traceback
from typing import Any, Dict, List

from codewiki.cli.utils.logging import configure_logging

# Configure logging and monitoring

logger = logging.getLogger(__name__)

# try:
#     # Configure logfire with environment variables for Docker compatibility
#     logfire_token = os.getenv('LOGFIRE_TOKEN')
#     logfire_project = os.getenv('LOGFIRE_PROJECT_NAME', 'default')
#     logfire_service = os.getenv('LOGFIRE_SERVICE_NAME', 'default')

#     if logfire_token:
#         # Configure with explicit token (for Docker)
#         logfire.configure(
#             token=logfire_token,
#             project_name=logfire_project,
#             service_name=logfire_service,
#         )
#     else:
#         # Use default configuration (for local development with logfire auth)
#         logfire.configure(
#             project_name=logfire_project,
#             service_name=logfire_service,
#         )

#     logfire.instrument_pydantic_ai()
#     logger.debug(f"Logfire configured successfully for project: {logfire_project}")

# except Exception as e:
#     logger.warning(f"Failed to configure logfire: {e}")

# Local imports
from codewiki.src.be.agent_tools.deps import CodeWikiDeps
from codewiki.src.be.agent_tools.read_code_components import read_code_components_tool
from codewiki.src.be.agent_tools.str_replace_editor import str_replace_editor_tool
from codewiki.src.be.agent_tools.generate_sub_module_documentations import (
    generate_sub_module_documentation_tool,
)
from codewiki.src.be.llm_logging import (
    format_payload,
    log_llm_content,
    log_llm_summary,
    write_llm_markdown_artifact,
)
from codewiki.src.be.llm_services import create_fallback_models
from codewiki.src.be.tracing import agent_model_label
from codewiki.src.be.utils import is_complex_module
from codewiki.src.config import (
    Config,
    MODULE_TREE_FILENAME,
    OVERVIEW_FILENAME,
)
from codewiki.src.utils import file_manager
from codewiki.src.be.dependency_analyzer.models.core import Node


class AgentOrchestrator:
    """Orchestrates the AI agents for documentation generation."""

    def __init__(self, config: Config):
        self.config = config
        configure_logging(int(getattr(config, "verbosity", 0)))
        self.fallback_models = create_fallback_models(config)
        self.custom_instructions = config.get_prompt_addition() if config else None

    def create_agent(
        self, module_name: str, components: Dict[str, Any], core_component_ids: List[str]
    ) -> tuple[Agent[CodeWikiDeps, str], str]:
        """Create an appropriate agent based on module complexity."""

        if is_complex_module(components, core_component_ids):
            system_prompt = self.config.prompts.build_system_prompt(
                module_name, self.custom_instructions
            )
            log_llm_content(
                logger,
                "AGENT SYSTEM PROMPT",
                system_prompt,
                prompt_type="module_generation",
                model=agent_model_label(self.config),
                context=module_name,
            )
            return (
                Agent[CodeWikiDeps, str](
                    self.fallback_models,
                    name=module_name,
                    deps_type=CodeWikiDeps,
                    tools=[
                        read_code_components_tool,
                        str_replace_editor_tool,
                        generate_sub_module_documentation_tool,
                    ],
                    system_prompt=system_prompt,
                ),
                system_prompt,
            )
        else:
            system_prompt = self.config.prompts.build_leaf_system_prompt(
                module_name, self.custom_instructions
            )
            log_llm_content(
                logger,
                "AGENT SYSTEM PROMPT",
                system_prompt,
                prompt_type="module_generation",
                model=agent_model_label(self.config),
                context=module_name,
            )
            return (
                Agent[CodeWikiDeps, str](
                    self.fallback_models,
                    name=module_name,
                    deps_type=CodeWikiDeps,
                    tools=[read_code_components_tool, str_replace_editor_tool],
                    system_prompt=system_prompt,
                ),
                system_prompt,
            )

    async def process_module(
        self,
        module_name: str,
        components: Dict[str, Node],
        core_component_ids: List[str],
        module_path: List[str],
        working_dir: str,
    ) -> tuple[Dict[str, Any], str]:
        """Process a single module and generate its documentation."""
        logger.info(f"Processing module: {module_name}")

        # Load or create module tree
        module_tree_path = os.path.join(working_dir, MODULE_TREE_FILENAME)
        module_tree = file_manager.load_json(module_tree_path) or {}

        # Create agent
        agent, system_prompt = self.create_agent(module_name, components, core_component_ids)

        # Create dependencies
        deps = CodeWikiDeps(
            absolute_docs_path=working_dir,
            absolute_repo_path=str(os.path.abspath(self.config.repo_path)),
            registry={},
            components=components,
            path_to_current_module=module_path,
            current_module_name=module_name,
            module_tree=module_tree,
            max_depth=self.config.max_depth,
            current_depth=1,
            config=self.config,
            custom_instructions=self.custom_instructions,
        )

        # check if overview docs already exists
        overview_docs_path = os.path.join(working_dir, OVERVIEW_FILENAME)
        if os.path.exists(overview_docs_path):
            logger.info(f"✓ Overview docs already exists at {overview_docs_path}")
            return module_tree, "cached"

        # check if module docs already exists
        docs_path = os.path.join(working_dir, f"{module_name}.md")
        if os.path.exists(docs_path):
            logger.info(f"✓ Module docs already exists at {docs_path}")
            return module_tree, "cached"

        # Run agent
        try:
            user_prompt = self.config.prompts.build_user_prompt(
                module_name=module_name,
                core_component_ids=core_component_ids,
                components=components,
                module_tree=deps.module_tree,
            )
            model_label = agent_model_label(self.config)
            log_llm_content(
                logger,
                "AGENT USER PROMPT",
                user_prompt,
                prompt_type="module_generation",
                model=model_label,
                context=module_name,
            )
            log_llm_summary(logger, "request", prompt_type="module_generation")
            started_at = time.perf_counter()
            result = await agent.run(
                user_prompt,
                deps=deps,
            )
            duration_ms = round((time.perf_counter() - started_at) * 1000)
            log_llm_summary(
                logger,
                "response",
                prompt_type="module_generation",
                duration_ms=duration_ms,
            )
            log_llm_content(
                logger,
                "AGENT RESULT",
                result.output,
                prompt_type="module_generation",
                model=model_label,
                context=module_name,
            )
            message_history, message_history_language = format_payload(result.new_messages_json())
            log_llm_content(
                logger,
                "AGENT MESSAGE HISTORY",
                message_history,
                prompt_type="module_generation",
                model=model_label,
                context=module_name,
            )
            write_llm_markdown_artifact(
                self.config,
                prompt_type="module_generation",
                model=model_label,
                context=module_name,
                duration_ms=duration_ms,
                sections=(
                    ("System Prompt", system_prompt, "text"),
                    ("User Prompt", user_prompt, "text"),
                    ("Result", result.output, "markdown"),
                    ("Message History", message_history, message_history_language),
                ),
            )

            # Save updated module tree
            file_manager.save_json(deps.module_tree, module_tree_path)
            logger.debug(f"Successfully processed module: {module_name}")

            return deps.module_tree, "generated"

        except Exception as e:
            logger.error(f"Error processing module {module_name}: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise
