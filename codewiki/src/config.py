from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Callable
import argparse
import os
from dotenv import load_dotenv

from codewiki.src.be.prompt_template import (
    DEFAULT_PROMPT_NAME,
    FilePromptTemplateSet,
    PromptBuilder,
)

load_dotenv()

# Constants
OUTPUT_BASE_DIR = "output"
DEPENDENCY_GRAPHS_DIR = "dependency_graphs"
DOCS_DIR = "docs"
FIRST_MODULE_TREE_FILENAME = "first_module_tree.json"
MODULE_TREE_FILENAME = "module_tree.json"
OVERVIEW_FILENAME = "overview.md"
MAX_DEPTH = 2
# Default max token settings
DEFAULT_MAX_TOKENS = 32_768
DEFAULT_MAX_TOKEN_PER_MODULE = 36_369
DEFAULT_MAX_TOKEN_PER_LEAF_MODULE = 16_000
DEFAULT_MERMAID_VALIDATOR = "mermaid_parser_py"
MERMAID_VALIDATORS = ("mermaid_parser_py", "mermaid_ink_api")
# Legacy constants (for backward compatibility)
MAX_TOKEN_PER_MODULE = DEFAULT_MAX_TOKEN_PER_MODULE
MAX_TOKEN_PER_LEAF_MODULE = DEFAULT_MAX_TOKEN_PER_LEAF_MODULE

# CLI context detection
_CLI_CONTEXT = False


def set_cli_context(enabled: bool = True):
    """Set whether we're running in CLI context (vs web app)."""
    global _CLI_CONTEXT
    _CLI_CONTEXT = enabled


def is_cli_context() -> bool:
    """Check if running in CLI context."""
    return _CLI_CONTEXT


# LLM services
# In CLI mode, these will be loaded from ~/.codewiki/config.json + keyring
# In web app mode, use environment variables
MAIN_MODEL = os.getenv("MAIN_MODEL", "claude-sonnet-4")
FALLBACK_MODEL_1 = os.getenv("FALLBACK_MODEL_1", "glm-4p5")
CLUSTER_MODEL = os.getenv("CLUSTER_MODEL", MAIN_MODEL)
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://0.0.0.0:4000/")
LLM_API_KEY = os.getenv("LLM_API_KEY", "sk-1234")


@dataclass
class Config:
    """Configuration class for CodeWiki."""

    repo_path: str
    output_dir: str
    dependency_graph_dir: str
    docs_dir: str
    max_depth: int
    # LLM configuration
    llm_base_url: str
    llm_api_key: str
    main_model: str
    cluster_model: str
    fallback_model: str = FALLBACK_MODEL_1
    # Max token settings
    max_tokens: int = DEFAULT_MAX_TOKENS
    max_token_per_module: int = DEFAULT_MAX_TOKEN_PER_MODULE
    max_token_per_leaf_module: int = DEFAULT_MAX_TOKEN_PER_LEAF_MODULE
    mermaid_validator: str = DEFAULT_MERMAID_VALIDATOR
    # Agent instructions for customization
    agent_instructions: Optional[Dict[str, Any]] = None
    prompt_name: str = DEFAULT_PROMPT_NAME
    verbosity: int = 0
    module_progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None
    prompts: PromptBuilder = field(init=False)

    def __post_init__(self) -> None:
        """Load prompt templates for the configured prompt set."""
        if self.mermaid_validator not in MERMAID_VALIDATORS:
            allowed = ", ".join(MERMAID_VALIDATORS)
            raise ValueError(
                f"Unsupported mermaid validator '{self.mermaid_validator}'. Allowed values: {allowed}"
            )
        self.prompts = PromptBuilder(FilePromptTemplateSet.from_name(self.prompt_name))

    @property
    def include_patterns(self) -> Optional[List[str]]:
        """Get file include patterns from agent instructions."""
        if self.agent_instructions:
            return self.agent_instructions.get("include_patterns")
        return None

    @property
    def exclude_patterns(self) -> Optional[List[str]]:
        """Get file exclude patterns from agent instructions."""
        if self.agent_instructions:
            return self.agent_instructions.get("exclude_patterns")
        return None

    @property
    def focus_modules(self) -> Optional[List[str]]:
        """Get focus modules from agent instructions."""
        if self.agent_instructions:
            return self.agent_instructions.get("focus_modules")
        return None

    @property
    def doc_type(self) -> Optional[str]:
        """Get documentation type from agent instructions."""
        if self.agent_instructions:
            return self.agent_instructions.get("doc_type")
        return None

    @property
    def custom_instructions(self) -> Optional[str]:
        """Get custom instructions from agent instructions."""
        if self.agent_instructions:
            return self.agent_instructions.get("custom_instructions")
        return None

    def get_prompt_addition(self) -> str:
        """Generate prompt additions based on agent instructions."""
        return self.prompts.build_prompt_addition(self.agent_instructions)

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "Config":
        """Create configuration from parsed arguments."""
        repo_name = os.path.basename(os.path.normpath(args.repo_path))
        sanitized_repo_name = "".join(c if c.isalnum() else "_" for c in repo_name)
        prompt_name = getattr(args, "prompt_name", DEFAULT_PROMPT_NAME)

        return cls(
            repo_path=args.repo_path,
            output_dir=OUTPUT_BASE_DIR,
            dependency_graph_dir=os.path.join(OUTPUT_BASE_DIR, DEPENDENCY_GRAPHS_DIR),
            docs_dir=os.path.join(OUTPUT_BASE_DIR, DOCS_DIR, f"{sanitized_repo_name}-docs"),
            max_depth=MAX_DEPTH,
            llm_base_url=LLM_BASE_URL,
            llm_api_key=LLM_API_KEY,
            main_model=MAIN_MODEL,
            cluster_model=CLUSTER_MODEL,
            fallback_model=FALLBACK_MODEL_1,
            mermaid_validator=DEFAULT_MERMAID_VALIDATOR,
            prompt_name=prompt_name,
            verbosity=0,
        )

    @classmethod
    def from_cli(
        cls,
        repo_path: str,
        output_dir: str,
        llm_base_url: str,
        llm_api_key: str,
        main_model: str,
        cluster_model: str,
        fallback_model: str = FALLBACK_MODEL_1,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        max_token_per_module: int = DEFAULT_MAX_TOKEN_PER_MODULE,
        max_token_per_leaf_module: int = DEFAULT_MAX_TOKEN_PER_LEAF_MODULE,
        mermaid_validator: str = DEFAULT_MERMAID_VALIDATOR,
        max_depth: int = MAX_DEPTH,
        agent_instructions: Optional[Dict[str, Any]] = None,
        prompt_name: str = DEFAULT_PROMPT_NAME,
        verbosity: int = 0,
        module_progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> "Config":
        """
        Create configuration for CLI context.

        Args:
            repo_path: Repository path
            output_dir: Output directory for generated docs
            llm_base_url: LLM API base URL
            llm_api_key: LLM API key
            main_model: Primary model
            cluster_model: Clustering model
            fallback_model: Fallback model
            max_tokens: Maximum tokens for LLM response
            max_token_per_module: Maximum tokens per module for clustering
            max_token_per_leaf_module: Maximum tokens per leaf module
            mermaid_validator: Mermaid validation backend (`mermaid_parser_py` or `mermaid_ink_api`)
            max_depth: Maximum depth for hierarchical decomposition
            agent_instructions: Custom agent instructions dict
            prompt_name: Prompt template set name
            verbosity: Verbosity level for CLI tracing and progress
            module_progress_callback: Optional callback for module progress events

        Returns:
            Config instance
        """
        os.path.basename(os.path.normpath(repo_path))
        base_output_dir = os.path.join(output_dir, "temp")

        return cls(
            repo_path=repo_path,
            output_dir=base_output_dir,
            dependency_graph_dir=os.path.join(base_output_dir, DEPENDENCY_GRAPHS_DIR),
            docs_dir=output_dir,
            max_depth=max_depth,
            llm_base_url=llm_base_url,
            llm_api_key=llm_api_key,
            main_model=main_model,
            cluster_model=cluster_model,
            fallback_model=fallback_model,
            max_tokens=max_tokens,
            max_token_per_module=max_token_per_module,
            max_token_per_leaf_module=max_token_per_leaf_module,
            mermaid_validator=mermaid_validator,
            agent_instructions=agent_instructions,
            prompt_name=prompt_name,
            verbosity=verbosity,
            module_progress_callback=module_progress_callback,
        )
