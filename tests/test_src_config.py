from argparse import Namespace

from codewiki.src.config import Config, is_cli_context, set_cli_context
from codewiki.src.be.prompt_template import PromptBuilder


def test_cli_context_flag_can_be_toggled():
    set_cli_context(True)
    assert is_cli_context() is True

    set_cli_context(False)
    assert is_cli_context() is False


def test_config_properties_and_prompt_addition_are_derived_from_agent_instructions():
    config = Config(
        repo_path="/tmp/project",
        output_dir="output",
        dependency_graph_dir="output/dependency_graphs",
        docs_dir="output/docs/project-docs",
        max_depth=2,
        llm_base_url="https://example.com",
        llm_api_key="secret",
        main_model="main-model",
        cluster_model="cluster-model",
        agent_instructions={
            "include_patterns": ["*.py"],
            "exclude_patterns": ["tests"],
            "focus_modules": ["src/core", "src/api"],
            "doc_type": "architecture",
            "custom_instructions": "Mention public APIs.",
        },
    )

    assert config.include_patterns == ["*.py"]
    assert config.exclude_patterns == ["tests"]
    assert config.focus_modules == ["src/core", "src/api"]
    assert config.doc_type == "architecture"
    assert config.custom_instructions == "Mention public APIs."
    assert config.prompt_name == "en"
    assert isinstance(config.prompts, PromptBuilder)

    prompt_addition = config.get_prompt_addition()
    assert "Apply the following additional instructions:" in prompt_addition
    assert "Focus on architecture documentation" in prompt_addition
    assert "src/core, src/api" in prompt_addition
    assert "Mention public APIs." in prompt_addition


def test_config_get_prompt_addition_handles_unknown_doc_type():
    config = Config(
        repo_path="/tmp/project",
        output_dir="output",
        dependency_graph_dir="output/dependency_graphs",
        docs_dir="output/docs/project-docs",
        max_depth=2,
        llm_base_url="https://example.com",
        llm_api_key="secret",
        main_model="main-model",
        cluster_model="cluster-model",
        agent_instructions={"doc_type": "internal-runbook"},
    )

    assert config.get_prompt_addition() == (
        "Apply the following additional instructions:\n"
        "- Focus on generating internal-runbook documentation."
    )


def test_config_get_prompt_addition_uses_prompt_set_localization():
    config = Config(
        repo_path="/tmp/project",
        output_dir="output",
        dependency_graph_dir="output/dependency_graphs",
        docs_dir="output/docs/project-docs",
        max_depth=2,
        llm_base_url="https://example.com",
        llm_api_key="secret",
        main_model="main-model",
        cluster_model="cluster-model",
        agent_instructions={
            "doc_type": "developer",
            "focus_modules": ["src/core"],
            "custom_instructions": "Укажи ограничения.",
        },
        prompt_name="ru",
    )

    prompt_addition = config.get_prompt_addition()

    assert "Соблюдайте следующие дополнительные указания:" in prompt_addition
    assert "Сфокусируйтесь на документации для разработчиков" in prompt_addition
    assert "src/core" in prompt_addition
    assert "Укажи ограничения." in prompt_addition


def test_config_from_args_builds_default_output_paths():
    args = Namespace(repo_path="/repos/my-repo.name")

    config = Config.from_args(args)

    assert config.output_dir == "output"
    assert config.dependency_graph_dir == "output/dependency_graphs"
    assert config.docs_dir == "output/docs/my_repo_name-docs"
    assert config.mermaid_validator == "mermaid_parser_py"
    assert config.prompt_name == "en"
    assert isinstance(config.prompts, PromptBuilder)


def test_config_from_cli_uses_temp_output_and_runtime_overrides():
    config = Config.from_cli(
        repo_path="/repos/sample-project",
        output_dir="/tmp/generated-docs",
        llm_base_url="https://llm.example.com",
        llm_api_key="api-key",
        main_model="main-model",
        cluster_model="cluster-model",
        fallback_model="fallback-model",
        mermaid_validator="mermaid_ink_api",
        max_tokens=2048,
        max_token_per_module=4096,
        max_token_per_leaf_module=1024,
        max_depth=4,
        agent_instructions={"doc_type": "developer"},
        prompt_name="en",
    )

    assert config.output_dir == "/tmp/generated-docs/temp"
    assert config.dependency_graph_dir == "/tmp/generated-docs/temp/dependency_graphs"
    assert config.docs_dir == "/tmp/generated-docs"
    assert config.max_tokens == 2048
    assert config.max_token_per_module == 4096
    assert config.max_token_per_leaf_module == 1024
    assert config.mermaid_validator == "mermaid_ink_api"
    assert config.max_depth == 4
    assert config.doc_type == "developer"
    assert config.prompt_name == "en"
    assert isinstance(config.prompts, PromptBuilder)
