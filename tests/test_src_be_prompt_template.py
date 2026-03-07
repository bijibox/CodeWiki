import re
from types import SimpleNamespace

import pytest

from codewiki.src.be.prompt_template import (
    DEFAULT_PROMPT_NAME,
    PROMPT_ADDITION_FILE_MAP,
    PROMPT_FILE_MAP,
    FilePromptTemplateSet,
    PromptBuilder,
    PromptTemplateError,
    available_prompt_names,
)

PLACEHOLDER_PATTERN = re.compile(r"\{[a-z_]+\}")
TAG_PATTERN = re.compile(r"</?[A-Za-z0-9_]+>")


def _extract_placeholders(template: str) -> list[str]:
    return PLACEHOLDER_PATTERN.findall(template)


def _extract_tags(template: str) -> list[str]:
    return TAG_PATTERN.findall(template)


def test_available_prompt_names_includes_default_prompt():
    assert DEFAULT_PROMPT_NAME in available_prompt_names()


@pytest.mark.parametrize("prompt_name", available_prompt_names())
def test_file_prompt_template_set_loads_every_prompt_set(prompt_name):
    template_set = FilePromptTemplateSet.from_name(prompt_name)

    assert template_set.prompt_name == prompt_name
    for attr_name in {**PROMPT_FILE_MAP, **PROMPT_ADDITION_FILE_MAP}:
        assert getattr(template_set, attr_name)


@pytest.mark.parametrize("prompt_name", available_prompt_names())
def test_prompt_templates_keep_default_placeholders_and_tags(prompt_name):
    template_set = FilePromptTemplateSet.from_name(prompt_name)
    default_template_set = FilePromptTemplateSet.from_name(DEFAULT_PROMPT_NAME)

    for attr_name in {**PROMPT_FILE_MAP, **PROMPT_ADDITION_FILE_MAP}:
        template = getattr(template_set, attr_name)
        default_template = getattr(default_template_set, attr_name)

        assert _extract_placeholders(template) == _extract_placeholders(default_template)
        assert _extract_tags(template) == _extract_tags(default_template)


@pytest.mark.parametrize("prompt_name", available_prompt_names())
def test_prompt_builder_builds_runtime_prompts_for_every_prompt_set(tmp_path, prompt_name):
    template_set = FilePromptTemplateSet.from_name(prompt_name)
    prompts = PromptBuilder(template_set)
    source_file = tmp_path / "module.py"
    source_file.write_text("def handler():\n    return 'ok'\n", encoding="utf-8")
    components = {
        "handler": SimpleNamespace(relative_path="src/module.py", file_path=str(source_file))
    }
    module_tree = {"module": {"components": ["handler"], "children": {}}}

    system_prompt = prompts.build_system_prompt("module", "Follow API style.")
    leaf_prompt = prompts.build_leaf_system_prompt("module")
    cluster_prompt = prompts.build_cluster_prompt("component_a", module_tree, "module")
    user_prompt = prompts.build_user_prompt("module", ["handler"], components, module_tree)
    repo_overview_prompt = prompts.build_repo_overview_prompt("repo", '{"a": 1}')
    module_overview_prompt = prompts.build_module_overview_prompt("module", '{"b": 2}')
    filter_folders_prompt = prompts.build_filter_folders_prompt("repo", "src/module.py")
    prompt_addition = prompts.build_prompt_addition(
        {
            "doc_type": "architecture",
            "focus_modules": ["src/module.py"],
            "custom_instructions": "Follow API style.",
        }
    )

    assert prompts.prompt_name == prompt_name
    assert "<CUSTOM_INSTRUCTIONS>" in system_prompt
    assert "Follow API style." in system_prompt
    assert "{custom_instructions}" not in leaf_prompt
    assert "module (current module)" in cluster_prompt
    assert "# File: src/module.py" in user_prompt
    assert "def handler()" in user_prompt
    assert '{"a": 1}' in repo_overview_prompt
    assert '{"b": 2}' in module_overview_prompt
    assert "src/module.py" in filter_folders_prompt
    assert "src/module.py" in prompt_addition
    assert "{repo_name}" not in repo_overview_prompt
    assert "{module_name}" not in module_overview_prompt
    assert "{files}" not in filter_folders_prompt


def test_prompt_builder_localizes_prompt_additions_for_english_prompt_set():
    prompts = PromptBuilder(FilePromptTemplateSet.from_name("en"))

    prompt_addition = prompts.build_prompt_addition(
        {
            "doc_type": "api",
            "focus_modules": ["src/core", "src/api"],
            "custom_instructions": "Mention public APIs.",
        }
    )

    assert "Apply the following additional instructions:" in prompt_addition
    assert "Focus on API documentation" in prompt_addition
    assert "src/core, src/api" in prompt_addition
    assert "Mention public APIs." in prompt_addition


def test_prompt_builder_localizes_prompt_additions_for_russian_prompt_set():
    prompts = PromptBuilder(FilePromptTemplateSet.from_name("ru"))

    prompt_addition = prompts.build_prompt_addition(
        {
            "doc_type": "architecture",
            "focus_modules": ["src/core", "src/api"],
            "custom_instructions": "Упомяни публичные API.",
        }
    )

    assert "Соблюдайте следующие дополнительные указания:" in prompt_addition
    assert "Сфокусируйтесь на архитектурной документации" in prompt_addition
    assert "src/core, src/api" in prompt_addition
    assert "Упомяни публичные API." in prompt_addition


@pytest.mark.parametrize("prompt_name", available_prompt_names())
def test_prompt_builder_uses_generic_doc_type_addition_for_unknown_doc_type(prompt_name):
    prompts = PromptBuilder(FilePromptTemplateSet.from_name(prompt_name))

    prompt_addition = prompts.build_prompt_addition({"doc_type": "internal-runbook"})

    assert "internal-runbook" in prompt_addition


def test_russian_prompts_explicitly_require_russian_documentation_language():
    template_set = FilePromptTemplateSet.from_name("ru")

    assert "Пишите весь естественный язык документации на русском языке" in (
        template_set.system_prompt
    )
    assert "Пишите весь естественный язык документации на русском языке" in (
        template_set.leaf_system_prompt
    )
    assert "Итоговую документацию пишите по-русски" in template_set.user_prompt


def test_file_prompt_template_set_rejects_unknown_prompt_set():
    with pytest.raises(PromptTemplateError) as exc_info:
        FilePromptTemplateSet.from_name("missing")

    assert "Unknown prompt set 'missing'" in str(exc_info.value)
    assert "en" in str(exc_info.value)


def test_file_prompt_template_set_rejects_missing_required_files(tmp_path):
    prompt_dir = tmp_path / "broken"
    prompt_dir.mkdir()
    (prompt_dir / "system_prompt.md").write_text("system", encoding="utf-8")

    with pytest.raises(PromptTemplateError) as exc_info:
        FilePromptTemplateSet.from_directory("broken", prompt_dir)

    assert "missing required prompt files" in str(exc_info.value)
    assert "leaf_system_prompt.md" in str(exc_info.value)
