import re
from types import SimpleNamespace

import pytest

from codewiki.src.be.prompt_template import (
    DEFAULT_PROMPT_NAME,
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
    for attr_name in PROMPT_FILE_MAP:
        assert getattr(template_set, attr_name)


@pytest.mark.parametrize("prompt_name", available_prompt_names())
def test_prompt_templates_keep_default_placeholders_and_tags(prompt_name):
    template_set = FilePromptTemplateSet.from_name(prompt_name)
    default_template_set = FilePromptTemplateSet.from_name(DEFAULT_PROMPT_NAME)

    for attr_name in PROMPT_FILE_MAP:
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
    assert "{repo_name}" not in repo_overview_prompt
    assert "{module_name}" not in module_overview_prompt
    assert "{files}" not in filter_folders_prompt


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
