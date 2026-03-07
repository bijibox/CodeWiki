from types import SimpleNamespace

import pytest

from codewiki.src.be.prompt_template import (
    FilePromptTemplateSet,
    PromptBuilder,
    PromptTemplateError,
    available_prompt_names,
)


def test_available_prompt_names_includes_en():
    assert "en" in available_prompt_names()


def test_file_prompt_template_set_loads_en():
    template_set = FilePromptTemplateSet.from_name("en")

    assert template_set.prompt_name == "en"
    assert "AI documentation assistant" in template_set.system_prompt
    assert "Generate comprehensive documentation" in template_set.user_prompt
    assert "brief overview of the {repo_name} repository" in template_set.repo_overview_prompt
    assert "relative paths of files, folders" in template_set.filter_folders_prompt


def test_prompt_builder_builds_runtime_prompts(tmp_path):
    template_set = FilePromptTemplateSet.from_name("en")
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

    assert prompts.prompt_name == "en"
    assert "<CUSTOM_INSTRUCTIONS>" in system_prompt
    assert "Follow API style." in system_prompt
    assert "{custom_instructions}" not in leaf_prompt
    assert "module (current module)" in cluster_prompt
    assert "# File: src/module.py" in user_prompt
    assert "def handler()" in user_prompt
    assert "brief overview of the repo repository" in repo_overview_prompt
    assert "brief overview of `module` module" in module_overview_prompt


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
