from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
from pydantic_ai import RunContext
from pydantic_core import ValidationError

from codewiki.src.be.agent_tools.deps import CodeWikiDeps
from codewiki.src.be.agent_tools.str_replace_editor import (
    str_replace_editor,
    str_replace_editor_tool,
)
from codewiki.src.config import Config


def _make_config(tmp_path: Path) -> Config:
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir(exist_ok=True)
    return Config.from_cli(
        repo_path=str(tmp_path),
        output_dir=str(docs_dir),
        llm_base_url="https://llm.example.com",
        llm_api_key="secret",
        main_model="main-model",
        cluster_model="cluster-model",
        fallback_model="fallback-model",
    )


@pytest.mark.asyncio
async def test_str_replace_editor_defaults_working_dir_to_docs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = _make_config(tmp_path)
    docs_dir = Path(config.docs_dir)
    deps = CodeWikiDeps(
        absolute_docs_path=str(docs_dir),
        absolute_repo_path=str(tmp_path),
        registry={},
        components={},
        path_to_current_module=[],
        current_module_name="module",
        module_tree={},
        max_depth=2,
        current_depth=1,
        config=config,
    )
    ctx = cast(RunContext[CodeWikiDeps], SimpleNamespace(deps=deps))

    async def _fake_validate_mermaid_diagrams(
        absolute_path: str,
        path: str,
        mermaid_validator: str,
    ) -> str:
        assert mermaid_validator == config.mermaid_validator
        return "ok"

    monkeypatch.setattr(
        "codewiki.src.be.agent_tools.str_replace_editor.validate_mermaid_diagrams",
        _fake_validate_mermaid_diagrams,
    )

    await str_replace_editor(
        ctx,
        command="create",
        path="module.md",
        file_text="# Module\n",
    )

    assert (docs_dir / "module.md").read_text(encoding="utf-8") == "# Module\n"


def test_str_replace_editor_schema_accepts_array_view_range() -> None:
    validated = str_replace_editor_tool.function_schema.validator.validate_json(
        '{"command":"view","path":"module.md","view_range":[1,50]}'
    )

    assert validated["view_range"] == [1, 50]


def test_str_replace_editor_schema_accepts_json_string_view_range() -> None:
    validated = str_replace_editor_tool.function_schema.validator.validate_json(
        '{"command":"view","path":"module.md","view_range":"[1, 50]"}'
    )

    assert validated["view_range"] == [1, 50]


def test_str_replace_editor_schema_rejects_invalid_json_string_view_range() -> None:
    with pytest.raises(ValidationError) as exc_info:
        str_replace_editor_tool.function_schema.validator.validate_json(
            '{"command":"view","path":"module.md","view_range":"oops"}'
        )

    assert "view_range must be a JSON array string like [1, 50]" in str(exc_info.value)
