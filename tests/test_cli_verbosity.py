import json
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
from pydantic_ai import RunContext

from codewiki.cli.utils.progress import ModuleProgressBar, ProgressTracker
from codewiki.src.be.agent_tools.deps import CodeWikiDeps
from codewiki.src.be.agent_tools.generate_sub_module_documentations import (
    generate_sub_module_documentation,
)
from codewiki.src.be.agent_orchestrator import AgentOrchestrator
from codewiki.src.be.dependency_analyzer.models.core import Node
from codewiki.src.be.llm_logging import write_llm_markdown_artifact
from codewiki.src.be.llm_services import call_llm
from codewiki.src.config import Config


def _make_backend_config(tmp_path: Path, verbosity: int) -> Config:
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
        verbosity=verbosity,
    )


def test_progress_tracker_verbosity_one_shows_stage_details(capsys: pytest.CaptureFixture[str]):
    tracker = ProgressTracker(total_stages=5, verbosity=1)

    tracker.start_stage(1, "Dependency Analysis")
    tracker.update_stage(0.5, "Parsing source files...")
    tracker.complete_stage("done")

    output = capsys.readouterr().out
    assert "Stage 1/5  Dependency Analysis" in output
    assert "Parsing source files..." in output
    assert "Stage 1/5  Dependency Analysis  DONE" in output


def test_progress_tracker_elapsed_shows_tenths(monkeypatch: pytest.MonkeyPatch) -> None:
    time_values = iter([100.0, 100.3])
    monkeypatch.setattr("codewiki.cli.utils.progress.time.time", lambda: next(time_values))

    tracker = ProgressTracker(total_stages=5, verbosity=0)

    assert tracker._format_elapsed() == "00:00.3"


def test_module_progress_bar_verbosity_two_shows_extended_progress(
    capsys: pytest.CaptureFixture[str],
):
    progress_bar = ModuleProgressBar(total_modules=2, verbosity=2)

    progress_bar.update(
        "core",
        phase="started",
        index=1,
        total=2,
        module_type="leaf",
        module_path="src/core",
        status="started",
    )
    progress_bar.update(
        "core",
        phase="finished",
        index=1,
        total=2,
        module_type="leaf",
        module_path="src/core",
        status="generated",
        duration_seconds=0.5,
    )
    progress_bar.finish()

    output = capsys.readouterr().out
    assert "[01/02] leaf  src/core  START" in output
    assert "[01/02] leaf  src/core  DONE 0.5s" in output


def test_module_progress_bar_verbosity_two_renders_terminal_statuses(
    capsys: pytest.CaptureFixture[str],
):
    progress_bar = ModuleProgressBar(total_modules=3, verbosity=2)

    progress_bar.update(
        "cached",
        phase="finished",
        index=2,
        total=3,
        module_type="leaf",
        module_path="src/cached",
        status="cached",
    )
    progress_bar.update(
        "failed",
        phase="finished",
        index=3,
        total=3,
        module_type="parent",
        module_path="src/failed",
        status="failed",
    )

    output = capsys.readouterr().out
    assert "[02/03] leaf  src/cached  CACHED" in output
    assert "[03/03] parent  src/failed  FAILED" in output


def test_call_llm_verbosity_three_shows_summary_only_and_writes_artifact(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
):
    config = _make_backend_config(tmp_path, verbosity=3)
    perf_counter_values = iter([10.0, 10.25])

    class FakeCompletions:
        def create(self, **kwargs):
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content="response text", reasoning=None),
                        finish_reason="stop",
                    )
                ],
                usage=SimpleNamespace(prompt_tokens=3, completion_tokens=2, total_tokens=5),
            )

    class FakeClient:
        def __init__(self):
            self.chat = SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setattr("codewiki.src.be.llm_services.create_openai_client", lambda _: FakeClient())
    monkeypatch.setattr(
        "codewiki.src.be.llm_services.count_tokens",
        lambda text: {
            "Explain this repository": 3,
            "response text": 2,
            "Reasoning trace": 4,
        }[text],
    )
    monkeypatch.setattr(
        "codewiki.src.be.llm_services.time.perf_counter", lambda: next(perf_counter_values)
    )

    result = call_llm(
        "Explain this repository",
        config,
        model="cluster-model",
        prompt_type="cluster_modules",
        context="root",
    )

    output = capsys.readouterr().out
    artifacts = sorted((Path(config.docs_dir) / "temp" / "llm").glob("*.md"))
    assert result == "response text"
    assert "LLM REQ  cluster_modules  in=3" in output
    assert "LLM RES  cluster_modules  dur=0.250s  out=2  tps=8.000" in output
    assert "Explain this repository" not in output
    assert "response text" not in output
    assert len(artifacts) == 1
    artifact_content = artifacts[0].read_text(encoding="utf-8")
    assert artifacts[0].name.endswith("-cluster_modules.md")
    assert "- Duration: `0.250 s`" in artifact_content
    assert "- Request tokens: `3`" in artifact_content
    assert "- Response tokens: `2`" in artifact_content
    assert "- Response speed: `8.000 tokens/s`" in artifact_content
    assert "- Finish reason: `stop`" in artifact_content
    assert "- Content missing: `false`" in artifact_content
    assert "## Request" in artifact_content
    assert "## Response" in artifact_content
    assert "Explain this repository" in artifact_content
    assert "response text" in artifact_content


def test_call_llm_verbosity_four_shows_full_prompt_and_response(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
):
    config = _make_backend_config(tmp_path, verbosity=4)
    perf_counter_values = iter([20.0, 20.015])

    class FakeCompletions:
        def create(self, **kwargs):
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content="response text", reasoning=None),
                        finish_reason="stop",
                    )
                ],
                usage=SimpleNamespace(prompt_tokens=3, completion_tokens=2, total_tokens=5),
            )

    class FakeClient:
        def __init__(self):
            self.chat = SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setattr("codewiki.src.be.llm_services.create_openai_client", lambda _: FakeClient())
    monkeypatch.setattr(
        "codewiki.src.be.llm_services.count_tokens",
        lambda text: {
            "Explain this repository": 3,
            "response text": 2,
            "Reasoning trace": 4,
        }[text],
    )
    monkeypatch.setattr(
        "codewiki.src.be.llm_services.time.perf_counter", lambda: next(perf_counter_values)
    )

    call_llm(
        "Explain this repository",
        config,
        model="cluster-model",
        prompt_type="cluster_modules",
        context="root",
    )

    output = capsys.readouterr().out
    assert "LLM REQ  cluster_modules  in=3" in output
    assert "LLM RES  cluster_modules  dur=0.015s  out=2  tps=133.333" in output
    assert "===== LLM REQUEST =====" in output
    assert "===== LLM RESPONSE =====" in output
    assert "type: cluster_modules" in output
    assert "context: root" in output
    assert "model: cluster-model" in output
    assert "Explain this repository" in output
    assert "response text" in output


def test_call_llm_writes_reasoning_section_when_present(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    config = _make_backend_config(tmp_path, verbosity=3)
    perf_counter_values = iter([21.0, 21.5])

    class FakeCompletions:
        def create(self, **kwargs):
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content="response text",
                            reasoning="Reasoning trace",
                        ),
                        finish_reason="stop",
                    )
                ],
                usage=SimpleNamespace(prompt_tokens=3, completion_tokens=2, total_tokens=5),
            )

    class FakeClient:
        def __init__(self):
            self.chat = SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setattr("codewiki.src.be.llm_services.create_openai_client", lambda _: FakeClient())
    monkeypatch.setattr(
        "codewiki.src.be.llm_services.count_tokens",
        lambda text: {
            "Explain this repository": 3,
            "response text": 2,
            "Reasoning trace": 4,
        }[text],
    )
    monkeypatch.setattr(
        "codewiki.src.be.llm_services.time.perf_counter", lambda: next(perf_counter_values)
    )

    result = call_llm(
        "Explain this repository",
        config,
        model="cluster-model",
        prompt_type="cluster_modules",
        context="root",
    )

    artifacts = sorted((Path(config.docs_dir) / "temp" / "llm").glob("*.md"))
    assert result == "response text"
    assert len(artifacts) == 1
    artifact_content = artifacts[0].read_text(encoding="utf-8")
    assert "## Response" in artifact_content
    assert "## Reasoning" in artifact_content
    assert "Reasoning trace" in artifact_content
    assert '"reasoning_tokens_estimated": 4' in artifact_content


def test_call_llm_writes_failure_artifact_with_reasoning_and_raw_response(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    config = _make_backend_config(tmp_path, verbosity=3)
    perf_counter_values = iter([22.0, 22.125])

    class FakeCompletions:
        def create(self, **kwargs):
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content=None,
                            reasoning="Reasoning trace",
                        ),
                        finish_reason="length",
                    )
                ],
                usage=SimpleNamespace(prompt_tokens=3, completion_tokens=32, total_tokens=35),
            )

    class FakeClient:
        def __init__(self):
            self.chat = SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setattr("codewiki.src.be.llm_services.create_openai_client", lambda _: FakeClient())
    monkeypatch.setattr(
        "codewiki.src.be.llm_services.count_tokens",
        lambda text: {"Explain this repository": 3, "Reasoning trace": 4}[text],
    )
    monkeypatch.setattr(
        "codewiki.src.be.llm_services.time.perf_counter", lambda: next(perf_counter_values)
    )

    with pytest.raises(RuntimeError, match="LLM response did not include message content"):
        call_llm(
            "Explain this repository",
            config,
            model="cluster-model",
            prompt_type="cluster_modules",
            context="root",
        )

    artifacts = sorted((Path(config.docs_dir) / "temp" / "llm").glob("*.md"))
    assert len(artifacts) == 1
    artifact_content = artifacts[0].read_text(encoding="utf-8")
    assert "- Finish reason: `length`" in artifact_content
    assert "- Content missing: `true`" in artifact_content
    assert "## Reasoning" in artifact_content
    assert "Reasoning trace" in artifact_content
    assert "## Raw Response" in artifact_content
    assert '"content": null' in artifact_content
    assert '"reasoning_tokens_estimated": 4' in artifact_content
    assert "## Error" in artifact_content
    assert "LLM response did not include message content" in artifact_content


def test_call_llm_writes_failure_artifact_without_reasoning(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    config = _make_backend_config(tmp_path, verbosity=3)
    perf_counter_values = iter([23.0, 23.125])

    class FakeCompletions:
        def create(self, **kwargs):
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content=None),
                        finish_reason="length",
                    )
                ]
            )

    class FakeClient:
        def __init__(self):
            self.chat = SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setattr("codewiki.src.be.llm_services.create_openai_client", lambda _: FakeClient())
    monkeypatch.setattr("codewiki.src.be.llm_services.count_tokens", lambda text: 3)
    monkeypatch.setattr(
        "codewiki.src.be.llm_services.time.perf_counter", lambda: next(perf_counter_values)
    )

    with pytest.raises(RuntimeError, match="LLM response did not include message content"):
        call_llm(
            "Explain this repository",
            config,
            model="cluster-model",
            prompt_type="cluster_modules",
            context="root",
        )

    artifacts = sorted((Path(config.docs_dir) / "temp" / "llm").glob("*.md"))
    assert len(artifacts) == 1
    artifact_content = artifacts[0].read_text(encoding="utf-8")
    assert "## Reasoning" not in artifact_content
    assert "## Raw Response" in artifact_content
    assert "## Error" in artifact_content


@pytest.mark.asyncio
async def test_agent_orchestrator_verbosity_three_shows_summary_only_and_writes_artifact(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
):
    source_file = tmp_path / "module.py"
    source_file.write_text("def foo() -> int:\n    return 1\n", encoding="utf-8")
    config = _make_backend_config(tmp_path, verbosity=3)
    perf_counter_values = iter([30.0, 30.5])
    token_counts = iter([11, 7, 5])

    monkeypatch.setattr("codewiki.src.be.agent_orchestrator.create_fallback_models", lambda _: "m")
    monkeypatch.setattr(
        "codewiki.src.be.agent_orchestrator.count_tokens", lambda text: next(token_counts)
    )
    monkeypatch.setattr(
        "codewiki.src.be.agent_orchestrator.time.perf_counter", lambda: next(perf_counter_values)
    )

    class FakeAgent:
        def __class_getitem__(cls, item):
            return cls

        def __init__(self, model, name, deps_type, tools, system_prompt):
            self.system_prompt = system_prompt

        async def run(self, user_prompt, deps):
            return SimpleNamespace(
                output="Generated documentation",
                new_messages_json=lambda: json.dumps(
                    [{"kind": "response", "parts": [{"part_kind": "text", "content": "ok"}]}]
                ).encode("utf-8"),
            )

    monkeypatch.setattr("codewiki.src.be.agent_orchestrator.Agent", FakeAgent)

    orchestrator = AgentOrchestrator(config)
    components = {
        "foo": Node(
            id="foo",
            name="foo",
            component_type="function",
            file_path=str(source_file),
            relative_path="module.py",
        )
    }

    module_tree, status = await orchestrator.process_module(
        "module",
        components,
        ["foo"],
        [],
        str(config.docs_dir),
    )

    output = capsys.readouterr().out
    artifacts = sorted((Path(config.docs_dir) / "temp" / "llm").glob("*.md"))
    assert module_tree == {}
    assert status == "generated"
    assert "leaf  module  START" in output
    assert "leaf  module  DONE 0.5s" in output
    assert "LLM REQ  module_generation  in=18" in output
    assert "LLM RES  module_generation  dur=0.500s  out=5  tps=10.000" in output
    assert "===== AGENT SYSTEM PROMPT =====" not in output
    assert "Generated documentation" not in output
    assert '"kind": "response"' not in output
    assert "Processing module:" not in output
    assert "Successfully processed module:" not in output
    assert len(artifacts) == 1
    artifact_content = artifacts[0].read_text(encoding="utf-8")
    assert "- Duration: `0.500 s`" in artifact_content
    assert "- Request tokens: `18`" in artifact_content
    assert "- Response tokens: `5`" in artifact_content
    assert "- Response speed: `10.000 tokens/s`" in artifact_content
    assert "## System Prompt" in artifact_content
    assert "## User Prompt" in artifact_content
    assert "## Result" in artifact_content
    assert "## Message History" in artifact_content
    assert "Generated documentation" in artifact_content
    assert '"kind": "response"' in artifact_content


@pytest.mark.asyncio
async def test_agent_orchestrator_verbosity_four_shows_prompts_and_messages(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
):
    source_file = tmp_path / "module.py"
    source_file.write_text("def foo() -> int:\n    return 1\n", encoding="utf-8")
    config = _make_backend_config(tmp_path, verbosity=4)
    perf_counter_values = iter([40.0, 40.125])
    token_counts = iter([11, 7, 5])

    monkeypatch.setattr("codewiki.src.be.agent_orchestrator.create_fallback_models", lambda _: "m")
    monkeypatch.setattr(
        "codewiki.src.be.agent_orchestrator.count_tokens", lambda text: next(token_counts)
    )
    monkeypatch.setattr(
        "codewiki.src.be.agent_orchestrator.time.perf_counter", lambda: next(perf_counter_values)
    )

    class FakeAgent:
        def __class_getitem__(cls, item):
            return cls

        def __init__(self, model, name, deps_type, tools, system_prompt):
            self.system_prompt = system_prompt

        async def run(self, user_prompt, deps):
            return SimpleNamespace(
                output="Generated documentation",
                new_messages_json=lambda: json.dumps(
                    [{"kind": "response", "parts": [{"part_kind": "text", "content": "ok"}]}]
                ).encode("utf-8"),
            )

    monkeypatch.setattr("codewiki.src.be.agent_orchestrator.Agent", FakeAgent)

    orchestrator = AgentOrchestrator(config)
    components = {
        "foo": Node(
            id="foo",
            name="foo",
            component_type="function",
            file_path=str(source_file),
            relative_path="module.py",
        )
    }

    await orchestrator.process_module(
        "module",
        components,
        ["foo"],
        [],
        str(config.docs_dir),
    )

    output = capsys.readouterr().out
    assert "leaf  module  START" in output
    assert "leaf  module  DONE 0.1s" in output
    assert "LLM REQ  module_generation  in=18" in output
    assert "LLM RES  module_generation  dur=0.125s  out=5  tps=40.000" in output
    assert "===== AGENT SYSTEM PROMPT =====" in output
    assert "===== AGENT USER PROMPT =====" in output
    assert "===== AGENT RESULT =====" in output
    assert "===== AGENT MESSAGE HISTORY =====" in output
    assert "Generated documentation" in output
    assert "def foo() -> int:" in output
    assert '"kind": "response"' in output


@pytest.mark.asyncio
async def test_generate_sub_module_documentation_verbosity_three_shows_summary_only(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
):
    source_file = tmp_path / "module.py"
    source_file.write_text("def foo() -> int:\n    return 1\n", encoding="utf-8")
    config = _make_backend_config(tmp_path, verbosity=3)
    perf_counter_values = iter([50.0, 50.25])
    token_counts = iter([1, 11, 7, 5])

    monkeypatch.setattr(
        "codewiki.src.be.agent_tools.generate_sub_module_documentations.create_fallback_models",
        lambda _: "m",
    )
    monkeypatch.setattr(
        "codewiki.src.be.agent_tools.generate_sub_module_documentations.is_complex_module",
        lambda *args, **kwargs: False,
    )
    monkeypatch.setattr(
        "codewiki.src.be.agent_tools.generate_sub_module_documentations.count_tokens",
        lambda text: next(token_counts),
    )
    monkeypatch.setattr(
        "codewiki.src.be.agent_tools.generate_sub_module_documentations.time.perf_counter",
        lambda: next(perf_counter_values),
    )

    class FakeAgent:
        def __class_getitem__(cls, item):
            return cls

        def __init__(self, model, name, deps_type, system_prompt, tools):
            self.system_prompt = system_prompt

        async def run(self, user_prompt, deps):
            return SimpleNamespace(
                output="Generated sub-module documentation",
                new_messages_json=lambda: json.dumps(
                    [{"kind": "response", "parts": [{"part_kind": "text", "content": "ok"}]}]
                ).encode("utf-8"),
            )

    monkeypatch.setattr(
        "codewiki.src.be.agent_tools.generate_sub_module_documentations.Agent", FakeAgent
    )

    components = {
        "foo": Node(
            id="foo",
            name="foo",
            component_type="function",
            file_path=str(source_file),
            relative_path="module.py",
        )
    }
    deps = SimpleNamespace(
        config=config,
        current_module_name="module",
        path_to_current_module=[],
        current_depth=0,
        max_depth=config.max_depth,
        custom_instructions=None,
        module_tree={},
        components=components,
    )
    ctx = cast(RunContext[CodeWikiDeps], SimpleNamespace(deps=deps))

    result = await generate_sub_module_documentation(ctx, {"child_docs": ["foo"]})

    output = capsys.readouterr().out
    artifacts = sorted((Path(config.docs_dir) / "temp" / "llm").glob("*.md"))
    assert "Generate successfully." in result
    assert "submodule  child_docs  START" in output
    assert "submodule  child_docs  DONE 0.2s" in output
    assert "LLM REQ  sub_module_generation  in=18" in output
    assert "LLM RES  sub_module_generation  dur=0.250s  out=5  tps=20.000" in output
    assert "Generated sub-module documentation" not in output
    assert "Generating documentation for sub-module:" not in output
    assert len(artifacts) == 1
    artifact_content = artifacts[0].read_text(encoding="utf-8")
    assert "- Duration: `0.250 s`" in artifact_content
    assert "- Request tokens: `18`" in artifact_content
    assert "- Response tokens: `5`" in artifact_content
    assert "- Response speed: `20.000 tokens/s`" in artifact_content


def test_write_llm_markdown_artifact_avoids_collisions(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    config = _make_backend_config(tmp_path, verbosity=0)
    fixed_now = datetime(2026, 3, 7, 12, 34, 56, 789000)

    class FixedDateTime:
        @classmethod
        def now(cls):
            return fixed_now

    monkeypatch.setattr("codewiki.src.be.llm_logging.datetime", FixedDateTime)

    first_path = write_llm_markdown_artifact(
        config,
        prompt_type="Repo Overview",
        sections=(("Request", "first", "text"),),
    )
    second_path = write_llm_markdown_artifact(
        config,
        prompt_type="Repo Overview",
        sections=(("Request", "second", "text"),),
    )

    assert first_path.name == "2026-03-07_12-34-56.789-repo-overview.md"
    assert second_path.name == "2026-03-07_12-34-56.789-repo-overview-2.md"
    assert first_path.read_text(encoding="utf-8")
    assert second_path.read_text(encoding="utf-8")
