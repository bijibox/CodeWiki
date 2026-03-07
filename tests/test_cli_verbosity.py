import json
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from codewiki.cli.utils.progress import ModuleProgressBar, ProgressTracker
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
    assert "Phase 1/5: Dependency Analysis" in output
    assert "Parsing source files..." in output
    assert "Dependency Analysis complete" in output


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
        module_type="leaf",
        module_path="src/core",
        status="generated",
    )
    progress_bar.finish()

    output = capsys.readouterr().out
    assert "[1/2] leaf src/core... generated" in output


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
                choices=[SimpleNamespace(message=SimpleNamespace(content="response text"))]
            )

    class FakeClient:
        def __init__(self):
            self.chat = SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setattr("codewiki.src.be.llm_services.create_openai_client", lambda _: FakeClient())
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
    assert "LLM request: type=cluster_modules" in output
    assert "LLM response: duration_ms=250" in output
    assert "Explain this repository" not in output
    assert "response text" not in output
    assert len(artifacts) == 1
    artifact_content = artifacts[0].read_text(encoding="utf-8")
    assert artifacts[0].name.endswith("-cluster_modules.md")
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
                choices=[SimpleNamespace(message=SimpleNamespace(content="response text"))]
            )

    class FakeClient:
        def __init__(self):
            self.chat = SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setattr("codewiki.src.be.llm_services.create_openai_client", lambda _: FakeClient())
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
    assert "LLM request: type=cluster_modules" in output
    assert "LLM response: duration_ms=15" in output
    assert "===== LLM REQUEST =====" in output
    assert "===== LLM RESPONSE =====" in output
    assert "type: cluster_modules" in output
    assert "context: root" in output
    assert "model: cluster-model" in output
    assert "Explain this repository" in output
    assert "response text" in output


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

    monkeypatch.setattr("codewiki.src.be.agent_orchestrator.create_fallback_models", lambda _: "m")
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
    assert "LLM request: type=module_generation" in output
    assert "LLM response: duration_ms=500" in output
    assert "===== AGENT SYSTEM PROMPT =====" not in output
    assert "Generated documentation" not in output
    assert '"kind": "response"' not in output
    assert len(artifacts) == 1
    artifact_content = artifacts[0].read_text(encoding="utf-8")
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

    monkeypatch.setattr("codewiki.src.be.agent_orchestrator.create_fallback_models", lambda _: "m")
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
    assert "LLM request: type=module_generation" in output
    assert "LLM response: duration_ms=125" in output
    assert "===== AGENT SYSTEM PROMPT =====" in output
    assert "===== AGENT USER PROMPT =====" in output
    assert "===== AGENT RESULT =====" in output
    assert "===== AGENT MESSAGE HISTORY =====" in output
    assert "Generated documentation" in output
    assert "def foo() -> int:" in output
    assert '"kind": "response"' in output


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
