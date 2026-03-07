import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from codewiki.cli.utils.progress import ModuleProgressBar, ProgressTracker
from codewiki.src.be.agent_orchestrator import AgentOrchestrator
from codewiki.src.be.dependency_analyzer.models.core import Node
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


def test_call_llm_verbosity_three_traces_prompt_and_response(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
):
    config = _make_backend_config(tmp_path, verbosity=3)

    class FakeCompletions:
        def create(self, **kwargs):
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="response text"))]
            )

    class FakeClient:
        def __init__(self):
            self.chat = SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setattr("codewiki.src.be.llm_services.create_openai_client", lambda _: FakeClient())

    result = call_llm(
        "Explain this repository",
        config,
        model="cluster-model",
        trace_label="cluster_modules",
        trace_context="root",
    )

    output = capsys.readouterr().out
    assert result == "response text"
    assert "===== LLM REQUEST =====" in output
    assert "===== LLM RESPONSE =====" in output
    assert "label: cluster_modules" in output
    assert "context: root" in output
    assert "model: cluster-model" in output
    assert "Explain this repository" in output
    assert "response text" in output


@pytest.mark.asyncio
async def test_agent_orchestrator_verbosity_three_traces_prompts_and_messages(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
):
    source_file = tmp_path / "module.py"
    source_file.write_text("def foo() -> int:\n    return 1\n", encoding="utf-8")
    config = _make_backend_config(tmp_path, verbosity=3)

    monkeypatch.setattr("codewiki.src.be.agent_orchestrator.create_fallback_models", lambda _: "m")

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
    assert module_tree == {}
    assert status == "generated"
    assert "===== AGENT SYSTEM PROMPT =====" in output
    assert "===== AGENT USER PROMPT =====" in output
    assert "===== AGENT RESULT =====" in output
    assert "===== AGENT MESSAGE HISTORY =====" in output
    assert "Generated documentation" in output
    assert "def foo() -> int:" in output
    assert '"kind": "response"' in output
