from typing import Any
from pathlib import Path
from types import SimpleNamespace

from click.testing import CliRunner
from pytest import MonkeyPatch

from codewiki.cli.commands.generate import generate_command
from codewiki.cli.models.config import AgentInstructions, Configuration


class _FakeConfigManager:
    def load(self) -> bool:
        return True

    def is_configured(self) -> bool:
        return True

    def get_config(self) -> Configuration:
        return Configuration(
            base_url="https://llm.example.com",
            main_model="main-model",
            cluster_model="cluster-model",
            fallback_model="fallback-model",
            agent_instructions=AgentInstructions(),
        )

    def get_api_key(self) -> str:
        return "secret"


def test_generate_command_accepts_prompt_name(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    captured: dict[str, Any] = {}

    class FakeGenerator:
        def __init__(
            self,
            repo_path: Path,
            output_dir: Path,
            config: dict[str, Any],
            verbosity: int = 0,
            generate_html: bool = False,
        ) -> None:
            captured["repo_path"] = repo_path
            captured["output_dir"] = output_dir
            captured["config"] = config
            captured["verbosity"] = verbosity

        def generate(self) -> SimpleNamespace:
            return SimpleNamespace(
                files_generated=["overview.md"],
                module_count=1,
                statistics=SimpleNamespace(total_files_analyzed=1, total_tokens_used=0),
            )

    monkeypatch.setattr("codewiki.cli.commands.generate.ConfigManager", _FakeConfigManager)
    monkeypatch.setattr(
        "codewiki.cli.commands.generate.validate_repository", lambda repo: (repo, [])
    )
    monkeypatch.setattr("codewiki.cli.commands.generate.check_writable_output", lambda path: None)
    monkeypatch.setattr("codewiki.cli.commands.generate.is_git_repository", lambda repo: False)
    monkeypatch.setattr("codewiki.cli.commands.generate.get_git_commit_hash", lambda repo: None)
    monkeypatch.setattr("codewiki.cli.commands.generate.get_git_branch", lambda repo: None)
    monkeypatch.setattr(
        "codewiki.cli.commands.generate.display_post_generation_instructions",
        lambda **kwargs: None,
    )
    monkeypatch.setattr("codewiki.cli.commands.generate.CLIDocumentationGenerator", FakeGenerator)

    runner = CliRunner()
    output_dir = tmp_path / "docs"
    result = runner.invoke(generate_command, ["--prompt-name", "en", "--output", str(output_dir)])

    assert result.exit_code == 0, result.output
    assert captured["config"]["prompt_name"] == "en"
    assert Path(captured["output_dir"]) == output_dir.resolve()
    assert captured["verbosity"] == 0


def test_generate_command_passes_verbosity_levels(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    captured: dict[str, Any] = {}

    class FakeGenerator:
        def __init__(
            self,
            repo_path: Path,
            output_dir: Path,
            config: dict[str, Any],
            verbosity: int = 0,
            generate_html: bool = False,
        ) -> None:
            captured["verbosity"] = verbosity
            captured["output_dir"] = output_dir

        def generate(self) -> SimpleNamespace:
            return SimpleNamespace(
                files_generated=["overview.md"],
                module_count=1,
                statistics=SimpleNamespace(total_files_analyzed=1, total_tokens_used=0),
            )

    monkeypatch.setattr("codewiki.cli.commands.generate.ConfigManager", _FakeConfigManager)
    monkeypatch.setattr(
        "codewiki.cli.commands.generate.validate_repository", lambda repo: (repo, [])
    )
    monkeypatch.setattr("codewiki.cli.commands.generate.check_writable_output", lambda path: None)
    monkeypatch.setattr("codewiki.cli.commands.generate.is_git_repository", lambda repo: False)
    monkeypatch.setattr("codewiki.cli.commands.generate.get_git_commit_hash", lambda repo: None)
    monkeypatch.setattr("codewiki.cli.commands.generate.get_git_branch", lambda repo: None)
    monkeypatch.setattr(
        "codewiki.cli.commands.generate.display_post_generation_instructions",
        lambda **kwargs: None,
    )
    monkeypatch.setattr("codewiki.cli.commands.generate.CLIDocumentationGenerator", FakeGenerator)

    runner = CliRunner()
    output_dir = tmp_path / "docs"
    result = runner.invoke(generate_command, ["-vvv", "--output", str(output_dir)])

    assert result.exit_code == 0, result.output
    assert captured["verbosity"] == 3


def test_generate_command_caps_verbosity_at_three(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    captured: dict[str, Any] = {}

    class FakeGenerator:
        def __init__(
            self,
            repo_path: Path,
            output_dir: Path,
            config: dict[str, Any],
            verbosity: int = 0,
            generate_html: bool = False,
        ) -> None:
            captured["verbosity"] = verbosity

        def generate(self) -> SimpleNamespace:
            return SimpleNamespace(
                files_generated=["overview.md"],
                module_count=1,
                statistics=SimpleNamespace(total_files_analyzed=1, total_tokens_used=0),
            )

    monkeypatch.setattr("codewiki.cli.commands.generate.ConfigManager", _FakeConfigManager)
    monkeypatch.setattr(
        "codewiki.cli.commands.generate.validate_repository", lambda repo: (repo, [])
    )
    monkeypatch.setattr("codewiki.cli.commands.generate.check_writable_output", lambda path: None)
    monkeypatch.setattr("codewiki.cli.commands.generate.is_git_repository", lambda repo: False)
    monkeypatch.setattr("codewiki.cli.commands.generate.get_git_commit_hash", lambda repo: None)
    monkeypatch.setattr("codewiki.cli.commands.generate.get_git_branch", lambda repo: None)
    monkeypatch.setattr(
        "codewiki.cli.commands.generate.display_post_generation_instructions",
        lambda **kwargs: None,
    )
    monkeypatch.setattr("codewiki.cli.commands.generate.CLIDocumentationGenerator", FakeGenerator)

    runner = CliRunner()
    result = runner.invoke(generate_command, ["-vvvv"])

    assert result.exit_code == 0, result.output
    assert captured["verbosity"] == 3


def test_generate_command_rejects_unknown_prompt_name():
    runner = CliRunner()

    result = runner.invoke(generate_command, ["--prompt-name", "missing"])

    assert result.exit_code != 0
    assert "Unknown prompt set 'missing'" in result.output
    assert "Available prompt sets: en" in result.output
