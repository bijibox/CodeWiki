from types import SimpleNamespace

import pytest

import codewiki.src.be.utils as be_utils


def test_is_complex_module_returns_true_only_for_multiple_files():
    components = {
        "a": SimpleNamespace(file_path="src/a.py"),
        "b": SimpleNamespace(file_path="src/a.py"),
        "c": SimpleNamespace(file_path="src/c.py"),
    }

    assert be_utils.is_complex_module(components, ["a", "b"]) is False
    assert be_utils.is_complex_module(components, ["a", "c"]) is True


def test_count_tokens_uses_encoder(monkeypatch):
    class FakeEncoder:
        def encode(self, text):
            return text.split()

    monkeypatch.setattr(be_utils, "enc", FakeEncoder())

    assert be_utils.count_tokens("one two three") == 3


def test_extract_mermaid_blocks_returns_line_numbers_and_content():
    content = "\n".join(
        [
            "# Title",
            "```mermaid",
            "graph TD",
            "A --> B",
            "```",
            "Text",
            "```mermaid title=Flow",
            "sequenceDiagram",
            "Alice->>Bob: Hi",
            "```",
            "```mermaid",
            "```",
        ]
    )

    blocks = be_utils.extract_mermaid_blocks(content)

    assert blocks == [
        (2, "graph TD\nA --> B"),
        (7, "sequenceDiagram\nAlice->>Bob: Hi"),
    ]


@pytest.mark.asyncio
async def test_validate_mermaid_diagrams_handles_missing_file(tmp_path):
    missing = tmp_path / "missing.md"

    result = await be_utils.validate_mermaid_diagrams(str(missing), "missing.md")

    assert result == f"Error: File '{missing}' does not exist"


@pytest.mark.asyncio
async def test_validate_mermaid_diagrams_reports_no_blocks(tmp_path):
    md_file = tmp_path / "docs.md"
    md_file.write_text("# No diagrams here", encoding="utf-8")

    result = await be_utils.validate_mermaid_diagrams(str(md_file), "docs.md")

    assert result == "No mermaid diagrams found in the file"


@pytest.mark.asyncio
async def test_validate_mermaid_diagrams_reports_success(monkeypatch, tmp_path):
    md_file = tmp_path / "docs.md"
    md_file.write_text("```mermaid\ngraph TD\nA --> B\n```", encoding="utf-8")

    async def fake_validate_single_diagram(diagram_content, diagram_num, line_start):
        assert diagram_content == "graph TD\nA --> B"
        assert diagram_num == 1
        assert line_start == 1
        return ""

    monkeypatch.setattr(be_utils, "validate_single_diagram", fake_validate_single_diagram)

    result = await be_utils.validate_mermaid_diagrams(str(md_file), "docs.md")

    assert result == "All mermaid diagrams in file: docs.md are syntax correct"


@pytest.mark.asyncio
async def test_validate_mermaid_diagrams_collects_errors(monkeypatch, tmp_path):
    md_file = tmp_path / "docs.md"
    md_file.write_text("```mermaid\ngraph TD\nA --> B\n```", encoding="utf-8")

    async def fake_validate_single_diagram(diagram_content, diagram_num, line_start):
        return "Diagram 1: Parse error on line 3"

    monkeypatch.setattr(be_utils, "validate_single_diagram", fake_validate_single_diagram)

    result = await be_utils.validate_mermaid_diagrams(str(md_file), "docs.md")

    assert result.startswith("Mermaid syntax errors found in file: docs.md")
    assert "Diagram 1: Parse error on line 3" in result
