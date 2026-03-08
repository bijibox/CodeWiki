import json
from pathlib import Path

from codewiki.cli.html_generator import HTMLGenerator


def test_html_generator_embeds_markdown_documents_for_offline_viewing(tmp_path: Path) -> None:
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()

    (docs_dir / "overview.md").write_text("# Overview\n\nHome page.\n", encoding="utf-8")
    (docs_dir / "CLI Module.md").write_text("# CLI Module\n\nDetails.\n", encoding="utf-8")
    (docs_dir / "module_tree.json").write_text(json.dumps({}), encoding="utf-8")
    (docs_dir / "metadata.json").write_text(json.dumps({"generation_info": {}}), encoding="utf-8")

    output_path = docs_dir / "index.html"

    HTMLGenerator().generate(
        output_path=output_path,
        title="CodeWiki",
        docs_dir=docs_dir,
    )

    html = output_path.read_text(encoding="utf-8")

    assert "const EMBEDDED_DOCS = {" in html
    assert '"overview.md": "# Overview\\n\\nHome page.\\n"' in html
    assert '"CLI Module.md": "# CLI Module\\n\\nDetails.\\n"' in html
    assert "const markdown = await getDocumentMarkdown(filename);" in html
    assert "Object.prototype.hasOwnProperty.call(EMBEDDED_DOCS, filename)" in html
