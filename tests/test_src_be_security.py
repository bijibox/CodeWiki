import pytest

from codewiki.src.be.dependency_analyzer.utils.security import (
    _inside,
    assert_safe_path,
    safe_open_text,
)


def test_inside_returns_true_for_nested_path_and_false_for_escape(tmp_path):
    nested = tmp_path / "nested" / "file.txt"
    nested.parent.mkdir(parents=True)
    nested.write_text("ok", encoding="utf-8")

    assert _inside(tmp_path, nested) is True
    assert _inside(tmp_path, tmp_path.parent / "escape.txt") is False


def test_assert_safe_path_rejects_escape_outside_base_dir(tmp_path):
    outside = tmp_path.parent / "outside.txt"

    with pytest.raises(PermissionError, match="Path escapes repo"):
        assert_safe_path(tmp_path, outside)


def test_assert_safe_path_rejects_symlinks(tmp_path):
    target = tmp_path / "target.txt"
    target.write_text("data", encoding="utf-8")
    link = tmp_path / "link.txt"

    try:
        link.symlink_to(target)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"Symlinks are not supported in this environment: {exc}")

    with pytest.raises(PermissionError, match="Symlink blocked"):
        assert_safe_path(tmp_path, link)


def test_safe_open_text_reads_file_contents(tmp_path):
    file_path = tmp_path / "docs.md"
    file_path.write_text("hello", encoding="utf-8")

    assert safe_open_text(tmp_path, file_path) == "hello"
