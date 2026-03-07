from codewiki.src.utils import FileManager


def test_file_manager_creates_directory_and_roundtrips_json(tmp_path):
    target_dir = tmp_path / "nested" / "dir"

    FileManager.ensure_directory(str(target_dir))
    assert target_dir.exists()

    payload = {"name": "codewiki", "items": [1, 2, 3]}
    json_path = target_dir / "data.json"
    FileManager.save_json(payload, str(json_path))

    assert FileManager.load_json(str(json_path)) == payload


def test_file_manager_load_json_returns_none_for_missing_file(tmp_path):
    missing_file = tmp_path / "missing.json"

    assert FileManager.load_json(str(missing_file)) is None


def test_file_manager_roundtrips_text(tmp_path):
    text_path = tmp_path / "note.txt"
    content = "CodeWiki test content"

    FileManager.save_text(content, str(text_path))

    assert FileManager.load_text(str(text_path)) == content
