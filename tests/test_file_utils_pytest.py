from Matcher.utils.file_utils import read_json_file, read_text_file, write_json_file, write_text_file


def test_write_and_read_json(tmp_path):
    path = tmp_path / "data.json"
    payload = {"a": 1, "b": "x"}
    write_json_file(payload, str(path))
    assert read_json_file(str(path)) == payload


def test_write_and_read_text(tmp_path):
    path = tmp_path / "data.txt"
    lines = ["a", "b", "c"]
    write_text_file(lines, str(path))
    assert read_text_file(str(path)) == lines
