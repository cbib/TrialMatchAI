import json
import os
import tempfile
from typing import Dict, List


def read_json_file(file_path: str) -> Dict:
    """Read a JSON file and return its contents."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        raise ValueError(f"Failed to read {file_path}: {str(e)}")


def write_json_file(data: Dict, file_path: str):
    """Atomically write data to a JSON file.

    Writes to a temp file in the same directory, fsyncs, then os.replace()s it
    into place — so a crash mid-write can never leave a truncated/partial file
    that resume logic would mistake for a completed artifact.
    """
    try:
        path = str(file_path)
        directory = os.path.dirname(path) or "."
        os.makedirs(directory, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=directory, prefix=".tmp-", suffix=".json")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)
        except Exception:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
    except Exception as e:
        raise ValueError(f"Failed to write {file_path}: {str(e)}")


def is_valid_json_file(file_path: str) -> bool:
    """True only if the path exists and contains parseable JSON.

    Used by resume gates so a present-but-corrupt/partial marker is treated as
    incomplete (re-run) rather than done.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            json.load(f)
        return True
    except Exception:
        return False


def read_text_file(file_path: str) -> List[str]:
    """Read lines from a text file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]
    except Exception as e:
        raise ValueError(f"Failed to read {file_path}: {str(e)}")


def write_text_file(lines: List[str], file_path: str):
    """Write lines to a text file."""
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
    except Exception as e:
        raise ValueError(f"Failed to write {file_path}: {str(e)}")


def create_directory(path: str):
    """Create a directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)
