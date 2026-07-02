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
    """Atomically write JSON (temp + fsync + os.replace) so a crash never leaves a partial file."""
    try:
        path = str(file_path)
        directory = os.path.dirname(path) or "."
        os.makedirs(directory, exist_ok=True)
        # Suffix must NOT end in .json, else an orphaned temp file would match resume globs.
        fd, tmp = tempfile.mkstemp(dir=directory, prefix=".tmp-", suffix=".json.part")
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
    """True iff the path exists and holds parseable JSON (so a partial marker re-runs)."""
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
    """Atomically write lines to a text file (see write_json_file)."""
    try:
        path = str(file_path)
        directory = os.path.dirname(path) or "."
        os.makedirs(directory, exist_ok=True)
        # Suffix must NOT end in .txt/.json, else an orphaned temp file matches a reader's glob.
        fd, tmp = tempfile.mkstemp(dir=directory, prefix=".tmp-", suffix=".txt.part")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))
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


def create_directory(path: str):
    """Create a directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)
