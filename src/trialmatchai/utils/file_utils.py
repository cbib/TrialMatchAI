import json
import os
from typing import Dict, List


def read_json_file(file_path: str) -> Dict:
    """Read a JSON file and return its contents."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        raise ValueError(f"Failed to read {file_path}: {str(e)}")


def write_json_file(data: Dict, file_path: str):
    """Write data to a JSON file."""
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
    except Exception as e:
        raise ValueError(f"Failed to write {file_path}: {str(e)}")


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
