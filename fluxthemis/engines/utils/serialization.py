from __future__ import annotations

import dataclasses
import json
import os
from pathlib import Path
from typing import Any


def to_jsonable(value: Any) -> Any:
    """Convert common numeric/container objects into JSON-safe values."""
    if dataclasses.is_dataclass(value):
        return to_jsonable(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, 'item'):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    if hasattr(value, 'tolist'):
        try:
            return value.tolist()
        except (TypeError, ValueError):
            pass
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def write_json_atomic(path: str | Path, value: Any) -> None:
    """Write JSON via an adjacent temporary file and atomic rename."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + '.tmp')
    with temporary_path.open('w', encoding='utf-8') as file:
        json.dump(to_jsonable(value), file, indent=2, ensure_ascii=False)
        file.write('\n')
    os.replace(temporary_path, output_path)
