"""Generate Swagger (OpenAPI) YAML spec for the FastAPI app."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.api import app  # noqa: E402


def _yaml_scalar(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    # Quote all strings to keep output safe and predictable
    return json.dumps(str(value))


def _to_yaml(value: Any, indent: int = 0) -> str:
    pad = " " * indent
    if isinstance(value, dict):
        if not value:
            return f"{pad}{{}}"
        lines = []
        for key, val in value.items():
            key_str = _yaml_scalar(key)
            if isinstance(val, (dict, list)) and val:
                lines.append(f"{pad}{key_str}:")
                lines.append(_to_yaml(val, indent + 2))
            else:
                lines.append(f"{pad}{key_str}: {_to_yaml(val, 0).lstrip()}")
        return "\n".join(lines)
    if isinstance(value, list):
        if not value:
            return f"{pad}[]"
        lines = []
        for item in value:
            if isinstance(item, (dict, list)) and item:
                lines.append(f"{pad}-")
                lines.append(_to_yaml(item, indent + 2))
            else:
                lines.append(f"{pad}- {_to_yaml(item, 0).lstrip()}")
        return "\n".join(lines)
    return f"{pad}{_yaml_scalar(value)}"


def main() -> None:
    spec = app.openapi()
    out_path = PROJECT_ROOT / "swagger.yaml"
    out_path.write_text(_to_yaml(spec) + "\n")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
