"""Generate OpenAPI (Swagger) spec for the FastAPI app."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.api import app


def main() -> None:
    spec = app.openapi()
    out_path = PROJECT_ROOT / "openapi.json"
    out_path.write_text(json.dumps(spec, indent=2))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
