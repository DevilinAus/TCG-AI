from __future__ import annotations

import json
from pathlib import Path
import threading
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[5]
DEFAULT_EXPERIENCE_DIR = PROJECT_ROOT / "standard_ml_data"


class StandardExperienceStore:
    def __init__(self, base_dir: Path | None = None) -> None:
        self.base_dir = base_dir or DEFAULT_EXPERIENCE_DIR
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def record_decision(self, request_payload: dict[str, Any], response_payload: dict[str, Any]) -> None:
        self._append_jsonl(
            "decisions.jsonl",
            {
                "request": request_payload,
                "response": response_payload,
            },
        )

    def record_outcome(self, outcome_payload: dict[str, Any]) -> None:
        self._append_jsonl("outcomes.jsonl", outcome_payload)

    def _append_jsonl(self, filename: str, payload: dict[str, Any]) -> None:
        line = json.dumps(payload, sort_keys=True)
        path = self.base_dir / filename
        with self._lock:
            with path.open("a", encoding="utf-8") as handle:
                handle.write(line)
                handle.write("\n")
