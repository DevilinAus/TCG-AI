#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.tcg_ai.game_modes.standard.ml.worker_launcher import main


if __name__ == "__main__":
    raise SystemExit(main())
