#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/watch_standard_self_play_status.sh http://127.0.0.1:8787
#
# Optional:
#   STATUS_INTERVAL_SECONDS=2 bash scripts/watch_standard_self_play_status.sh http://192.168.1.20:8787
#
# This is meant for a second terminal while the coordinator is running.
# It repeatedly polls the coordinator status endpoint and prints a compact
# summary so you can see leased/completed chunks and aggregate game counts.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

COORDINATOR_BASE_URL="${1:-${TCG_AI_STANDARD_SELF_PLAY_COORDINATOR_URL:-http://127.0.0.1:8787}}"
STATUS_INTERVAL_SECONDS="${STATUS_INTERVAL_SECONDS:-2}"
STATUS_URL="${COORDINATOR_BASE_URL%/}/api/standard-self-play/status"

while true; do
  clear
  echo "[status] $(date '+%Y-%m-%d %H:%M:%S')"
  echo "[status] ${STATUS_URL}"
  echo
  python3 - "$STATUS_URL" <<'PY'
import json
import sys
from urllib import error as urllib_error
from urllib import request as urllib_request

status_url = sys.argv[1]
try:
    with urllib_request.urlopen(status_url, timeout=10.0) as response:
        payload = json.loads(response.read().decode("utf-8"))
except (urllib_error.URLError, TimeoutError, OSError) as exc:
    print(f"Coordinator unavailable: {exc}")
    raise SystemExit(0)

aggregate = payload.get("aggregate", {})
workers = payload.get("workers", {})
deck_wins = aggregate.get("deck_wins", {})
recovery = payload.get("recovery", {})

print(f"run_id: {payload.get('run_id')}")
print(
    "tasks: "
    f"completed={payload.get('completed_tasks')} "
    f"leased={payload.get('leased_tasks')} "
    f"pending={payload.get('pending_tasks')} "
    f"total={payload.get('total_tasks')}"
)
print(
    "games: "
    f"{aggregate.get('games', 0)} "
    f"samples={aggregate.get('samples', 0)} "
    f"turns={aggregate.get('turns', 0)} "
    f"actions={aggregate.get('actions', 0)}"
)
print(
    "wins: "
    f"ampharos={deck_wins.get('ampharos-ex-battle-deck', 0)} "
    f"lucario={deck_wins.get('lucario-ex-battle-deck', 0)}"
)
print(
    "recovery: "
    f"completed_on_disk={recovery.get('completed_tasks_from_artifacts', 0)} "
    f"legacy_upgrades={recovery.get('upgraded_legacy_summaries', 0)} "
    f"issues={recovery.get('integrity_issue_count', 0)}"
)
print(f"run_complete: {payload.get('run_complete')}")
print()
print("workers:")
if not workers:
    print("  (none yet)")
else:
    for worker_id, worker_state in sorted(workers.items()):
        print(
            "  "
            f"{worker_id}: leased={worker_state.get('leased_task_index')} "
            f"submitted={worker_state.get('submitted_tasks', 0)} "
            f"last_seen={worker_state.get('last_seen_at')}"
        )
PY
  sleep "$STATUS_INTERVAL_SECONDS"
done
