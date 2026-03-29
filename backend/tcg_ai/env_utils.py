from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ENV_FILENAMES = (".env", ".env.local")


def project_env_paths(project_root: Path | None = None) -> tuple[Path, ...]:
    root = project_root or PROJECT_ROOT
    return tuple(root / filename for filename in PROJECT_ENV_FILENAMES)


def parse_env_line(line: str) -> tuple[str, str] | None:
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None
    if stripped.startswith("export "):
        stripped = stripped[len("export ") :].lstrip()
    if "=" not in stripped:
        return None
    key, value = stripped.split("=", 1)
    key = key.strip()
    value = value.strip()
    if not key:
        return None
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        value = value[1:-1]
    return key, value


def load_project_env(
    *,
    prefixes: tuple[str, ...] = ("TCG_AI_",),
    project_root: Path | None = None,
    override: bool = True,
) -> dict[str, str]:
    env_sources: dict[str, str] = {}
    for env_path in project_env_paths(project_root):
        if not env_path.exists() or not env_path.is_file():
            continue
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            parsed = parse_env_line(raw_line)
            if parsed is None:
                continue
            key, value = parsed
            if prefixes and not key.startswith(prefixes):
                continue
            if override or key not in os.environ:
                os.environ[key] = value
                env_sources[key] = str(env_path)
    return env_sources
