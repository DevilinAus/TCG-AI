from __future__ import annotations

import logging
import os
from pathlib import Path

_LOGGER_NAME = "tcg_ai"
_LOG_FORMAT = "%(asctime)s.%(msecs)03d %(levelname)s [%(name)s] %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_LOG_DIR = _PROJECT_ROOT / "runtime_logs"


def _resolved_level_name() -> str:
    raw_level = os.environ.get("TCG_AI_LOG_LEVEL", "INFO").strip().upper()
    return raw_level or "INFO"


def default_log_path(filename: str) -> Path:
    return _DEFAULT_LOG_DIR / filename


def _resolved_log_path(log_file: str | os.PathLike[str] | None) -> Path | None:
    env_override = os.environ.get("TCG_AI_LOG_FILE", "").strip()
    if env_override:
        return Path(env_override).expanduser().resolve()
    if log_file is None:
        return None
    return Path(log_file).expanduser().resolve()


def configure_tcg_ai_logging(log_file: str | os.PathLike[str] | None = None) -> logging.Logger:
    logger = logging.getLogger(_LOGGER_NAME)
    level_name = _resolved_level_name()
    level = getattr(logging, level_name, logging.INFO)
    formatter = logging.Formatter(_LOG_FORMAT, _DATE_FORMAT)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False
    resolved_log_path = _resolved_log_path(log_file)
    if resolved_log_path is not None:
        resolved_log_path.parent.mkdir(parents=True, exist_ok=True)
        handler_name = f"{_LOGGER_NAME}.file:{resolved_log_path}"
        if not any(getattr(handler, "name", "") == handler_name for handler in logger.handlers):
            file_handler = logging.FileHandler(resolved_log_path, encoding="utf-8")
            file_handler.set_name(handler_name)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
    logger.setLevel(level)
    for handler in logger.handlers:
        handler.setLevel(level)
    return logger


def get_logger(name: str) -> logging.Logger:
    configure_tcg_ai_logging()
    if name == _LOGGER_NAME or name.startswith(f"{_LOGGER_NAME}."):
        return logging.getLogger(name)
    return logging.getLogger(f"{_LOGGER_NAME}.{name}")
