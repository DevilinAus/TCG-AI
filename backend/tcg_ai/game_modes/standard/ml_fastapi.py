from __future__ import annotations

import os
from typing import Any

try:
    from fastapi import FastAPI, Header, HTTPException
    import uvicorn
except Exception as exc:  # pragma: no cover - optional dependency path
    FastAPI = None
    Header = None
    HTTPException = RuntimeError
    uvicorn = None
    _IMPORT_ERROR = exc
else:  # pragma: no cover - exercised when optional dependencies are installed
    _IMPORT_ERROR = None

from .ml.service import StandardMlService


def create_app() -> Any:
    if FastAPI is None:
        raise RuntimeError(
            "FastAPI dependencies are not installed. Install the 'standard-ml' extras to use this server."
        ) from _IMPORT_ERROR

    app = FastAPI(title="TCG AI Standard ML Worker", version="0.1.0")
    service = StandardMlService()
    expected_token = os.environ.get("TCG_AI_STANDARD_REMOTE_API_TOKEN")

    def _require_token(provided_token: str | None) -> None:
        if not expected_token:
            return
        if provided_token != expected_token:
            raise HTTPException(status_code=401, detail="Invalid Standard ML API token.")

    @app.get("/healthz")
    def healthz() -> dict[str, Any]:
        return service.health()

    @app.get("/readyz")
    def readyz() -> dict[str, Any]:
        return service.ready()

    @app.post("/api/standard-ml/decision")
    def decision(
        payload: dict[str, Any],
        x_standard_ml_token: str | None = Header(default=None),
    ) -> dict[str, Any]:
        _require_token(x_standard_ml_token)
        return service.choose_action(payload)

    @app.post("/api/standard-ml/batch-eval")
    def batch_eval(
        payload: dict[str, Any],
        x_standard_ml_token: str | None = Header(default=None),
    ) -> dict[str, Any]:
        _require_token(x_standard_ml_token)
        return service.evaluate_batch(payload)

    @app.post("/api/standard-ml/outcome")
    def outcome(
        payload: dict[str, Any],
        x_standard_ml_token: str | None = Header(default=None),
    ) -> dict[str, Any]:
        _require_token(x_standard_ml_token)
        return service.record_outcome(payload)

    return app


def run(host: str = "0.0.0.0", port: int = 8100) -> None:  # pragma: no cover - runtime helper
    if uvicorn is None:
        raise RuntimeError(
            "uvicorn is not installed. Install the 'standard-ml' extras to use this server."
        ) from _IMPORT_ERROR
    uvicorn.run(create_app(), host=host, port=port)


if __name__ == "__main__":  # pragma: no cover - runtime helper
    run()
