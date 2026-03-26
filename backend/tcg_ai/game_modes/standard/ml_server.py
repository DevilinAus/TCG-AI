from __future__ import annotations

from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os

from .ml.service import StandardMlService


class StandardMlRequestHandler(BaseHTTPRequestHandler):
    service = StandardMlService()

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/healthz":
            self._write_json(HTTPStatus.OK, self.service.health())
            return
        if self.path == "/readyz":
            self._write_json(HTTPStatus.OK, self.service.ready())
            return
        self._write_json(HTTPStatus.NOT_FOUND, {"error": "Unknown endpoint."})

    def do_POST(self) -> None:  # noqa: N802
        try:
            self._require_token()
            payload = self._read_json()
            if self.path in {"/", "/api/standard-ml/decision"}:
                body = self.service.choose_action(payload)
                self._write_json(HTTPStatus.OK, body)
                return
            if self.path == "/api/standard-ml/batch-eval":
                body = self.service.evaluate_batch(payload)
                self._write_json(HTTPStatus.OK, body)
                return
            if self.path == "/api/standard-ml/outcome":
                body = self.service.record_outcome(payload)
                self._write_json(HTTPStatus.OK, body)
                return
            self._write_json(
                HTTPStatus.NOT_FOUND,
                {"error": "Unknown endpoint."},
            )
        except PermissionError as exc:
            self._write_json(HTTPStatus.UNAUTHORIZED, {"error": str(exc)})
        except (TypeError, ValueError, KeyError, json.JSONDecodeError) as exc:
            self._write_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})

    def log_message(self, format: str, *args: object) -> None:
        del format, args

    def _read_json(self) -> dict[str, object]:
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length)
        payload = json.loads(body.decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Request body must be a JSON object.")
        return payload

    def _write_json(self, status: HTTPStatus, payload: dict[str, object]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _require_token(self) -> None:
        expected_token = os.environ.get("TCG_AI_STANDARD_REMOTE_API_TOKEN")
        if not expected_token:
            return
        provided_token = self.headers.get("X-Standard-ML-Token", "")
        if provided_token != expected_token:
            raise PermissionError("Invalid Standard ML API token.")


def run(host: str = "0.0.0.0", port: int = 8100) -> None:
    server = ThreadingHTTPServer((host, port), StandardMlRequestHandler)
    print(f"Standard ML server listening on http://{host}:{port}")
    server.serve_forever()


if __name__ == "__main__":
    run()
