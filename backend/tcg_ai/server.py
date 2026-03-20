from __future__ import annotations

import mimetypes
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import secrets
import threading
from typing import Any
from urllib.parse import parse_qs, urlparse

from .bot import choose_action
from .engine import apply_action, create_game
from .learning import EpisodeStep, RewardLearner, calculate_reward, extract_action_features, summarize_state
from .presentation import serialize_state

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FRONTEND_DIR = PROJECT_ROOT / "frontend"


class ApiError(Exception):
    def __init__(
        self,
        message: str,
        code: str,
        status: HTTPStatus = HTTPStatus.BAD_REQUEST,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.code = code
        self.status = status


class GameSession:
    def __init__(
        self,
        learner: RewardLearner,
        seed: int | None = None,
        human_first: bool = True,
    ) -> None:
        self.lock = threading.Lock()
        self.learner = learner
        self.state = create_game(seed=seed, human_first=human_first)
        self.ai_episode_steps: list[EpisodeStep] = []
        self.ai_episode_reward = 0.0
        self.ai_episode_finished = False
        self.ai_replay_delay_ms = 900

    def snapshot(self, session_id: str) -> dict[str, Any]:
        return self._serialize_state(session_id)

    def _serialize_state(self, session_id: str) -> dict[str, Any]:
        return serialize_state(
            self.state,
            session_id=session_id,
            viewer=0,
            ai_learning=self.learner.snapshot(),
        )

    def reset(self, human_first: bool = True, seed: int | None = None) -> None:
        with self.lock:
            self.state = create_game(seed=seed, human_first=human_first)
            self.ai_episode_steps.clear()
            self.ai_episode_reward = 0.0
            self.ai_episode_finished = False

    def human_action(self, action: dict[str, Any]) -> None:
        with self.lock:
            if self.state.current_player != 0:
                raise ApiError(
                    "It is not the human player's turn.",
                    "wrong_turn",
                    HTTPStatus.BAD_REQUEST,
                )
            apply_action(self.state, action)
            self._finalize_ai_episode_if_finished(completed_by_ai_action=False)

    def ai_step(self, session_id: str) -> dict[str, Any] | None:
        with self.lock:
            return self._perform_ai_step(session_id)

    def ai_turn(self, session_id: str) -> list[dict[str, Any]]:
        replay_steps: list[dict[str, Any]] = []
        with self.lock:
            steps = 0
            while self.state.winner is None and self.state.current_player == 1:
                step = self._perform_ai_step(session_id)
                if step is None:
                    break
                replay_steps.append(step)
                steps += 1
                if steps >= 20:
                    self.state.log.append("AI turn loop stopped after 20 actions for safety.")
                    break
        return replay_steps

    def _perform_ai_step(self, session_id: str) -> dict[str, Any] | None:
        if self.state.winner is not None or self.state.current_player != 1:
            return None

        action = choose_action(self.state, 1, learner=self.learner)
        if action is None:
            return None

        before = summarize_state(self.state, 1)
        features = extract_action_features(self.state, 1, action)
        apply_action(self.state, action)
        reward = calculate_reward(before, summarize_state(self.state, 1), action)
        self.learner.record_step_reward(features, action["type"], reward)
        self.ai_episode_steps.append(
            EpisodeStep(features=features, action_type=action["type"], reward=reward)
        )
        self.ai_episode_reward += reward
        self._finalize_ai_episode_if_finished(completed_by_ai_action=True)
        return {
            "action": {
                "type": action["type"],
                "label": action["label"],
            },
            "delay_ms": _replay_delay_for_action(action["type"], default_delay_ms=self.ai_replay_delay_ms),
            "state": self._serialize_state(session_id),
        }

    def _finalize_ai_episode_if_finished(self, completed_by_ai_action: bool) -> None:
        if self.ai_episode_finished or self.state.winner is None:
            return

        terminal_reward = 30.0 if self.state.winner == 1 else -30.0
        total_episode_reward = (
            self.ai_episode_reward
            if completed_by_ai_action
            else self.ai_episode_reward + terminal_reward
        )
        self.learner.record_episode_result(
            self.ai_episode_steps,
            terminal_reward=terminal_reward,
            winner=self.state.winner,
            learner_player_index=1,
            total_episode_reward=total_episode_reward,
            skip_last_step=completed_by_ai_action,
        )
        self.ai_episode_finished = True


class SessionStore:
    def __init__(self, learner: RewardLearner) -> None:
        self._lock = threading.Lock()
        self._learner = learner
        self._sessions: dict[str, GameSession] = {}

    def create(self, human_first: bool = True, seed: int | None = None) -> tuple[str, GameSession]:
        session_id = secrets.token_urlsafe(9)
        session = GameSession(self._learner, seed=seed, human_first=human_first)
        with self._lock:
            self._sessions[session_id] = session
        return session_id, session

    def get(self, session_id: str) -> GameSession:
        with self._lock:
            session = self._sessions.get(session_id)
        if session is None:
            raise ApiError("Unknown session ID.", "session_not_found", HTTPStatus.NOT_FOUND)
        return session


class TcgApplication:
    def __init__(self) -> None:
        self.learner = RewardLearner()
        self.sessions = SessionStore(self.learner)

    def get_game(self, session_id: str) -> dict[str, Any]:
        session = self.sessions.get(session_id)
        return session.snapshot(session_id)

    def new_game(self, payload: dict[str, Any]) -> dict[str, Any]:
        session_id, session = self.sessions.create(
            human_first=payload.get("human_first", True),
            seed=payload.get("seed"),
        )
        return session.snapshot(session_id)

    def human_action(self, payload: dict[str, Any]) -> dict[str, Any]:
        session_id = self._require_string(payload, "session_id", "missing_session_id")
        action = payload.get("action")
        if not isinstance(action, dict):
            raise ApiError("Missing action payload.", "missing_action", HTTPStatus.BAD_REQUEST)

        session = self.sessions.get(session_id)
        try:
            session.human_action(action)
        except ValueError as exc:
            raise ApiError(str(exc), "illegal_action", HTTPStatus.BAD_REQUEST) from exc
        return session.snapshot(session_id)

    def ai_turn(self, payload: dict[str, Any]) -> dict[str, Any]:
        session_id = self._require_string(payload, "session_id", "missing_session_id")
        session = self.sessions.get(session_id)
        replay_steps = session.ai_turn(session_id)
        snapshot = session.snapshot(session_id)
        snapshot["ai_turn_replay"] = {
            "step_delay_ms": session.ai_replay_delay_ms,
            "steps": replay_steps,
        }
        return snapshot

    def ai_step(self, payload: dict[str, Any]) -> dict[str, Any]:
        session_id = self._require_string(payload, "session_id", "missing_session_id")
        session = self.sessions.get(session_id)
        step = session.ai_step(session_id)
        snapshot = session.snapshot(session_id)
        snapshot["ai_step"] = None
        if step is not None:
            snapshot["ai_step"] = {
                "action": step["action"],
                "delay_ms": step["delay_ms"],
            }
        return snapshot

    @staticmethod
    def _require_string(payload: dict[str, Any], key: str, code: str) -> str:
        value = payload.get(key)
        if not isinstance(value, str) or not value:
            raise ApiError(f"Missing field: {key}", code, HTTPStatus.BAD_REQUEST)
        return value


DEFAULT_APPLICATION = TcgApplication()


def make_handler(application: TcgApplication) -> type[BaseHTTPRequestHandler]:
    class ConfiguredTcgRequestHandler(TcgRequestHandler):
        app = application

    return ConfiguredTcgRequestHandler


def _replay_delay_for_action(action_type: str, default_delay_ms: int) -> int:
    if action_type == "attack":
        return default_delay_ms + 350
    if action_type in {"promote", "play_switch"}:
        return default_delay_ms + 150
    if action_type == "end_turn":
        return max(450, default_delay_ms - 250)
    return default_delay_ms


class TcgRequestHandler(BaseHTTPRequestHandler):
    server_version = "TcgAiHTTP/0.1"
    app = DEFAULT_APPLICATION

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/api/game":
                session_id = self._query_value(parsed.query, "session_id")
                self._send_json(self.app.get_game(session_id))
                return

            self._serve_static_path(parsed.path)
        except ApiError as exc:
            self._send_json({"error": exc.message, "code": exc.code}, status=exc.status)

    def do_HEAD(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        try:
            self._serve_static_path(parsed.path, include_body=False)
        except ApiError as exc:
            self._send_json({"error": exc.message, "code": exc.code}, status=exc.status)

    def do_POST(self) -> None:  # noqa: N802
        try:
            payload = self._read_json()
            parsed = urlparse(self.path)
            if parsed.path == "/api/new-game":
                self._send_json(self.app.new_game(payload))
                return

            if parsed.path == "/api/action":
                self._send_json(self.app.human_action(payload))
                return

            if parsed.path == "/api/ai-turn":
                self._send_json(self.app.ai_turn(payload))
                return

            if parsed.path == "/api/ai-step":
                self._send_json(self.app.ai_step(payload))
                return

            raise ApiError("Route not found.", "route_not_found", HTTPStatus.NOT_FOUND)
        except ApiError as exc:
            self._send_json({"error": exc.message, "code": exc.code}, status=exc.status)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
        return

    def _serve_static_path(self, request_path: str, include_body: bool = True) -> None:
        relative_path = "index.html" if request_path in {"", "/"} else request_path.lstrip("/")
        resolved = (FRONTEND_DIR / relative_path).resolve()
        frontend_root = FRONTEND_DIR.resolve()
        if frontend_root not in resolved.parents and resolved != frontend_root:
            raise ApiError("Static path not allowed.", "static_path_invalid", HTTPStatus.NOT_FOUND)
        if not resolved.exists() or not resolved.is_file():
            raise ApiError("Static file not found.", "static_not_found", HTTPStatus.NOT_FOUND)

        content = resolved.read_bytes()
        content_type, _ = mimetypes.guess_type(str(resolved))
        header_value = content_type or "application/octet-stream"
        if header_value.startswith("text/") or header_value in {
            "application/javascript",
            "application/json",
            "image/svg+xml",
        }:
            header_value = f"{header_value}; charset=utf-8"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", header_value)
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        if include_body:
            self.wfile.write(content)

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length == 0:
            return {}
        body = self.rfile.read(length)
        try:
            return json.loads(body.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise ApiError("Malformed JSON payload.", "malformed_json", HTTPStatus.BAD_REQUEST) from exc

    def _send_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        encoded = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    @staticmethod
    def _query_value(raw_query: str, key: str) -> str:
        query = parse_qs(raw_query)
        values = query.get(key)
        if not values or not values[0]:
            raise ApiError(f"Missing query parameter: {key}", f"missing_{key}", HTTPStatus.BAD_REQUEST)
        return values[0]


def build_server(
    host: str = "127.0.0.1",
    port: int = 8000,
    application: TcgApplication | None = None,
) -> ThreadingHTTPServer:
    return ThreadingHTTPServer((host, port), make_handler(application or TcgApplication()))


def run_server(host: str = "127.0.0.1", port: int = 8000) -> None:
    server = build_server(host=host, port=port)
    print(f"Serving TCG AI starter at http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...")
    finally:
        server.server_close()


if __name__ == "__main__":
    run_server()
