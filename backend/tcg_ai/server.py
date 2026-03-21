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

from .game_modes import (
    DEFAULT_GAME_MODE,
    GameModeDefinition,
    available_game_mode_snapshots,
    get_game_mode,
)
from .learning import EpisodeStep
from .trainers import TrainerProfile, TrainerStore

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FRONTEND_DIR = PROJECT_ROOT / "frontend"
AI_ACTION_DELAY_MIN_MS = 5_000
AI_ACTION_DELAY_MAX_MS = 8_000
OPENING_DIE_SIDES = 6


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
        trainer: TrainerProfile,
        trainer_store: TrainerStore,
        mode: GameModeDefinition,
        human_deck_id: str | None = None,
        seed: int | None = None,
        human_first: bool = True,
    ) -> None:
        self.lock = threading.Lock()
        self.trainer = trainer
        self.trainer_store = trainer_store
        self.mode = mode
        self.game_mode = mode.game_mode
        self.human_deck_id = human_deck_id
        self.ai_deck_id = (
            mode.paired_deck_id_for(human_deck_id)
            if human_deck_id is not None and mode.paired_deck_id_for is not None
            else None
        )
        self.learner = trainer.learner_for(self.game_mode)
        self.state = mode.create_game(
            seed=seed,
            human_first=human_first,
            ai_name=trainer.name,
            human_deck_id=human_deck_id,
        )
        self.ai_episode_steps: list[EpisodeStep] = []
        self.ai_episode_reward = 0.0
        self.ai_episode_finished = False
        self.ai_replay_delay_ms = AI_ACTION_DELAY_MIN_MS
        self.ai_damage_dealt = 0
        self.ai_prizes_taken = 0
        self.ai_progress_awarded = False

    def snapshot(self, session_id: str) -> dict[str, Any]:
        return self._serialize_state(session_id)

    def _serialize_state(self, session_id: str) -> dict[str, Any]:
        return self.mode.serialize_state(
            self.state,
            session_id=session_id,
            viewer=0,
            ai_learning=self.learner.snapshot(),
        )

    def reset(
        self,
        human_first: bool = True,
        seed: int | None = None,
        human_deck_id: str | None = None,
    ) -> None:
        with self.lock:
            if human_deck_id is not None:
                self.human_deck_id = human_deck_id
                self.ai_deck_id = (
                    self.mode.paired_deck_id_for(human_deck_id)
                    if self.mode.paired_deck_id_for is not None
                    else None
                )
            self.state = self.mode.create_game(
                seed=seed,
                human_first=human_first,
                ai_name=self.trainer.name,
                human_deck_id=self.human_deck_id,
            )
            self.ai_episode_steps.clear()
            self.ai_episode_reward = 0.0
            self.ai_episode_finished = False
            self.ai_damage_dealt = 0
            self.ai_prizes_taken = 0
            self.ai_progress_awarded = False

    def human_action(self, action: dict[str, Any]) -> None:
        with self.lock:
            if self.state.current_player != 0:
                raise ApiError(
                    "It is not the human player's turn.",
                    "wrong_turn",
                    HTTPStatus.BAD_REQUEST,
                )
            self.mode.apply_action(self.state, action)
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

        action = self.mode.choose_action(self.state, 1, learner=self.learner)
        if action is None:
            return None

        before = self.mode.summarize_state(self.state, 1)
        features = self.mode.extract_action_features(self.state, 1, action)
        self.mode.apply_action(self.state, action)
        after = self.mode.summarize_state(self.state, 1)
        reward = self.mode.calculate_reward(before, after, action)
        self.learner.record_step_reward(features, action["type"], reward)
        self.ai_episode_steps.append(
            EpisodeStep(features=features, action_type=action["type"], reward=reward)
        )
        self.ai_episode_reward += reward
        self.ai_damage_dealt += max(0, before.opponent.total_remaining_hp - after.opponent.total_remaining_hp)
        self.ai_prizes_taken += max(0, after.player.prizes_taken - before.player.prizes_taken)
        self._finalize_ai_episode_if_finished(completed_by_ai_action=True)
        return {
            "action": {
                "type": action["type"],
                "label": action["label"],
            },
            "delay_ms": _replay_delay_for_action(self.state),
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
        if not self.ai_progress_awarded:
            self.trainer.gain_experience(
                damage_dealt=self.ai_damage_dealt,
                prizes_taken=self.ai_prizes_taken,
                game_mode=self.game_mode,
            )
            self.ai_progress_awarded = True
        self.trainer_store.save_to_disk()
        self.ai_episode_finished = True


class SessionStore:
    def __init__(self, trainers: TrainerStore) -> None:
        self._lock = threading.Lock()
        self._trainers = trainers
        self._sessions: dict[str, GameSession] = {}

    def create(
        self,
        trainer: TrainerProfile,
        mode: GameModeDefinition,
        human_deck_id: str | None = None,
        human_first: bool = True,
        seed: int | None = None,
    ) -> tuple[str, GameSession]:
        session_id = secrets.token_urlsafe(9)
        session = GameSession(
            trainer,
            self._trainers,
            mode=mode,
            human_deck_id=human_deck_id,
            seed=seed,
            human_first=human_first,
        )
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
    def __init__(self, trainer_state_path: Path | None = None) -> None:
        self.trainers = TrainerStore(state_path=trainer_state_path)
        default_trainer = self.trainers.get(self.trainers.default_trainer_id)
        if default_trainer is None:
            raise ValueError("Default trainer not found.")
        self.learner = default_trainer.learner_for(DEFAULT_GAME_MODE)
        self.sessions = SessionStore(self.trainers)

    def lobby(self) -> dict[str, Any]:
        trainer = self.trainers.get(self.trainers.default_trainer_id)
        if trainer is None:
            raise ApiError("Default trainer not found.", "trainer_not_found", HTTPStatus.INTERNAL_SERVER_ERROR)

        mode = self._default_mode()
        snapshot = {
            "game_mode": mode.game_mode,
            "available_game_modes": available_game_mode_snapshots(selected_id=mode.game_mode),
            "ai_trainer": trainer.snapshot(selected=True, game_mode=mode.game_mode),
            "available_trainers": self.trainers.snapshots(
                selected_id=trainer.trainer_id,
                game_mode=mode.game_mode,
            ),
        }
        return self._attach_deck_payload(
            snapshot,
            mode=mode,
            human_deck_id=mode.default_human_deck_id,
            ai_deck_id=mode.paired_deck_id_for(mode.default_human_deck_id)
            if mode.default_human_deck_id is not None and mode.paired_deck_id_for is not None
            else None,
        )

    def get_game(self, session_id: str) -> dict[str, Any]:
        session = self.sessions.get(session_id)
        return self._build_snapshot(session_id, session)

    def new_game(self, payload: dict[str, Any]) -> dict[str, Any]:
        mode = self._resolve_game_mode(payload.get("game_mode"), require_available=True)
        trainer = self._resolve_trainer(payload.get("trainer_id"))
        human_deck_id = self._resolve_human_deck_id(mode, payload.get("human_deck_id"))
        human_first, opening_roll = self._resolve_human_first(payload)
        session_id, session = self.sessions.create(
            trainer=trainer,
            mode=mode,
            human_deck_id=human_deck_id,
            human_first=human_first,
            seed=payload.get("seed"),
        )
        if opening_roll is not None:
            starter_name = session.state.players[session.state.current_player].name
            session.state.log.insert(1, f"Opening die roll: {opening_roll}. {starter_name} goes first.")
        return self._build_snapshot(session_id, session)

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
        return self._build_snapshot(session_id, session)

    def ai_turn(self, payload: dict[str, Any]) -> dict[str, Any]:
        session_id = self._require_string(payload, "session_id", "missing_session_id")
        session = self.sessions.get(session_id)
        replay_steps = session.ai_turn(session_id)
        snapshot = self._build_snapshot(session_id, session)
        snapshot["ai_turn_replay"] = {
            "step_delay_ms": session.ai_replay_delay_ms,
            "steps": [
                {
                    **step,
                    "state": self._attach_session_payload(step["state"], session),
                }
                for step in replay_steps
            ],
        }
        return snapshot

    def ai_step(self, payload: dict[str, Any]) -> dict[str, Any]:
        session_id = self._require_string(payload, "session_id", "missing_session_id")
        session = self.sessions.get(session_id)
        step = session.ai_step(session_id)
        snapshot = self._build_snapshot(session_id, session)
        snapshot["ai_step"] = None
        if step is not None:
            snapshot["ai_step"] = {
                "action": step["action"],
                "delay_ms": step["delay_ms"],
            }
        return snapshot

    def _build_snapshot(self, session_id: str, session: GameSession) -> dict[str, Any]:
        return self._attach_session_payload(session.snapshot(session_id), session)

    def _attach_session_payload(self, snapshot: dict[str, Any], session: GameSession) -> dict[str, Any]:
        snapshot["game_mode"] = session.game_mode
        snapshot["available_game_modes"] = available_game_mode_snapshots(selected_id=session.game_mode)
        snapshot["ai_trainer"] = session.trainer.snapshot(selected=True, game_mode=session.game_mode)
        snapshot["available_trainers"] = self.trainers.snapshots(
            selected_id=session.trainer.trainer_id,
            game_mode=session.game_mode,
        )
        return self._attach_deck_payload(
            snapshot,
            mode=session.mode,
            human_deck_id=session.human_deck_id,
            ai_deck_id=session.ai_deck_id,
        )

    @staticmethod
    def _attach_deck_payload(
        snapshot: dict[str, Any],
        mode: GameModeDefinition,
        human_deck_id: str | None,
        ai_deck_id: str | None,
    ) -> dict[str, Any]:
        if not mode.supports_decks:
            return snapshot

        snapshot["human_deck_id"] = human_deck_id
        snapshot["ai_deck_id"] = ai_deck_id
        snapshot["available_decks"] = mode.available_deck_snapshots(selected_id=human_deck_id)
        return snapshot

    def _resolve_trainer(self, trainer_id: Any) -> TrainerProfile:
        if trainer_id is None:
            trainer_id = self.trainers.default_trainer_id
        if not isinstance(trainer_id, str) or not trainer_id:
            raise ApiError("Missing field: trainer_id", "missing_trainer_id", HTTPStatus.BAD_REQUEST)
        trainer = self.trainers.get(trainer_id)
        if trainer is None:
            raise ApiError("Unknown trainer ID.", "trainer_not_found", HTTPStatus.BAD_REQUEST)
        return trainer

    @staticmethod
    def _resolve_game_mode(
        game_mode: Any,
        require_available: bool = False,
    ) -> GameModeDefinition:
        if game_mode is None:
            game_mode = DEFAULT_GAME_MODE
        if not isinstance(game_mode, str) or not game_mode:
            raise ApiError("Missing field: game_mode", "missing_game_mode", HTTPStatus.BAD_REQUEST)

        resolved_mode = get_game_mode(game_mode)
        if resolved_mode is None:
            raise ApiError("Unknown game mode.", "game_mode_not_found", HTTPStatus.BAD_REQUEST)
        if require_available and not resolved_mode.available:
            raise ApiError("Game mode is not available yet.", "game_mode_unavailable", HTTPStatus.BAD_REQUEST)
        return resolved_mode

    @staticmethod
    def _resolve_human_deck_id(mode: GameModeDefinition, human_deck_id: Any) -> str | None:
        if not mode.supports_decks:
            return None
        if human_deck_id is None:
            return mode.default_human_deck_id
        if not isinstance(human_deck_id, str) or not human_deck_id:
            raise ApiError(
                "Missing field: human_deck_id",
                "missing_human_deck_id",
                HTTPStatus.BAD_REQUEST,
            )
        if mode.deck_definitions is None or human_deck_id not in mode.deck_definitions:
            raise ApiError("Unknown human deck ID.", "human_deck_not_found", HTTPStatus.BAD_REQUEST)
        return human_deck_id

    @staticmethod
    def _resolve_human_first(
        payload: dict[str, Any],
    ) -> tuple[bool, int | None]:
        requested_order = payload.get("human_first")
        if isinstance(requested_order, bool):
            return requested_order, None

        opening_roll = roll_starting_player_die()
        human_first = opening_roll <= OPENING_DIE_SIDES // 2
        return human_first, opening_roll

    @staticmethod
    def _require_string(payload: dict[str, Any], key: str, code: str) -> str:
        value = payload.get(key)
        if not isinstance(value, str) or not value:
            raise ApiError(f"Missing field: {key}", code, HTTPStatus.BAD_REQUEST)
        return value

    @staticmethod
    def _default_mode() -> GameModeDefinition:
        mode = get_game_mode(DEFAULT_GAME_MODE)
        if mode is None:
            raise ApiError("Default game mode not found.", "game_mode_not_found", HTTPStatus.INTERNAL_SERVER_ERROR)
        return mode


DEFAULT_APPLICATION = TcgApplication()


def make_handler(application: TcgApplication) -> type[BaseHTTPRequestHandler]:
    class ConfiguredTcgRequestHandler(TcgRequestHandler):
        app = application

    return ConfiguredTcgRequestHandler


def _replay_delay_for_action(state) -> int:
    return state.rng.randint(AI_ACTION_DELAY_MIN_MS, AI_ACTION_DELAY_MAX_MS)


def roll_starting_player_die() -> int:
    return secrets.randbelow(OPENING_DIE_SIDES) + 1


class TcgRequestHandler(BaseHTTPRequestHandler):
    server_version = "TcgAiHTTP/0.1"
    app = DEFAULT_APPLICATION

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/api/lobby":
                self._send_json(self.app.lobby())
                return

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
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
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
