const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const { execFileSync } = require("node:child_process");
const { JSDOM } = require("jsdom");

const ROOT = path.resolve(__dirname, "..");

function loadStandardSnapshot() {
  const pythonScript = `
from __future__ import annotations
import json
from backend.tcg_ai.server import TcgApplication

app = TcgApplication()
state = app.new_game({
    "game_mode": "standard",
    "human_first": True,
    "human_deck_id": "ampharos-ex-battle-deck",
    "seed": 1,
    "standard_ai_mode": "local",
})
print(json.dumps(state))
`

  return JSON.parse(
    execFileSync("python3", ["-c", pythonScript], {
      cwd: ROOT,
      encoding: "utf8",
    }),
  );
}

function setupDom(fetchImpl) {
  const html = fs.readFileSync(path.join(ROOT, "frontend/index.html"), "utf8");
  const dom = new JSDOM(html, {
    runScripts: "outside-only",
    url: "http://localhost/",
  });
  const { window } = dom;
  window.fetch = fetchImpl;
  window.confirm = () => true;
  window.requestAnimationFrame = (callback) => {
    callback();
    return 1;
  };
  window.cancelAnimationFrame = () => {};
  window.eval(fs.readFileSync(path.join(ROOT, "frontend/selection-state.js"), "utf8"));
  window.eval(fs.readFileSync(path.join(ROOT, "frontend/app.js"), "utf8"));
  return dom;
}

function jsonResponse(payload, { ok = true, status = 200 } = {}) {
  return {
    ok,
    status,
    async json() {
      return payload;
    },
  };
}

test("NN toggle checks remote readiness and starts a fresh Standard game in remote mode", async () => {
  const snapshot = loadStandardSnapshot();
  const fetchCalls = [];
  const confirmMessages = [];
  const remoteSnapshot = {
    ...snapshot,
    session_id: "remote-session-123",
    standard_ai_mode: "remote",
  };

  const dom = setupDom(async (url, options = {}) => {
    fetchCalls.push({
      url,
      method: options.method || "GET",
      body: options.body || null,
    });
    if (url === "/api/standard-ml-status") {
      return jsonResponse({
        configured: true,
        ready: true,
        model_loaded: true,
        backend: "torch",
        checkpoint_path: "/models/champion.pt",
        ready_url: "http://127.0.0.1:8100/readyz",
        error: null,
      });
    }
    if (url === "/api/new-game") {
      return jsonResponse(remoteSnapshot);
    }
    throw new Error(`Unexpected fetch call: ${url}`);
  });
  const { window } = dom;
  window.confirm = (message) => {
    confirmMessages.push(message);
    return true;
  };
  const app = window.__TCG_APP_TEST_API__;

  app.setCurrentState(snapshot);
  app.uiState.selectedStandardAiMode = "local";
  app.render(snapshot);

  const toggle = window.document.getElementById("standard-ai-mode-toggle");
  assert.ok(toggle, "The NN mode toggle should render.");
  toggle.checked = true;

  await app.handleStandardAiModeToggle({ target: toggle });

  assert.equal(confirmMessages.length, 1, "Switching modes mid-game should ask for confirmation.");
  assert.match(confirmMessages[0], /reset the current game/i);
  assert.equal(fetchCalls.length, 2, "The UI should check readiness before starting a new game.");
  assert.equal(fetchCalls[0].url, "/api/standard-ml-status");
  assert.equal(fetchCalls[0].method, "GET");
  assert.equal(fetchCalls[1].url, "/api/new-game");
  const newGamePayload = JSON.parse(fetchCalls[1].body);
  assert.equal(newGamePayload.standard_ai_mode, "remote");
  assert.equal(newGamePayload.game_mode, "standard");
  assert.equal(app.uiState.selectedStandardAiMode, "remote");
  assert.equal(toggle.checked, true, "The toggle should stay enabled after the remote game starts.");
});

test("NN toggle stays local when the remote worker is unavailable", async () => {
  const snapshot = loadStandardSnapshot();
  const fetchCalls = [];

  const dom = setupDom(async (url, options = {}) => {
    fetchCalls.push({
      url,
      method: options.method || "GET",
    });
    if (url === "/api/standard-ml-status") {
      return jsonResponse({
        configured: true,
        ready: false,
        model_loaded: false,
        backend: null,
        checkpoint_path: null,
        ready_url: "http://127.0.0.1:8100/readyz",
        error: "Remote Standard NN worker is unavailable.",
      });
    }
    throw new Error(`Unexpected fetch call: ${url}`);
  });
  const { window } = dom;
  const app = window.__TCG_APP_TEST_API__;

  app.setCurrentState(snapshot);
  app.uiState.selectedStandardAiMode = "local";
  app.render(snapshot);

  const toggle = window.document.getElementById("standard-ai-mode-toggle");
  assert.ok(toggle, "The NN mode toggle should render.");
  toggle.checked = true;

  await app.handleStandardAiModeToggle({ target: toggle });

  assert.equal(fetchCalls.length, 1, "A failed readiness check should stop before creating a new game.");
  assert.equal(fetchCalls[0].url, "/api/standard-ml-status");
  assert.equal(app.uiState.selectedStandardAiMode, "local");
  assert.equal(toggle.checked, false, "The toggle should revert when NN mode is unavailable.");
});
