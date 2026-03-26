const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const { execFileSync } = require("node:child_process");
const { JSDOM } = require("jsdom");

const ROOT = path.resolve(__dirname, "..");

function loadAcuPunchTureSnapshot() {
  const pythonScript = `
from __future__ import annotations
import json
from backend.tcg_ai.server import TcgApplication
from backend.tcg_ai.game_modes.standard.engine import card_definition
from backend.tcg_ai.game_modes.standard.models import PokemonInPlay

app = TcgApplication()
state = app.new_game({
    "game_mode": "standard",
    "human_first": True,
    "human_deck_id": "lucario-ex-battle-deck",
    "seed": 1,
})
session = app.sessions.get(state["session_id"])
session.state.setup_phase = None
session.state.current_player = 0
session.state.turn_number = 2
session.state.players[0].turns_taken = 2
session.state.players[1].turns_taken = 2

human = session.state.players[0]
opponent = session.state.players[1]

def take(player, name: str) -> str:
    for zone_name in ("hand", "deck", "discard", "prizes"):
        zone = getattr(player, zone_name)
        for instance_id in list(zone):
            if card_definition(session.state, instance_id).name != name:
                continue
            zone.remove(instance_id)
            return instance_id
    raise AssertionError(f"Missing {name}")

human.active = PokemonInPlay(stack=[take(human, "Medicham")])
human.active.attached_energy = [take(human, "Basic Fighting Energy")]
human.bench = []
opponent.active = PokemonInPlay(stack=[take(opponent, "Ampharos ex")])
opponent.bench = []

snapshot = app.get_game(state["session_id"])
print(json.dumps(snapshot))
`;

  return JSON.parse(
    execFileSync("python3", ["-c", pythonScript], {
      cwd: ROOT,
      encoding: "utf8",
    }),
  );
}

function setupDom() {
  const html = fs.readFileSync(path.join(ROOT, "frontend/index.html"), "utf8");
  const dom = new JSDOM(html, {
    runScripts: "outside-only",
    url: "http://localhost/",
  });
  const { window } = dom;
  window.fetch = async () => {
    throw new Error("Unexpected fetch call in Acu-Punch-Ture popup test.");
  };
  window.requestAnimationFrame = (callback) => {
    callback();
    return 1;
  };
  window.cancelAnimationFrame = () => {};
  window.eval(fs.readFileSync(path.join(ROOT, "frontend/selection-state.js"), "utf8"));
  window.eval(fs.readFileSync(path.join(ROOT, "frontend/app.js"), "utf8"));
  return dom;
}

function findAcuPunchTureButton(window) {
  return window.document.querySelector('#player-active .attack-chip-button[data-attack-index="0"]');
}

test("Acu-Punch-Ture opens an attack-effect chooser with one option per opposing attack", async () => {
  const snapshot = loadAcuPunchTureSnapshot();
  const dom = setupDom();
  const { window } = dom;
  const app = window.__TCG_APP_TEST_API__;

  app.setCurrentState(snapshot);
  app.render(snapshot);

  const attackButton = findAcuPunchTureButton(window);
  assert.ok(attackButton, "Acu-Punch-Ture attack button should render.");
  attackButton.dispatchEvent(new window.MouseEvent("click", { bubbles: true }));

  const overlay = window.document.getElementById("deck-browser-overlay");
  assert.ok(overlay, "Deck browser overlay should exist.");
  assert.equal(overlay.hidden, false, "Clicking Acu-Punch-Ture should open the attack-effect chooser.");
  assert.equal(
    window.document.querySelector(".deck-browser__eyebrow")?.textContent,
    "Choose Attack Effect",
    "Acu-Punch-Ture should open the lightweight attack chooser.",
  );
  assert.equal(
    window.document.querySelector(".deck-browser h3")?.textContent,
    "Acu-Punch-Ture",
    "The popup title should use the attack name.",
  );

  const optionButtons = [...window.document.querySelectorAll("#deck-browser-overlay .action-button")];
  assert.equal(optionButtons.length, 2, "One button should render for each opposing Ampharos ex attack.");
  assert.deepEqual(
    optionButtons.map((button) => button.textContent).sort(),
    [
      "Use Acu-Punch-Ture and block Electro Ball",
      "Use Acu-Punch-Ture and block Thunderstrike Tail",
    ],
  );
});

test("Acu-Punch-Ture chooser submits the selected blocked attack option", async () => {
  const snapshot = loadAcuPunchTureSnapshot();
  const dom = setupDom();
  const { window } = dom;
  const app = window.__TCG_APP_TEST_API__;

  app.setCurrentState(snapshot);
  app.render(snapshot);

  let submittedAction = null;
  app.setSubmitActionOverride(async (actionView) => {
    submittedAction = actionView;
  });

  const attackButton = findAcuPunchTureButton(window);
  assert.ok(attackButton, "Acu-Punch-Ture attack button should render.");
  attackButton.dispatchEvent(new window.MouseEvent("click", { bubbles: true }));

  const thunderstrikeButton = [...window.document.querySelectorAll("#deck-browser-overlay .action-button")].find(
    (button) => button.textContent === "Use Acu-Punch-Ture and block Thunderstrike Tail",
  );
  assert.ok(thunderstrikeButton, "The Thunderstrike Tail option should render.");

  thunderstrikeButton.dispatchEvent(new window.MouseEvent("click", { bubbles: true }));

  assert.ok(submittedAction, "Choosing an option should submit an attack action.");
  assert.equal(
    submittedAction.action.blocked_attack_index,
    1,
    "The submitted action should carry the chosen blocked attack index.",
  );
});
