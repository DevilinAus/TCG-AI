const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const { execFileSync } = require("node:child_process");
const { JSDOM } = require("jsdom");

const ROOT = path.resolve(__dirname, "..");

function loadEnergyRetrievalSnapshot() {
  const pythonScript = `
from __future__ import annotations
import json
from backend.tcg_ai.server import TcgApplication
from backend.tcg_ai.game_modes.standard.engine import apply_action, list_legal_actions, card_definition

app = TcgApplication()
state = app.new_game({
    "game_mode": "standard",
    "human_first": True,
    "human_deck_id": "ampharos-ex-battle-deck",
    "seed": 1,
})
session = app.sessions.get(state["session_id"])
active_action = next(action for action in list_legal_actions(session.state) if action["type"] == "play_basic_to_active")
apply_action(session.state, active_action)
end_setup = next(action for action in list_legal_actions(session.state) if action["type"] == "end_setup")
apply_action(session.state, end_setup)

player = session.state.players[0]

def take(name: str) -> str:
    for zone_name in ("hand", "deck", "discard", "prizes"):
        zone = getattr(player, zone_name)
        for instance_id in list(zone):
            if card_definition(session.state, instance_id).name != name:
                continue
            zone.remove(instance_id)
            return instance_id
    raise AssertionError(f"Missing {name}")

energy_retrieval_id = take("Energy Retrieval")
lightning_a_id = take("Basic Lightning Energy")
lightning_b_id = take("Basic Lightning Energy")
potion_id = take("Potion")

player.hand = [energy_retrieval_id]
player.discard = [lightning_a_id, lightning_b_id, potion_id]

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
    throw new Error("Unexpected fetch call in Energy Retrieval popup test.");
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

function findHandCardElement(window, cardName) {
  return window.document
    .querySelector(`#player-hand img[alt="${cardName}"]`)
    ?.closest(".mini-card");
}

test("Energy Retrieval opens the discard pile picker and only basic Energy cards are selectable", async () => {
  const snapshot = loadEnergyRetrievalSnapshot();
  const dom = setupDom();
  const { window } = dom;
  const app = window.__TCG_APP_TEST_API__;

  app.setCurrentState(snapshot);
  app.render(snapshot);

  const energyRetrievalElement = findHandCardElement(window, "Energy Retrieval");
  assert.ok(energyRetrievalElement, "Energy Retrieval should render in hand.");

  energyRetrievalElement.dispatchEvent(new window.MouseEvent("click", { bubbles: true }));

  const boardPanel = window.document.getElementById("player-board-panel");
  assert.ok(boardPanel, "Player board panel should render.");
  boardPanel.dispatchEvent(new window.MouseEvent("click", { bubbles: true }));

  const overlay = window.document.getElementById("deck-browser-overlay");
  assert.ok(overlay, "Deck browser overlay should exist.");
  assert.equal(overlay.hidden, false, "The discard pile picker should open after clicking the board background.");
  assert.equal(
    window.document.querySelector(".deck-browser__eyebrow")?.textContent,
    "Discard Pile",
    "Energy Retrieval should browse the discard pile.",
  );
  assert.equal(
    window.document.querySelector(".deck-browser h3")?.textContent,
    "Energy Retrieval",
    "The popup title should use the item name.",
  );

  const confirmButton = window.document.querySelector(".deck-browser__confirm");
  assert.ok(confirmButton, "Confirm button should render.");
  assert.equal(
    confirmButton.disabled,
    false,
    "Energy Retrieval should allow confirming with an empty recovery selection.",
  );

  const energyButtons = [...window.document.querySelectorAll('.deck-browser-card[title="Basic Lightning Energy"]')];
  const potionButton = window.document.querySelector('.deck-browser-card[title="Potion"]');
  assert.equal(energyButtons.length, 2, "Both discarded Basic Energy cards should render.");
  assert.ok(potionButton, "Non-Energy discard cards should still render in the popup.");

  for (const energyButton of energyButtons) {
    assert.ok(energyButton.classList.contains("is-selectable"), "Basic Energy should be selectable.");
    assert.ok(energyButton.classList.contains("is-match"), "Basic Energy should match the recovery filter.");
  }
  assert.ok(
    !potionButton.classList.contains("is-selectable"),
    "Non-Energy cards should not be selectable for Energy Retrieval.",
  );
  assert.ok(
    !potionButton.classList.contains("is-match"),
    "Non-Energy cards should not be highlighted as filter matches.",
  );
});

test("Energy Retrieval popup submits the selected discarded Energy cards", async () => {
  const snapshot = loadEnergyRetrievalSnapshot();
  const dom = setupDom();
  const { window } = dom;
  const app = window.__TCG_APP_TEST_API__;

  const recoveryAction = snapshot.legal_actions.find(
    (action) =>
      action.type === "play_item" &&
      action.source?.card_id === "sv1-171" &&
      Array.isArray(action.action?.recover_from_discard_ids) &&
      action.action.recover_from_discard_ids.length === 2,
  );
  assert.ok(recoveryAction, "A two-card Energy Retrieval action should be available.");

  app.setCurrentState(snapshot);
  app.render(snapshot);

  const energyRetrievalElement = findHandCardElement(window, "Energy Retrieval");
  assert.ok(energyRetrievalElement, "Energy Retrieval should render in hand.");
  energyRetrievalElement.dispatchEvent(new window.MouseEvent("click", { bubbles: true }));

  const boardPanel = window.document.getElementById("player-board-panel");
  assert.ok(boardPanel, "Player board panel should render.");
  boardPanel.dispatchEvent(new window.MouseEvent("click", { bubbles: true }));

  let submittedAction = null;
  app.setSubmitActionOverride(async (actionView) => {
    submittedAction = actionView;
  });

  for (const recoveredId of recoveryAction.action.recover_from_discard_ids) {
    const cardButton = window.document.querySelector(
      `.deck-browser-card[data-instance-id="${recoveredId}"]`,
    );
    assert.ok(cardButton, "Each recoverable Energy card should render in the popup.");
    cardButton.dispatchEvent(new window.MouseEvent("click", { bubbles: true }));
    assert.ok(cardButton.classList.contains("is-selected"), "Chosen Energy cards should highlight.");
  }

  const confirmButton = window.document.querySelector(".deck-browser__confirm");
  assert.ok(confirmButton, "Confirm button should render.");
  confirmButton.dispatchEvent(new window.MouseEvent("click", { bubbles: true }));

  assert.ok(submittedAction, "Confirm should submit an Energy Retrieval action.");
  assert.deepEqual(
    [...submittedAction.action.recover_from_discard_ids].sort(),
    [...recoveryAction.action.recover_from_discard_ids].sort(),
    "Confirm should submit the chosen discarded Energy ids.",
  );
});
