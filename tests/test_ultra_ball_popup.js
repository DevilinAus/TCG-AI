const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const { execFileSync } = require("node:child_process");
const { JSDOM } = require("jsdom");

const ROOT = path.resolve(__dirname, "..");

function loadUltraBallSnapshot() {
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

def move(name: str) -> str:
    for zone_name in ("hand", "deck", "discard"):
        zone = getattr(player, zone_name)
        for instance_id in list(zone):
            if card_definition(session.state, instance_id).name != name:
                continue
            if zone_name != "hand":
                zone.remove(instance_id)
                player.hand.append(instance_id)
            return instance_id
    raise AssertionError(f"Missing {name}")

ultra_ball_id = move("Ultra Ball")
discard_a_id = move("Potion")
discard_b_id = move("Switch")
player.hand = [ultra_ball_id, discard_a_id, discard_b_id]
snapshot = app.get_game(state["session_id"])
print(json.dumps(snapshot))
`

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
    throw new Error("Unexpected fetch call in Ultra Ball popup test.");
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

test("Ultra Ball opens a discard picker popup when selected and the board background is clicked", async () => {
  const snapshot = loadUltraBallSnapshot();
  const dom = setupDom();
  const { window } = dom;
  const app = window.__TCG_APP_TEST_API__;

  app.setCurrentState(snapshot);
  app.render(snapshot);

  const ultraBallElement = findHandCardElement(window, "Ultra Ball");
  assert.ok(ultraBallElement, "Ultra Ball should render in hand.");

  ultraBallElement.dispatchEvent(new window.MouseEvent("click", { bubbles: true }));

  const boardPanel = window.document.getElementById("player-board-panel");
  assert.ok(boardPanel, "Player board panel should render.");
  boardPanel.dispatchEvent(new window.MouseEvent("click", { bubbles: true }));

  const overlay = window.document.getElementById("deck-browser-overlay");
  assert.ok(overlay, "Deck browser overlay should exist.");
  assert.equal(overlay.hidden, false, "The discard popup should open after clicking the board background.");
  assert.equal(
    window.document.querySelector(".deck-browser__eyebrow")?.textContent,
    "Discard Cards",
    "The first popup stage should be the discard picker.",
  );
  assert.ok(
    window.document.querySelector('.deck-browser-card[title="Potion"]'),
    "Discard candidates from hand should render in the popup.",
  );
});

test("Ultra Ball discard popup lets the player choose two discards and then opens full deck search", async () => {
  const snapshot = loadUltraBallSnapshot();
  const dom = setupDom();
  const { window } = dom;
  const app = window.__TCG_APP_TEST_API__;

  app.setCurrentState(snapshot);
  app.render(snapshot);

  let ultraBallElement = findHandCardElement(window, "Ultra Ball");
  let potionElement = findHandCardElement(window, "Potion");
  let switchElement = findHandCardElement(window, "Switch");
  assert.ok(ultraBallElement, "Ultra Ball should render in hand.");
  assert.ok(potionElement, "Potion should render in hand.");
  assert.ok(switchElement, "Switch should render in hand.");

  ultraBallElement.dispatchEvent(new window.MouseEvent("click", { bubbles: true }));

  ultraBallElement = findHandCardElement(window, "Ultra Ball");
  potionElement = findHandCardElement(window, "Potion");
  switchElement = findHandCardElement(window, "Switch");

  assert.ok(ultraBallElement.classList.contains("is-selected"), "Ultra Ball should be selected after clicking it.");
  assert.ok(
    !potionElement.classList.contains("is-targetable"),
    "Discardable hand cards should not glow like hand-targeted actions.",
  );
  assert.ok(
    !switchElement.classList.contains("is-targetable"),
    "Discardable hand cards should stay neutral until they are chosen.",
  );

  const boardPanel = window.document.getElementById("player-board-panel");
  assert.ok(boardPanel, "Player board panel should render.");
  boardPanel.dispatchEvent(new window.MouseEvent("click", { bubbles: true }));

  const discardPotionButton = window.document.querySelector('.deck-browser-card[title="Potion"]');
  const discardSwitchButton = window.document.querySelector('.deck-browser-card[title="Switch"]');
  const discardConfirmButton = window.document.querySelector(".deck-browser__confirm");
  assert.ok(discardPotionButton, "Potion should be selectable in the discard popup.");
  assert.ok(discardSwitchButton, "Switch should be selectable in the discard popup.");
  assert.ok(discardConfirmButton, "Discard confirm button should render.");
  assert.equal(discardConfirmButton.disabled, true, "Discard confirm should start disabled.");

  discardPotionButton.dispatchEvent(new window.MouseEvent("click", { bubbles: true }));
  assert.ok(discardPotionButton.classList.contains("is-selected"), "The first discard choice should highlight.");
  assert.equal(discardConfirmButton.disabled, true, "Discard confirm should stay disabled after one choice.");

  discardSwitchButton.dispatchEvent(new window.MouseEvent("click", { bubbles: true }));
  assert.ok(discardSwitchButton.classList.contains("is-selected"), "The second discard choice should highlight.");
  assert.equal(discardConfirmButton.disabled, false, "Discard confirm should enable after two choices.");

  discardConfirmButton.dispatchEvent(new window.MouseEvent("click", { bubbles: true }));

  assert.deepEqual(
    [...app.uiState.selectedDiscardIds].sort(),
    ["Potion", "Switch"]
      .map((name) => snapshot.players[0].hand.find((card) => card.name === name)?.instance_id)
      .sort(),
    "Confirming the discard popup should store the chosen discard ids.",
  );
  assert.equal(
    window.document.querySelector(".deck-browser__eyebrow")?.textContent,
    "Full Deck Search",
    "After confirming discards, the full deck search popup should open.",
  );
  assert.ok(
    window.document.querySelector(".deck-browser-card.is-selectable"),
    "Once the popup opens, selectable deck cards should be visible.",
  );
});

test("Ultra Ball popup selects a Pokemon card, highlights it, and confirms it into the hand UI", async () => {
  const snapshot = loadUltraBallSnapshot();
  const dom = setupDom();
  const { window } = dom;
  const app = window.__TCG_APP_TEST_API__;

  const ultraBallCard = snapshot.players[0].hand.find((card) => card.name === "Ultra Ball");
  assert.ok(ultraBallCard, "Ultra Ball should be in the hand snapshot.");

  const ultraBallAction = snapshot.legal_actions.find(
    (action) =>
      action.type === "play_item" &&
      action.source?.instance_id === ultraBallCard.instance_id &&
      Array.isArray(action.action?.search_deck_ids) &&
      action.action.search_deck_ids.length === 1,
  );
  assert.ok(ultraBallAction, "Ultra Ball search action should be available.");

  const selectedDeckCardId = ultraBallAction.action.search_deck_ids[0];
  const selectedDeckCard = snapshot.players[0].deck_cards.find((card) => card.instance_id === selectedDeckCardId);
  assert.ok(selectedDeckCard, "Selected deck card should be present in the deck snapshot.");

  app.setCurrentState(snapshot);
  app.uiState.selectedCardId = ultraBallCard.instance_id;
  app.uiState.selectedDiscardIds = [...ultraBallAction.action.discard_from_hand_ids];
  app.render(snapshot);
  assert.equal(app.openDeckBrowserForSelection(snapshot), true);

  const cardButton = window.document.querySelector(
    `.deck-browser-card[data-instance-id="${selectedDeckCardId}"]`,
  );
  assert.ok(cardButton, "Selectable deck card button should render in the popup.");
  assert.equal(cardButton.disabled, false);

  cardButton.dispatchEvent(new window.MouseEvent("click", { bubbles: true }));

  assert.ok(
    cardButton.classList.contains("is-selected"),
    "Clicking a valid Pokemon should add the selected highlight class.",
  );

  const confirmButton = window.document.querySelector(".deck-browser__confirm");
  assert.ok(confirmButton, "Confirm button should render.");
  assert.equal(confirmButton.disabled, false, "Confirm should enable once a card is selected.");

  let submittedAction = null;
  app.setSubmitActionOverride(async (actionView) => {
    submittedAction = actionView;
    const updated = structuredClone(snapshot);
    const movedCard = updated.players[0].deck_cards.find((card) => card.instance_id === selectedDeckCardId);
    updated.players[0].deck_cards = updated.players[0].deck_cards.filter(
      (card) => card.instance_id !== selectedDeckCardId,
    );
    updated.players[0].deck_count = updated.players[0].deck_cards.length;
    updated.players[0].hand = [...updated.players[0].hand, movedCard];
    updated.players[0].hand_count = updated.players[0].hand.length;
    updated.legal_actions = [];
    app.setCurrentState(updated);
    app.render(updated);
  });

  confirmButton.dispatchEvent(new window.MouseEvent("click", { bubbles: true }));

  assert.ok(submittedAction, "Confirm should submit the selected Ultra Ball search action.");
  assert.deepEqual(
    submittedAction.action.search_deck_ids,
    [selectedDeckCardId],
    "Confirm should submit the chosen Pokemon search id.",
  );
  assert.ok(
    window.document.querySelector(`#player-hand img[alt="${selectedDeckCard.name}"]`),
    "After confirm, the selected Pokemon should appear in the hand UI.",
  );
});
