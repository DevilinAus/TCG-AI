const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const { execFileSync } = require("node:child_process");
const { JSDOM } = require("jsdom");

const ROOT = path.resolve(__dirname, "..");

function loadPokegearSnapshot() {
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

pokegear_id = move("Pok\\u00e9gear 3.0")
player.hand = [pokegear_id]

top_supporter_id = move("Nemona")
outside_supporter_id = move("Youngster")
ordered_top_cards = [
    move("Mareep"),
    move("Wattrel"),
    top_supporter_id,
    move("Rotom"),
    move("Starly"),
    move("Switch"),
    move("Potion"),
]
prioritized_ids = set(ordered_top_cards + [outside_supporter_id])
player.deck = ordered_top_cards + [
    instance_id
    for instance_id in player.deck
    if instance_id not in prioritized_ids
] + [outside_supporter_id]

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
    throw new Error("Unexpected fetch call in Pokegear popup test.");
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

test("Pokegear opens a top-seven search popup and only supporters are selectable", async () => {
  const snapshot = loadPokegearSnapshot();
  const dom = setupDom();
  const { window } = dom;
  const app = window.__TCG_APP_TEST_API__;

  app.setCurrentState(snapshot);
  app.render(snapshot);

  const pokegearElement = findHandCardElement(window, "Pok\u00e9gear 3.0");
  assert.ok(pokegearElement, "Pokegear should render in hand.");

  pokegearElement.dispatchEvent(new window.MouseEvent("click", { bubbles: true }));

  const boardPanel = window.document.getElementById("player-board-panel");
  assert.ok(boardPanel, "Player board panel should render.");
  boardPanel.dispatchEvent(new window.MouseEvent("click", { bubbles: true }));

  const overlay = window.document.getElementById("deck-browser-overlay");
  assert.ok(overlay, "Deck browser overlay should exist.");
  assert.equal(overlay.hidden, false, "The top-seven search popup should open.");
  assert.equal(
    window.document.querySelector(".deck-browser__eyebrow")?.textContent,
    "Top 7 Search",
    "Pokegear should browse only the top seven cards.",
  );

  const visibleCards = [...window.document.querySelectorAll(".deck-browser-card")];
  assert.equal(visibleCards.length, 7, "Only the top seven cards should render.");
  const confirmButton = window.document.querySelector(".deck-browser__confirm");
  assert.ok(confirmButton, "The confirm button should render.");
  assert.equal(
    confirmButton.disabled,
    false,
    "Pokegear should allow confirming with zero selected cards.",
  );

  const supporterButton = window.document.querySelector('.deck-browser-card[title="Nemona"]');
  const nonSupporterButton = window.document.querySelector('.deck-browser-card[title="Mareep"]');
  const hiddenSupporterButton = window.document.querySelector('.deck-browser-card[title="Youngster"]');
  assert.ok(supporterButton, "The visible supporter should render.");
  assert.ok(nonSupporterButton, "A visible non-supporter should render.");
  assert.equal(hiddenSupporterButton, null, "Cards outside the top seven should stay hidden.");
  assert.ok(
    supporterButton.classList.contains("is-selectable"),
    "Supporters in the top seven should be selectable.",
  );
  assert.ok(
    supporterButton.classList.contains("is-match"),
    "Supporters in the top seven should be highlighted as matches.",
  );
  assert.ok(
    !nonSupporterButton.classList.contains("is-selectable"),
    "Non-supporters in the top seven should not be selectable.",
  );
  assert.ok(
    !nonSupporterButton.classList.contains("is-match"),
    "Non-supporters should not be highlighted as matching Pokegear.",
  );
});
