const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const { execFileSync } = require("node:child_process");
const { JSDOM } = require("jsdom");

const ROOT = path.resolve(__dirname, "..");

function loadCallForFamilySnapshot({ benchCount = 0 } = {}) {
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
bench_count = ${benchCount}

def take(player, name: str) -> str:
    for zone_name in ("hand", "deck", "discard"):
        zone = getattr(player, zone_name)
        for instance_id in list(zone):
            if card_definition(session.state, instance_id).name != name:
                continue
            zone.remove(instance_id)
            return instance_id
    raise AssertionError(f"Missing {name}")

def find(player, name: str) -> str:
    for zone_name in ("hand", "deck", "discard"):
        zone = getattr(player, zone_name)
        for instance_id in zone:
            if card_definition(session.state, instance_id).name == name:
                return instance_id
    raise AssertionError(f"Missing {name}")

human.active = PokemonInPlay(stack=[take(human, "Squawkabilly")])
human.active.attached_energy = [take(human, "Basic Fighting Energy")]
human.bench = []
opponent.active = PokemonInPlay(stack=[take(opponent, "Mareep")])
opponent.bench = []

for _ in range(bench_count):
    for zone_name in ("hand", "deck", "discard"):
        zone = getattr(human, zone_name)
        chosen_id = next(
            (
                instance_id
                for instance_id in list(zone)
                if card_definition(session.state, instance_id).kind == "pokemon"
                and card_definition(session.state, instance_id).is_basic
            ),
            None,
        )
        if chosen_id is None:
            continue
        zone.remove(chosen_id)
        human.bench.append(PokemonInPlay(stack=[chosen_id]))
        break
    else:
        raise AssertionError("Missing Basic Pokemon for bench setup")

if bench_count == 0:
    riolu_id = find(human, "Riolu")
    lechonk_id = find(human, "Lechonk")
    oinkologne_id = find(human, "Oinkologne")
    prioritized_ids = {riolu_id, lechonk_id, oinkologne_id}
    human.deck = [riolu_id, oinkologne_id, lechonk_id] + [
        instance_id
        for instance_id in human.deck
        if instance_id not in prioritized_ids
    ]

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
    throw new Error("Unexpected fetch call in Call for Family popup test.");
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

function findCallForFamilyButton(window) {
  return window.document.querySelector('#player-active .attack-chip-button[data-attack-index="0"]');
}

test("Call for Family attack button opens full deck search and only basics are selectable", async () => {
  const snapshot = loadCallForFamilySnapshot();
  const dom = setupDom();
  const { window } = dom;
  const app = window.__TCG_APP_TEST_API__;

  app.setCurrentState(snapshot);
  app.render(snapshot);

  const attackButton = findCallForFamilyButton(window);
  assert.ok(attackButton, "Call for Family attack button should render.");
  assert.equal(attackButton.disabled, false, "Call for Family should be usable.");

  attackButton.dispatchEvent(new window.MouseEvent("click", { bubbles: true }));

  const overlay = window.document.getElementById("deck-browser-overlay");
  assert.ok(overlay, "Deck browser overlay should exist.");
  assert.equal(overlay.hidden, false, "Clicking the attack should open the deck browser.");
  assert.equal(
    window.document.querySelector(".deck-browser__eyebrow")?.textContent,
    "Full Deck Search",
    "Call for Family should browse the full deck.",
  );
  assert.equal(
    window.document.querySelector(".deck-browser h3")?.textContent,
    "Call for Family",
    "The popup title should use the attack name.",
  );

  const confirmButton = window.document.querySelector(".deck-browser__confirm");
  assert.ok(confirmButton, "Confirm button should render.");
  assert.equal(
    confirmButton.disabled,
    false,
    "Call for Family should allow confirming without selecting any cards.",
  );

  const rioluButton = window.document.querySelector('.deck-browser-card[title="Riolu"]');
  const lechonkButton = window.document.querySelector('.deck-browser-card[title="Lechonk"]');
  const oinkologneButton = window.document.querySelector('.deck-browser-card[title="Oinkologne"]');
  assert.ok(rioluButton, "Riolu should appear in the deck browser.");
  assert.ok(lechonkButton, "Lechonk should appear in the deck browser.");
  assert.ok(oinkologneButton, "Oinkologne should appear in the deck browser.");
  assert.ok(rioluButton.classList.contains("is-selectable"), "Riolu should be selectable.");
  assert.ok(rioluButton.classList.contains("is-match"), "Riolu should match the attack filter.");
  assert.ok(lechonkButton.classList.contains("is-selectable"), "Lechonk should be selectable.");
  assert.ok(lechonkButton.classList.contains("is-match"), "Lechonk should match the attack filter.");
  assert.ok(
    !oinkologneButton.classList.contains("is-selectable"),
    "Evolution Pokemon should not be selectable.",
  );
  assert.ok(
    !oinkologneButton.classList.contains("is-match"),
    "Evolution Pokemon should not be highlighted as matches.",
  );
});

test("Call for Family popup confirms two selected basics as the chosen attack action", async () => {
  const snapshot = loadCallForFamilySnapshot();
  const dom = setupDom();
  const { window } = dom;
  const app = window.__TCG_APP_TEST_API__;

  app.setCurrentState(snapshot);
  app.render(snapshot);

  const attackButton = findCallForFamilyButton(window);
  assert.ok(attackButton, "Call for Family attack button should render.");
  attackButton.dispatchEvent(new window.MouseEvent("click", { bubbles: true }));

  const rioluButton = window.document.querySelector('.deck-browser-card[title="Riolu"]');
  const lechonkButton = window.document.querySelector('.deck-browser-card[title="Lechonk"]');
  const confirmButton = window.document.querySelector(".deck-browser__confirm");
  assert.ok(rioluButton, "Riolu should be selectable in the deck browser.");
  assert.ok(lechonkButton, "Lechonk should be selectable in the deck browser.");
  assert.ok(confirmButton, "Confirm button should render.");

  let submittedAction = null;
  app.setSubmitActionOverride(async (actionView) => {
    submittedAction = actionView;
  });

  rioluButton.dispatchEvent(new window.MouseEvent("click", { bubbles: true }));
  lechonkButton.dispatchEvent(new window.MouseEvent("click", { bubbles: true }));

  assert.ok(rioluButton.classList.contains("is-selected"), "Riolu should show as selected.");
  assert.ok(lechonkButton.classList.contains("is-selected"), "Lechonk should show as selected.");

  confirmButton.dispatchEvent(new window.MouseEvent("click", { bubbles: true }));

  assert.ok(submittedAction, "Confirm should submit an attack action.");
  assert.deepEqual(
    [...submittedAction.action.search_deck_ids].sort(),
    [
      rioluButton.dataset.instanceId,
      lechonkButton.dataset.instanceId,
    ].sort(),
    "Confirm should submit both chosen Basic Pokemon ids.",
  );
  assert.equal(
    submittedAction.action.attack_index,
    0,
    "The submitted action should still point at Call for Family.",
  );
});

test("Call for Family opens deck search with a one-card cap when only one bench space remains", async () => {
  const snapshot = loadCallForFamilySnapshot({ benchCount: 4 });
  const dom = setupDom();
  const { window } = dom;
  const app = window.__TCG_APP_TEST_API__;

  app.setCurrentState(snapshot);
  app.render(snapshot);

  const attackButton = findCallForFamilyButton(window);
  assert.ok(attackButton, "Call for Family attack button should render.");
  attackButton.dispatchEvent(new window.MouseEvent("click", { bubbles: true }));

  const overlay = window.document.getElementById("deck-browser-overlay");
  assert.ok(overlay, "Deck browser overlay should exist.");
  assert.equal(
    overlay.hidden,
    false,
    "The deck browser should still open when one bench space remains.",
  );
  const metaText = window.document.querySelector(".deck-browser__meta")?.textContent || "";
  assert.match(
    metaText,
    /Choose up to 1 card to place into bench/i,
    "The popup should only allow selecting one Basic Pokemon.",
  );

  const selectableButtons = [...window.document.querySelectorAll(".deck-browser-card.is-selectable")];
  assert.ok(selectableButtons.length > 0, "At least one Basic Pokemon should still be selectable.");
  selectableButtons[0].dispatchEvent(new window.MouseEvent("click", { bubbles: true }));
  assert.ok(
    selectableButtons[0].classList.contains("is-selected"),
    "Selecting one Basic Pokemon should still work with one slot left.",
  );

  const nonSelectedButtons = selectableButtons.slice(1);
  if (nonSelectedButtons.length) {
    nonSelectedButtons[0].dispatchEvent(new window.MouseEvent("click", { bubbles: true }));
    assert.ok(
      nonSelectedButtons[0].classList.contains("is-selected"),
      "The popup should let the player switch which single Pokemon is selected.",
    );
    assert.ok(
      !selectableButtons[0].classList.contains("is-selected"),
      "Only one Pokemon should stay selected when one bench slot remains.",
    );
  }
});

test("Call for Family auto-submits without opening deck search when the bench is full", async () => {
  const snapshot = loadCallForFamilySnapshot({ benchCount: 5 });
  const dom = setupDom();
  const { window } = dom;
  const app = window.__TCG_APP_TEST_API__;

  app.setCurrentState(snapshot);
  app.render(snapshot);

  let submittedAction = null;
  app.setSubmitActionOverride(async (actionView) => {
    submittedAction = actionView;
  });

  const attackButton = findCallForFamilyButton(window);
  assert.ok(attackButton, "Call for Family attack button should render.");
  attackButton.dispatchEvent(new window.MouseEvent("click", { bubbles: true }));

  const overlay = window.document.getElementById("deck-browser-overlay");
  assert.ok(overlay, "Deck browser overlay should exist.");
  assert.equal(
    overlay.hidden,
    true,
    "The deck browser should stay closed when the bench is full.",
  );
  assert.ok(submittedAction, "Call for Family should submit immediately when the bench is full.");
  assert.deepEqual(
    submittedAction.action.search_deck_ids,
    [],
    "The auto-submitted action should skip the deck search.",
  );
  assert.equal(
    submittedAction.action.attack_index,
    0,
    "The submitted action should still be Call for Family.",
  );
});
