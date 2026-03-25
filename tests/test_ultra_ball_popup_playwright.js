const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const { execFileSync } = require("node:child_process");

const ROOT = path.resolve(__dirname, "..");

if (fs.existsSync(path.join(ROOT, ".playwright-browsers"))) {
  process.env.PLAYWRIGHT_BROWSERS_PATH = path.join(ROOT, ".playwright-browsers");
}

const { chromium } = require("@playwright/test");

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
`;

  return JSON.parse(
    execFileSync("python3", ["-c", pythonScript], {
      cwd: ROOT,
      encoding: "utf8",
    }),
  );
}

function loadFrontendHtml() {
  return fs
    .readFileSync(path.join(ROOT, "frontend/index.html"), "utf8")
    .replace('    <link rel="stylesheet" href="/styles.css" />\n', "")
    .replace('    <script src="/selection-state.js" defer></script>\n', "")
    .replace('    <script src="/app.js" defer></script>\n', "");
}

function resolveChromiumLaunchOptions() {
  const repoBrowsersRoot = path.join(ROOT, ".playwright-browsers");
  const repoChromiumExecutable = path.join(
    repoBrowsersRoot,
    "chromium-1208",
    "chrome-mac-arm64",
    "Google Chrome for Testing.app",
    "Contents",
    "MacOS",
    "Google Chrome for Testing",
  );
  if (fs.existsSync(repoChromiumExecutable)) {
    return {
      headless: true,
      executablePath: repoChromiumExecutable,
    };
  }
  return { headless: true };
}

test("Ultra Ball discard popup flows into deck search and confirms the selected Pokemon into hand in Chromium", async () => {
  const snapshot = loadUltraBallSnapshot();
  const browser = await chromium.launch(resolveChromiumLaunchOptions());
  const page = await browser.newPage();

  try {
    await page.addInitScript(() => {
      window.fetch = async () => {
        throw new Error("Unexpected fetch call in browser popup test.");
      };
      window.requestAnimationFrame = (callback) => {
        callback();
        return 1;
      };
      window.cancelAnimationFrame = () => {};
    });

    await page.setContent(loadFrontendHtml(), { waitUntil: "domcontentloaded" });
    await page.addScriptTag({ content: fs.readFileSync(path.join(ROOT, "frontend/selection-state.js"), "utf8") });
    await page.addScriptTag({ content: fs.readFileSync(path.join(ROOT, "frontend/app.js"), "utf8") });
    await page.waitForFunction(() => Boolean(window.__TCG_APP_TEST_API__));

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

    await page.evaluate((payload) => {
      const app = window.__TCG_APP_TEST_API__;
      app.setCurrentState(payload);
      app.render(payload);
    }, snapshot);

    await page.locator('#player-hand img[alt="Ultra Ball"]').click();
    await page.evaluate(() => {
      document.getElementById("player-board-panel")?.dispatchEvent(
        new MouseEvent("click", { bubbles: true }),
      );
    });

    await page.waitForFunction(
      () => document.querySelector(".deck-browser__eyebrow")?.textContent === "Discard Cards",
    );

    const discardPotionButton = page.locator('.deck-browser-card[title="Potion"]');
    const discardSwitchButton = page.locator('.deck-browser-card[title="Switch"]');
    const discardConfirmButton = page.locator(".deck-browser__confirm");

    await discardPotionButton.click();
    await discardSwitchButton.click();
    assert.equal(await discardConfirmButton.isDisabled(), false, "Discard confirm should enable after two choices.");
    await discardConfirmButton.click();

    await page.waitForFunction(
      () => document.querySelector(".deck-browser__eyebrow")?.textContent === "Full Deck Search",
    );

    const cardButton = page.locator(`.deck-browser-card[data-instance-id="${selectedDeckCardId}"]`);
    await cardButton.click();
    await expectClass(page, `.deck-browser-card[data-instance-id="${selectedDeckCardId}"]`, "is-selected");

    const confirmButton = page.locator(".deck-browser__confirm");
    assert.equal(await confirmButton.isDisabled(), false, "Confirm should enable after choosing a Pokemon.");

    await page.evaluate(({ snapshot, selectedDeckCardId }) => {
      const app = window.__TCG_APP_TEST_API__;
      app.setSubmitActionOverride(async (actionView) => {
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
        window.__submittedUltraBallAction = actionView;
      });
    }, { snapshot, selectedDeckCardId });

    await confirmButton.click();

    const submittedSearchIds = await page.evaluate(
      () => window.__submittedUltraBallAction?.action?.search_deck_ids || null,
    );
    assert.deepEqual(
      submittedSearchIds,
      [selectedDeckCardId],
      "Confirm should submit the selected Pokemon search id.",
    );

    const chosenHandCard = page.locator(`#player-hand img[alt="${selectedDeckCard.name}"]`);
    assert.equal(
      await chosenHandCard.count(),
      1,
      "After confirm, the selected Pokemon should appear in the hand UI.",
    );
  } finally {
    await browser.close();
  }
});

async function expectClass(page, selector, className) {
  await page.waitForFunction(
    ({ selector, className }) =>
      document.querySelector(selector)?.classList.contains(className) || false,
    { selector, className },
  );
}
