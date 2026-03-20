const test = require("node:test");
const assert = require("node:assert/strict");

const {
  deriveInteractionContext,
  findBenchPlayActionForSelection,
  isBoardRefClickable,
  refsMatch,
  resolveSelectedBoardTargetClick,
  resolveSelectedCardClick,
  sanitizeSelectionState,
} = require("../frontend/selection-state.js");

function makeRefs() {
  return {
    playerActive: { player_index: 0, zone: "active", bench_index: null, instance_id: "active-1" },
    playerBench: { player_index: 0, zone: "bench", bench_index: 0, instance_id: "bench-1" },
    playerEnergy: { player_index: 0, zone: "energy", bench_index: null, instance_id: null },
    opponentActive: { player_index: 1, zone: "active", bench_index: null, instance_id: "opp-active-1" },
  };
}

function makeState() {
  const refs = makeRefs();
  return {
    winner: null,
    pending_promotion_for: null,
    legal_actions: [],
    players: [
      {
        hand: [
          { instance_id: "hand-energy", name: "Fire Energy" },
          { instance_id: "hand-potion", name: "Potion" },
          { instance_id: "hand-basic", name: "Growlithe" },
        ],
        active: { name: "Charmander", ref: refs.playerActive },
        bench: [{ name: "Growlithe", ref: refs.playerBench }],
        energy_zone: [{ instance_id: "attached-1" }],
      },
      {
        hand: [],
        active: { name: "Squirtle", ref: refs.opponentActive },
        bench: [],
        energy_zone: [],
      },
    ],
  };
}

test("sanitizeSelectionState clears stale hand and board selections", () => {
  const state = makeState();
  const sanitized = sanitizeSelectionState(
    {
      selectedCardId: "missing-card",
      selectedBoardTarget: { player_index: 0, zone: "bench", bench_index: 2, instance_id: "missing-bench" },
    },
    state,
  );

  assert.equal(sanitized.selectedCardId, null);
  assert.equal(sanitized.selectedBoardTarget, null);
});

test("sanitizeSelectionState clears a stale idle energy target", () => {
  const state = makeState();
  const refs = makeRefs();
  const sanitized = sanitizeSelectionState(
    {
      selectedCardId: null,
      selectedBoardTarget: refs.playerEnergy,
    },
    state,
  );

  assert.equal(sanitized.selectedBoardTarget, null);
});

test("sanitizeSelectionState keeps the energy target selected for a matching hand action", () => {
  const state = makeState();
  const refs = makeRefs();
  state.legal_actions = [
    {
      type: "play_energy",
      source: { zone: "hand", instance_id: "hand-energy" },
      target: refs.playerEnergy,
    },
  ];

  const sanitized = sanitizeSelectionState(
    {
      selectedCardId: "hand-energy",
      selectedBoardTarget: refs.playerEnergy,
    },
    state,
  );

  assert.ok(refsMatch(sanitized.selectedBoardTarget, refs.playerEnergy));
});

test("sanitizeSelectionState keeps an empty active spot selected", () => {
  const state = makeState();
  const activeSpotRef = { player_index: 0, zone: "active", bench_index: null, instance_id: null };
  state.players[0].active = null;

  const sanitized = sanitizeSelectionState(
    {
      selectedCardId: null,
      selectedBoardTarget: activeSpotRef,
    },
    state,
  );

  assert.ok(refsMatch(sanitized.selectedBoardTarget, activeSpotRef));
});

test("deriveInteractionContext highlights board targets for a selected hand card", () => {
  const state = makeState();
  const refs = makeRefs();
  state.legal_actions = [
    {
      type: "play_energy",
      source: { zone: "hand", instance_id: "hand-energy" },
      target: refs.playerEnergy,
    },
  ];

  const context = deriveInteractionContext(state, {
    selectedCardId: "hand-energy",
    selectedBoardTarget: null,
  });

  assert.equal(context.actions.length, 0);
  assert.match(context.instructions, /Choose a highlighted target on your board/);
  assert.equal(context.highlightedTargets.length, 1);
  assert.ok(refsMatch(context.highlightedTargets[0], refs.playerEnergy));
});

test("isBoardRefClickable keeps the energy pile inert when no hand card is selected", () => {
  const state = makeState();
  const refs = makeRefs();

  const clickable = isBoardRefClickable({
    state,
    uiState: {
      selectedCardId: null,
      selectedBoardTarget: null,
    },
    context: {
      highlightedTargets: [],
    },
    aiIsRunning: false,
    ref: refs.playerEnergy,
  });

  assert.equal(clickable, false);
});

test("isBoardRefClickable allows the energy pile when it is a highlighted hand-card target", () => {
  const state = makeState();
  const refs = makeRefs();
  state.legal_actions = [
    {
      type: "play_energy",
      source: { zone: "hand", instance_id: "hand-energy" },
      target: refs.playerEnergy,
    },
  ];
  const context = deriveInteractionContext(state, {
    selectedCardId: "hand-energy",
    selectedBoardTarget: null,
  });

  const clickable = isBoardRefClickable({
    state,
    uiState: {
      selectedCardId: "hand-energy",
      selectedBoardTarget: null,
    },
    context,
    aiIsRunning: false,
    ref: refs.playerEnergy,
  });

  assert.equal(clickable, true);
});

test("resolveSelectedBoardTargetClick auto-submits a hand-targeted action", () => {
  const state = makeState();
  const refs = makeRefs();
  const action = {
    type: "play_energy",
    source: { zone: "hand", instance_id: "hand-energy" },
    target: refs.playerEnergy,
  };
  state.legal_actions = [action];

  const result = resolveSelectedBoardTargetClick({
    state,
    uiState: {
      selectedCardId: "hand-energy",
      selectedBoardTarget: null,
    },
    targetRef: refs.playerEnergy,
    aiIsRunning: false,
  });

  assert.equal(result.autoAction, action);
  assert.equal(result.nextUiState, null);
});

test("resolveSelectedCardClick auto-submits when the target is already selected", () => {
  const state = makeState();
  const refs = makeRefs();
  const action = {
    type: "play_energy",
    source: { zone: "hand", instance_id: "hand-energy" },
    target: refs.playerEnergy,
  };
  state.legal_actions = [action];

  const result = resolveSelectedCardClick({
    state,
    uiState: {
      selectedCardId: null,
      selectedBoardTarget: refs.playerEnergy,
    },
    instanceId: "hand-energy",
    aiIsRunning: false,
  });

  assert.equal(result.autoAction, action);
  assert.equal(result.nextUiState, null);
});

test("resolveSelectedBoardTargetClick hands off to the active Pokemon when a hand target is invalid", () => {
  const state = makeState();
  const refs = makeRefs();
  state.legal_actions = [
    {
      type: "play_energy",
      source: { zone: "hand", instance_id: "hand-energy" },
      target: refs.playerEnergy,
    },
    {
      type: "attack",
      source: refs.playerActive,
      target: refs.opponentActive,
    },
  ];

  const result = resolveSelectedBoardTargetClick({
    state,
    uiState: {
      selectedCardId: "hand-energy",
      selectedBoardTarget: null,
    },
    targetRef: refs.playerActive,
    aiIsRunning: false,
  });

  assert.equal(result.autoAction, null);
  assert.equal(result.nextUiState.selectedCardId, null);
  assert.ok(refsMatch(result.nextUiState.selectedBoardTarget, refs.playerActive));
});

test("resolveSelectedBoardTargetClick keeps the hand selection when the invalid board target has no action", () => {
  const state = makeState();
  const refs = makeRefs();
  state.legal_actions = [
    {
      type: "play_energy",
      source: { zone: "hand", instance_id: "hand-energy" },
      target: refs.playerEnergy,
    },
  ];

  const result = resolveSelectedBoardTargetClick({
    state,
    uiState: {
      selectedCardId: "hand-energy",
      selectedBoardTarget: null,
    },
    targetRef: refs.playerBench,
    aiIsRunning: false,
  });

  assert.equal(result.autoAction, null);
  assert.equal(result.nextUiState.selectedCardId, "hand-energy");
  assert.equal(result.nextUiState.selectedBoardTarget, null);
});

test("resolveSelectedBoardTargetClick uses a two-step promotion flow", () => {
  const state = makeState();
  const refs = makeRefs();
  const action = {
    type: "promote",
    source: refs.playerBench,
    target: { player_index: 0, zone: "active", bench_index: null, instance_id: null },
  };
  state.pending_promotion_for = 0;
  state.players[0].active = null;
  state.legal_actions = [action];

  const selectBenchResult = resolveSelectedBoardTargetClick({
    state,
    uiState: {
      selectedCardId: null,
      selectedBoardTarget: null,
    },
    targetRef: refs.playerBench,
    aiIsRunning: false,
  });

  assert.equal(selectBenchResult.autoAction, null);
  assert.ok(refsMatch(selectBenchResult.nextUiState.selectedBoardTarget, refs.playerBench));

  const promoteResult = resolveSelectedBoardTargetClick({
    state,
    uiState: {
      selectedCardId: null,
      selectedBoardTarget: refs.playerBench,
    },
    targetRef: { player_index: 0, zone: "active", bench_index: null, instance_id: null },
    aiIsRunning: false,
  });

  assert.equal(promoteResult.autoAction, action);
  assert.equal(promoteResult.nextUiState, null);
});

test("resolveSelectedBoardTargetClick toggles the same target off", () => {
  const refs = makeRefs();
  const result = resolveSelectedBoardTargetClick({
    state: makeState(),
    uiState: {
      selectedCardId: null,
      selectedBoardTarget: refs.playerBench,
    },
    targetRef: refs.playerBench,
    aiIsRunning: false,
  });

  assert.equal(result.autoAction, null);
  assert.equal(result.nextUiState.selectedBoardTarget, null);
});

test("findBenchPlayActionForSelection finds the matching bench action", () => {
  const state = makeState();
  const action = {
    type: "bench_basic",
    source: { zone: "hand", instance_id: "hand-basic" },
  };
  state.legal_actions = [action];

  assert.equal(
    findBenchPlayActionForSelection(state, "hand-basic", false),
    action,
  );
});

test("deriveInteractionContext treats opponent-only targets as direct actions", () => {
  const state = makeState();
  const refs = makeRefs();
  const action = {
    type: "attack",
    source: refs.playerActive,
    target: refs.opponentActive,
  };
  state.legal_actions = [action];

  const context = deriveInteractionContext(state, {
    selectedCardId: null,
    selectedBoardTarget: refs.playerActive,
  });

  assert.deepEqual(context.actions, [action]);
  assert.equal(context.highlightedTargets.length, 0);
  assert.doesNotMatch(context.instructions, /highlighted board targets/);
});

test("deriveInteractionContext highlights the empty active spot during promotion", () => {
  const state = makeState();
  const refs = makeRefs();
  const activeSpotRef = { player_index: 0, zone: "active", bench_index: null, instance_id: null };
  state.players[0].active = null;
  state.pending_promotion_for = 0;
  state.legal_actions = [
    {
      type: "promote",
      source: refs.playerBench,
      target: activeSpotRef,
    },
  ];

  const context = deriveInteractionContext(state, {
    selectedCardId: null,
    selectedBoardTarget: refs.playerBench,
  });

  assert.equal(context.actions.length, 0);
  assert.equal(context.highlightedTargets.length, 1);
  assert.ok(refsMatch(context.highlightedTargets[0], activeSpotRef));
});

test("deriveInteractionContext highlights compatible hand cards for a selected board target", () => {
  const state = makeState();
  const refs = makeRefs();
  state.legal_actions = [
    {
      type: "play_potion",
      source: { zone: "hand", instance_id: "hand-potion" },
      target: refs.playerActive,
    },
  ];

  const context = deriveInteractionContext(state, {
    selectedCardId: null,
    selectedBoardTarget: refs.playerActive,
  });

  assert.equal(context.actions.length, 0);
  assert.ok(context.highlightedHandIds.has("hand-potion"));
  assert.match(context.instructions, /highlighted hand cards/);
});

test("isBoardRefClickable keeps actionable board refs clickable while a hand card is selected", () => {
  const refs = makeRefs();
  const context = {
    highlightedTargets: [refs.playerEnergy],
  };
  const state = makeState();
  state.legal_actions = [
    {
      type: "attack",
      source: refs.playerActive,
      target: refs.opponentActive,
    },
  ];

  assert.equal(
    isBoardRefClickable({
      state,
      uiState: {
        selectedCardId: "hand-energy",
        selectedBoardTarget: null,
      },
      context,
      aiIsRunning: false,
      ref: refs.playerEnergy,
    }),
    true,
  );
  assert.equal(
    isBoardRefClickable({
      state,
      uiState: {
        selectedCardId: "hand-energy",
        selectedBoardTarget: null,
      },
      context,
      aiIsRunning: false,
      ref: refs.playerActive,
    }),
    true,
  );
  assert.equal(
    isBoardRefClickable({
      state,
      uiState: {
        selectedCardId: "hand-energy",
        selectedBoardTarget: null,
      },
      context,
      aiIsRunning: false,
      ref: refs.playerBench,
    }),
    false,
  );
});
