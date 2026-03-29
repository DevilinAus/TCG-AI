const SESSION_STORAGE_KEY = "tcg_ai_session_id";
const AI_HUMAN_DELAY_MIN_MS = 5000;
const AI_HUMAN_DELAY_MAX_MS = 8000;
const FALLBACK_AI_STEP_DELAY_MS = 6500;
const FACE_DOWN_CARD_IMAGE_URL = "/assets/cards/shared/card-back.png";
const BENCH_LIMIT = 5;

let currentState = null;
let lobbyState = null;
let previousState = null;
let aiIsRunning = false;
let aiAutoRunQueued = false;
let aiAutoRunPaused = false;
let stateRequestEpoch = 0;
let submitActionOverride = null;

const uiState = {
  selectedCardId: null,
  selectedDiscardIds: [],
  discardBrowseRequest: null,
  discardBrowseSelectedIds: [],
  selectedBoardTarget: null,
  pendingAttackActionIds: [],
  attackOptionRequest: null,
  availableContextActions: [],
  deckBrowseRequest: null,
  deckBrowseSelectedIds: [],
  selectedGameModeId: null,
  selectedTrainerId: null,
  selectedHumanDeckId: null,
  selectedStandardAiMode: null,
  standardAiModePending: false,
  standardMlStatus: null,
};

const pointerState = {
  clientX: null,
  clientY: null,
};

const deckBrowserDragState = {
  pointerId: null,
  startClientX: 0,
  startClientY: 0,
  startScrollLeft: 0,
  isDragging: false,
};

const {
  deriveInteractionContext,
  findBackgroundPlayActionForSelection,
  findBenchPlayActionForSelection,
  findBoardTargetLabelForPlayer,
  isBoardRefClickable,
  refsMatch,
  resolveSelectedBoardTargetClick,
  resolveSelectedCardClick,
  sanitizeSelectionState,
} = globalThis.TcgUiSelection;

async function requestJson(url, options = {}) {
  const response = await fetch(url, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });

  const payload = await response.json();
  if (!response.ok) {
    const error = new Error(payload.error || "Request failed");
    error.code = payload.code;
    throw error;
  }
  return payload;
}

function sleep(milliseconds) {
  return new Promise((resolve) => {
    window.setTimeout(resolve, milliseconds);
  });
}

function randomDelay(minMilliseconds, maxMilliseconds) {
  const min = Math.ceil(minMilliseconds);
  const max = Math.floor(maxMilliseconds);
  return Math.floor(Math.random() * (max - min + 1)) + min;
}

function waitForPaint() {
  return new Promise((resolve) => {
    window.requestAnimationFrame(() => resolve());
  });
}

function syncAdaptiveActiveCardLayout() {
  const activePockets = document.querySelectorAll(".player-active-pocket, .opponent-active-pocket");
  for (const pocket of activePockets) {
    pocket.classList.remove("has-wide-active-card");
  }

  const activeCards = document.querySelectorAll(".active-slot .board-card:not(.is-compact):not(.active-slot-placeholder)");
  let anyCardNeedsWideLayout = false;
  for (const card of activeCards) {
    card.classList.remove("is-wide-layout");
    const attackToplines = card.querySelectorAll(".attack-topline");
    const needsWideLayout = [...attackToplines].some((topline) => {
      const stats = topline.querySelector(".attack-stats");
      if (!stats) {
        return false;
      }
      return topline.scrollWidth > topline.clientWidth || stats.offsetTop > topline.offsetTop;
    });
    if (!needsWideLayout) {
      continue;
    }

    anyCardNeedsWideLayout = true;
    card.classList.add("is-wide-layout");
  }

  if (anyCardNeedsWideLayout) {
    for (const pocket of activePockets) {
      pocket.classList.add("has-wide-active-card");
    }
  }

  const measuredWidths = [...activeCards]
    .map((card) => Math.ceil(card.getBoundingClientRect().width))
    .filter((width) => width > 0);
  const fallbackWidth = anyCardNeedsWideLayout ? 540 : 480;
  const resolvedWidth = measuredWidths.length ? Math.max(...measuredWidths) : fallbackWidth;
  document.documentElement.style.setProperty("--active-card-runtime-width", `${resolvedWidth}px`);
}

function syncCursorForPointerPosition() {
  if (pointerState.clientX === null || pointerState.clientY === null) {
    document.body.style.cursor = "";
    return;
  }

  const hoveredElement = document.elementFromPoint(pointerState.clientX, pointerState.clientY);
  const interactiveAncestor = hoveredElement?.closest(
    [
      ".mini-card.is-clickable",
      ".board-card.is-clickable",
      ".active-slot-placeholder.is-clickable",
      ".attack-chip-button:not(:disabled)",
      ".end-turn-button.is-ready:not(:disabled)",
      ".action-button:not(:disabled)",
      "button:not(:disabled)",
      "select:not(:disabled)",
      "label",
    ].join(", "),
  );
  document.body.style.cursor = interactiveAncestor ? "pointer" : "";
}

function handlePointerMove(event) {
  pointerState.clientX = event.clientX;
  pointerState.clientY = event.clientY;
  syncCursorForPointerPosition();
  syncAttackDragIndicatorPosition();
}

function handlePointerLeaveWindow() {
  pointerState.clientX = null;
  pointerState.clientY = null;
  document.body.style.cursor = "";
  syncAttackDragIndicatorPosition();
}

function getStoredSessionId() {
  return window.sessionStorage.getItem(SESSION_STORAGE_KEY);
}

function setStoredSessionId(sessionId) {
  window.sessionStorage.setItem(SESSION_STORAGE_KEY, sessionId);
}

function clearStoredSessionId() {
  window.sessionStorage.removeItem(SESSION_STORAGE_KEY);
}

function resetToLobbyState(gameModeId = null) {
  stateRequestEpoch += 1;
  clearStoredSessionId();
  currentState = null;
  lobbyState = null;
  previousState = null;
  aiIsRunning = false;
  aiAutoRunQueued = false;
  aiAutoRunPaused = false;
  resetSelections();
  uiState.selectedGameModeId = gameModeId || null;
}

async function refreshGame() {
  const requestEpoch = stateRequestEpoch;
  const sessionId = getStoredSessionId();
  if (!sessionId) {
    await loadLobby();
    return;
  }

  try {
    const payload = await requestJson(`/api/game?session_id=${encodeURIComponent(sessionId)}`);
    if (requestEpoch !== stateRequestEpoch) {
      return;
    }
    currentState = payload;
    lobbyState = null;
    aiAutoRunPaused = false;
    uiState.selectedGameModeId = uiState.selectedGameModeId || currentState.game_mode || null;
    uiState.selectedTrainerId = uiState.selectedTrainerId || currentState.ai_trainer?.id || null;
    uiState.selectedHumanDeckId = uiState.selectedHumanDeckId || currentState.human_deck_id || null;
    uiState.selectedStandardAiMode = currentState.standard_ai_mode || "local";
    sanitizeSelections(currentState);
    render(currentState);
    maybeRunAiTurn(currentState);
  } catch (error) {
    if (requestEpoch !== stateRequestEpoch) {
      return;
    }
    if (error.code === "session_not_found" || error.code === "missing_session_id") {
      clearStoredSessionId();
      currentState = null;
      previousState = null;
      resetSelections();
      await loadLobby();
      return;
    }
    updateStatus(error.message);
  }
}

async function loadLobby(requestedGameModeId = null) {
  const requestEpoch = stateRequestEpoch;
  try {
    const query = requestedGameModeId ? `?game_mode=${encodeURIComponent(requestedGameModeId)}` : "";
    const payload = await requestJson(`/api/lobby${query}`);
    if (requestEpoch !== stateRequestEpoch) {
      return;
    }
    lobbyState = payload;
    currentState = null;
    previousState = null;
    aiAutoRunPaused = false;
    resetSelections();
    uiState.selectedGameModeId = lobbyState.game_mode || null;
    if (!uiState.selectedStandardAiMode) {
      uiState.selectedStandardAiMode = lobbyState.standard_ai_mode || "local";
    }

    const availableTrainerIds = new Set((lobbyState.available_trainers || []).map((trainer) => trainer.id));
    if (!availableTrainerIds.has(uiState.selectedTrainerId)) {
      uiState.selectedTrainerId = lobbyState.ai_trainer?.id || null;
    }

    const availableDeckIds = new Set((lobbyState.available_decks || []).map((deck) => deck.id));
    if (!availableDeckIds.has(uiState.selectedHumanDeckId)) {
      uiState.selectedHumanDeckId =
        lobbyState.human_deck_id ||
        lobbyState.available_decks?.find((deck) => deck.selected)?.id ||
        lobbyState.available_decks?.[0]?.id ||
        null;
    }
    renderLobby(lobbyState);
  } catch (error) {
    if (requestEpoch !== stateRequestEpoch) {
      return;
    }
    updateStatus(error.message);
  }
}

async function newGame(payloadOverrides = null) {
  const requestEpoch = stateRequestEpoch;
  try {
    const payloadBody = {};
    const selectionState = currentState || lobbyState;
    const gameMode = resolveGameMode(selectionState);
    if (gameMode) {
      payloadBody.game_mode = gameMode;
    }
    const trainerId = resolveTrainerSelection(selectionState);
    if (trainerId) {
      payloadBody.trainer_id = trainerId;
    }
    const humanDeckId = resolveHumanDeckSelection(selectionState);
    if (humanDeckId) {
      payloadBody.human_deck_id = humanDeckId;
    }
    if (gameMode === "standard") {
      payloadBody.standard_ai_mode = resolveStandardAiMode(selectionState);
    }
    if (payloadOverrides && typeof payloadOverrides === "object") {
      Object.assign(payloadBody, payloadOverrides);
    }
    const payload = await requestJson("/api/new-game", {
      method: "POST",
      body: JSON.stringify(payloadBody),
    });
    if (requestEpoch !== stateRequestEpoch) {
      return;
    }
    currentState = payload;
    lobbyState = null;
    previousState = null;
    aiAutoRunPaused = false;
    uiState.selectedGameModeId = payload.game_mode || uiState.selectedGameModeId;
    uiState.selectedTrainerId = payload.ai_trainer?.id || uiState.selectedTrainerId;
    uiState.selectedHumanDeckId = payload.human_deck_id || uiState.selectedHumanDeckId;
    uiState.selectedStandardAiMode = payload.standard_ai_mode || "local";
    resetSelections();
    setStoredSessionId(payload.session_id);
    render(currentState);
    maybeRunAiTurn(currentState);
    return true;
  } catch (error) {
    if (requestEpoch !== stateRequestEpoch) {
      return false;
    }
    updateStatus(error.message);
    return false;
  }
}

async function submitAction(actionView) {
  if (submitActionOverride) {
    await submitActionOverride(actionView);
    return;
  }
  if (!currentState || aiIsRunning) {
    return;
  }

  const requestEpoch = stateRequestEpoch;
  try {
    aiAutoRunPaused = false;
    if (actionView.type === "attack" || actionView.type === "mulligan") {
      resetSelections();
    }
    const payload = await requestJson("/api/action", {
      method: "POST",
      body: JSON.stringify({
        session_id: currentState.session_id,
        action: actionView.action,
      }),
    });
    if (requestEpoch !== stateRequestEpoch) {
      return;
    }
    currentState = payload;
    sanitizeSelections(currentState);
    render(currentState);
    maybeRunAiTurn(currentState);
  } catch (error) {
    if (requestEpoch !== stateRequestEpoch) {
      return;
    }
    updateStatus(error.message);
  }
}

async function runAiTurn() {
  if (!currentState || aiIsRunning) {
    return;
  }

  const requestEpoch = stateRequestEpoch;
  aiAutoRunPaused = false;
  aiAutoRunQueued = false;
  aiIsRunning = true;
  render(currentState);
  updateStatus("AI is thinking...");
  try {
    if (!usesInstantStandardAiReplay(currentState)) {
      await sleep(randomDelay(AI_HUMAN_DELAY_MIN_MS, AI_HUMAN_DELAY_MAX_MS));
    }
    if (requestEpoch !== stateRequestEpoch) {
      return;
    }
    while (currentState && currentState.current_player === 1 && currentState.winner === null) {
      const payload = await requestJson("/api/ai-step", {
        method: "POST",
        body: JSON.stringify({ session_id: currentState.session_id }),
      });
      if (requestEpoch !== stateRequestEpoch) {
        return;
      }
      const step = payload.ai_step;
      currentState = payload;
      sanitizeSelections(currentState);
      render(currentState);
      await waitForPaint();

      if (currentState.current_player !== 1 || currentState.winner !== null) {
        break;
      }

      if (!step?.action) {
        if (currentState.current_player === 1 && currentState.winner === null) {
          await runAiTurnReplayFallback();
        }
        break;
      }

      updateStatus(buildAiReplayStatus(step));
      await waitForPaint();

      if ((step.delay_ms ?? FALLBACK_AI_STEP_DELAY_MS) > 0) {
        await sleep(step.delay_ms || FALLBACK_AI_STEP_DELAY_MS);
      }
      if (requestEpoch !== stateRequestEpoch) {
        return;
      }
    }
  } catch (error) {
    if (requestEpoch !== stateRequestEpoch) {
      return;
    }
    aiAutoRunPaused = true;
    updateStatus(`AI auto-run stopped: ${error.message} Refresh or restart the backend server, then try again.`);
  } finally {
    aiIsRunning = false;
    if (requestEpoch === stateRequestEpoch && currentState) {
      render(currentState);
    }
  }
}

async function runAiTurnReplayFallback() {
  if (!currentState) {
    return;
  }

  const requestEpoch = stateRequestEpoch;
  const payload = await requestJson("/api/ai-turn", {
    method: "POST",
    body: JSON.stringify({ session_id: currentState.session_id }),
  });
  if (requestEpoch !== stateRequestEpoch) {
    return;
  }

  const replaySteps = payload.ai_turn_replay?.steps || [];
  for (const replayStep of replaySteps) {
    currentState = replayStep.state;
    sanitizeSelections(currentState);
    render(currentState);
    updateStatus(buildAiReplayStatus(replayStep));
    await waitForPaint();

    if (currentState.current_player !== 1 || currentState.winner !== null) {
      break;
    }

    if ((replayStep.delay_ms ?? FALLBACK_AI_STEP_DELAY_MS) > 0) {
      await sleep(replayStep.delay_ms || FALLBACK_AI_STEP_DELAY_MS);
    }
    if (requestEpoch !== stateRequestEpoch) {
      return;
    }
  }

  currentState = payload;
  sanitizeSelections(currentState);
}

function buildAiReplayStatus(step) {
  const label = step.action?.label || "AI takes an action";
  return `AI action: ${label}`;
}

function resolveTrainerSelection(state) {
  return (
    uiState.selectedTrainerId ||
    state?.ai_trainer?.id ||
    state?.available_trainers?.[0]?.id ||
    ""
  );
}

function resolveHumanDeckSelection(state) {
  return (
    uiState.selectedHumanDeckId ||
    state?.human_deck_id ||
    state?.available_decks?.find((deck) => deck.selected)?.id ||
    state?.available_decks?.[0]?.id ||
    ""
  );
}

function resolveGameMode(state) {
  return (
    uiState.selectedGameModeId ||
    state?.game_mode ||
    state?.available_game_modes?.find((mode) => mode.selected)?.id ||
    "my_first_battle"
  );
}

function resolveStandardAiMode(state) {
  return (
    uiState.selectedStandardAiMode ||
    state?.standard_ai_mode ||
    "local"
  );
}

function usesInstantStandardAiReplay(state) {
  return resolveGameMode(state) === "standard" && resolveStandardAiMode(state) === "remote";
}

function resolveGameModeMetadata(state) {
  const gameMode = resolveGameMode(state);
  return (
    state?.available_game_modes?.find((mode) => mode.id === gameMode) ||
    state?.available_game_modes?.find((mode) => mode.selected) ||
    null
  );
}

function usesSharedEnergyPool(state) {
  return resolveGameMode(state) === "my_first_battle";
}

function renderAppIdentity(state) {
  const appTitle = document.getElementById("app-title");
  if (!appTitle) {
    return;
  }

  const modes = state.available_game_modes || [];
  const selectedModeId = resolveGameMode(state);
  appTitle.innerHTML = modes
    .map((mode) => `
      <button
        type="button"
        class="game-mode-pill${mode.id === selectedModeId ? " is-selected" : ""}${mode.available ? "" : " is-unavailable"}"
        data-game-mode="${escapeHtml(mode.id)}"
        aria-disabled="${mode.available ? "false" : "true"}"
      >
        ${escapeHtml(mode.name)}
      </button>
    `)
    .join("");

  for (const button of appTitle.querySelectorAll("[data-game-mode]")) {
    button.addEventListener("click", () => handleGameModeChange(button.dataset.gameMode || null));
  }
}

function buildTrainerLabel(trainer) {
  return `${trainer.name} • Lv. ${trainer.level}`;
}

function buildDeckLabel(deck) {
  return deck.name;
}

function getTrainerLevelProgress(trainer) {
  const xpIntoLevel = Math.max(0, Number(trainer?.xp_into_level || 0));
  const xpToNextLevel = Math.max(0, Number(trainer?.xp_to_next_level || 0));
  const totalLevelXp = Math.max(1, xpIntoLevel + xpToNextLevel);
  const progressPercent = Math.max(0, Math.min(100, (xpIntoLevel / totalLevelXp) * 100));
  return {
    xpIntoLevel,
    xpToNextLevel,
    totalLevelXp,
    progressPercent,
    nextLevel: Math.max(1, Number(trainer?.level || 1)) + 1,
  };
}

function handleTrainerChange(event) {
  uiState.selectedTrainerId = event.target.value || null;
  if (currentState) {
    render(currentState);
  } else if (lobbyState) {
    renderLobby(lobbyState);
  }
}

function handleDeckChange(event) {
  uiState.selectedHumanDeckId = event.target.value || null;
  if (currentState) {
    render(currentState);
  } else if (lobbyState) {
    renderLobby(lobbyState);
  }
}

async function handleStandardAiModeToggle(event) {
  const toggle = event.target;
  const selectionState = currentState || lobbyState;
  const gameMode = resolveGameMode(selectionState);
  const previousMode = resolveStandardAiMode(selectionState);
  const requestedMode = toggle.checked ? "remote" : "local";
  let finalStatusMessage = null;
  if (gameMode !== "standard") {
    toggle.checked = false;
    renderActiveView();
    return;
  }
  if (requestedMode === previousMode && currentState?.standard_ai_mode === requestedMode) {
    renderActiveView();
    return;
  }

  uiState.standardAiModePending = true;
  renderActiveView();
  try {
    if (requestedMode === "remote") {
      const status = await requestJson("/api/standard-ml-status");
      uiState.standardMlStatus = status;
      if (!(status.configured && status.ready && status.model_loaded)) {
        throw new Error(status.error || "Remote Standard NN mode is unavailable.");
      }
    } else {
      uiState.standardMlStatus = null;
    }

    const hasActiveGame = !!currentState && currentState.winner === null;
    if (
      hasActiveGame &&
      !window.confirm("Switching NN mode will reset the current game and start a new one. Continue?")
    ) {
      finalStatusMessage = "NN mode unchanged.";
      return;
    }

    uiState.selectedStandardAiMode = requestedMode;
    const started = await newGame({ standard_ai_mode: requestedMode });
    if (!started) {
      uiState.selectedStandardAiMode = previousMode;
      renderActiveView();
    }
  } catch (error) {
    uiState.selectedStandardAiMode = previousMode;
    finalStatusMessage = error.message;
  } finally {
    uiState.standardAiModePending = false;
    renderActiveView();
    if (finalStatusMessage) {
      updateStatus(finalStatusMessage);
    }
  }
}

function handleGameModeChange(gameModeId) {
  const nextGameModeId = gameModeId || null;
  const activeState = currentState || lobbyState;
  const currentGameModeId = resolveGameMode(activeState);
  if (nextGameModeId === currentGameModeId && !currentState) {
    return;
  }

  resetToLobbyState(nextGameModeId);
  void loadLobby(uiState.selectedGameModeId);
}

function maybeRunAiTurn(state) {
  if (state.winner !== null) {
    return;
  }
  if (state.current_player === 1) {
    queueAiTurn();
  }
}

function queueAiTurn() {
  if (
    aiAutoRunQueued ||
    aiAutoRunPaused ||
    aiIsRunning ||
    !currentState ||
    currentState.current_player !== 1 ||
    currentState.winner !== null
  ) {
    return;
  }

  aiAutoRunQueued = true;
  window.setTimeout(() => {
    aiAutoRunQueued = false;
    if (!currentState || currentState.current_player !== 1 || currentState.winner !== null || aiIsRunning) {
      return;
    }
    void runAiTurn();
  }, 0);
}

function resetSelections() {
  uiState.selectedCardId = null;
  uiState.selectedDiscardIds = [];
  uiState.discardBrowseRequest = null;
  uiState.discardBrowseSelectedIds = [];
  uiState.selectedBoardTarget = null;
  uiState.pendingAttackActionIds = [];
  uiState.attackOptionRequest = null;
  uiState.availableContextActions = [];
  uiState.deckBrowseRequest = null;
  uiState.deckBrowseSelectedIds = [];
}

function sanitizeSelections(state) {
  const sanitized = sanitizeSelectionState(uiState, state);
  uiState.selectedCardId = sanitized.selectedCardId;
  uiState.selectedBoardTarget = sanitized.selectedBoardTarget;
  const discardRequirement = resolveDiscardFromHandRequirement(state, uiState.selectedCardId);
  const handIds = new Set((state.players?.[0]?.hand || []).map((card) => card.instance_id));
  const allowedDiscardIds = new Set(
    (discardRequirement?.candidateIds || []).filter((instanceId) => handIds.has(instanceId)),
  );
  uiState.selectedDiscardIds = (uiState.selectedDiscardIds || []).filter((instanceId) =>
    allowedDiscardIds.has(instanceId),
  );
  if (
    discardRequirement &&
    uiState.selectedDiscardIds.length > discardRequirement.chooseCount
  ) {
    uiState.selectedDiscardIds = uiState.selectedDiscardIds.slice(0, discardRequirement.chooseCount);
  }
  const discardBrowseRequest = resolveDiscardBrowseRequest(state, uiState.selectedCardId);
  if (!discardBrowseRequest) {
    uiState.discardBrowseRequest = null;
    uiState.discardBrowseSelectedIds = [];
  } else {
    const discardBrowseSelectableIds = new Set(discardBrowseRequest.selectableHandCardIds || []);
    uiState.discardBrowseSelectedIds = (uiState.discardBrowseSelectedIds || []).filter((instanceId) =>
      discardBrowseSelectableIds.has(instanceId),
    );
    if (uiState.discardBrowseSelectedIds.length > discardBrowseRequest.chooseCount) {
      uiState.discardBrowseSelectedIds = uiState.discardBrowseSelectedIds.slice(0, discardBrowseRequest.chooseCount);
    }
  }
  uiState.pendingAttackActionIds = (uiState.pendingAttackActionIds || []).filter((actionId) =>
    state.legal_actions.some(
      (actionView) =>
        actionView.action_id === actionId &&
        refsMatch(actionView.source, uiState.selectedBoardTarget),
    ),
  );
  if (!uiState.attackOptionRequest) {
    // no-op
  } else {
    const actionViews = state.legal_actions.filter((actionView) =>
      uiState.attackOptionRequest.actionIds.includes(actionView.action_id),
    );
    if (!actionViews.length) {
      uiState.attackOptionRequest = null;
    } else {
      uiState.attackOptionRequest = {
        ...uiState.attackOptionRequest,
        actionIds: actionViews.map((actionView) => actionView.action_id),
      };
    }
  }
  const deckBrowseRequest = resolveDeckBrowseRequest(state);
  if (!deckBrowseRequest) {
    uiState.deckBrowseRequest = null;
    uiState.deckBrowseSelectedIds = [];
    return;
  }
  uiState.deckBrowseRequest = deckBrowseRequest;
  const deckBrowseSelectableIds = new Set(deckBrowseRequest.selectableDeckCardIds || []);
  uiState.deckBrowseSelectedIds = (uiState.deckBrowseSelectedIds || []).filter((instanceId) =>
    deckBrowseSelectableIds.has(instanceId),
  );
  const chooseCount = deckBrowseRequest.chooseCount || 1;
  if (uiState.deckBrowseSelectedIds.length > chooseCount) {
    uiState.deckBrowseSelectedIds = uiState.deckBrowseSelectedIds.slice(0, chooseCount);
  }
}

function render(state) {
  const previousSnapshot = previousState;
  const human = state.players[0];
  const ai = state.players[1];
  const context = deriveContext(state);
  const mulliganAction = findOpeningMulliganAction(state);
  const faceDownCardImageUrl = resolveFaceDownCardImageUrl(state);
  const sharedEnergyEnabled = usesSharedEnergyPool(state);
  const playerActiveTargetRef = human.active?.ref || {
    player_index: 0,
    zone: "active",
    bench_index: null,
    instance_id: null,
  };
  const opponentActiveTargetRef = ai.active?.ref || {
    player_index: 1,
    zone: "active",
    bench_index: null,
    instance_id: null,
  };
  const playerEnergyTargetRef = {
    player_index: 0,
    zone: "energy",
    bench_index: null,
    instance_id: null,
  };
  const opponentEnergyTargetRef = {
    player_index: 1,
    zone: "energy",
    bench_index: null,
    instance_id: null,
  };
  uiState.availableContextActions = context.actions;

  renderAppIdentity(state);
  renderMatchMeta(state);
  renderStandardAiModeToggle(state);
  renderTrainerPicker(state);
  renderDeckPicker(state);
  renderOpponentIdentity(state);
  renderTurnHighlights(state);
  renderPlayerEndTurnButton(state);
  const selectionSummaryElement = document.getElementById("selection-summary");
  if (selectionSummaryElement) {
    selectionSummaryElement.textContent = describeSelection(state, context);
  }
  setSharedEnergyVisibility(sharedEnergyEnabled);
  renderDeckPile(document.getElementById("player-deck"), human.deck_pile, {
    imageUrl: faceDownCardImageUrl,
  });
  renderDeckPile(document.getElementById("opponent-deck"), ai.deck_pile, {
    imageUrl: faceDownCardImageUrl,
  });
  renderSharedEnergySpot(document.getElementById("player-shared-energy"), human.energy_zone, {
    clickable: sharedEnergyEnabled && isBoardRefClickable({
      state,
      uiState,
      context,
      aiIsRunning,
      ref: playerEnergyTargetRef,
    }),
    selectedTarget: uiState.selectedBoardTarget,
    context,
    targetRef: playerEnergyTargetRef,
  });
  renderSharedEnergySpot(document.getElementById("opponent-shared-energy"), ai.energy_zone, {
    clickable: false,
    selectedTarget: null,
    context,
    targetRef: opponentEnergyTargetRef,
  });
  renderPrizePile(document.getElementById("player-prizes"), human.prize_pile, {
    imageUrl: faceDownCardImageUrl,
    compactPile: state.game_mode === "standard",
  });
  renderPrizePile(document.getElementById("opponent-prizes"), ai.prize_pile, {
    imageUrl: faceDownCardImageUrl,
    compactPile: state.game_mode === "standard",
  });
  renderDiscard(document.getElementById("player-discard"), human.discard_top, human.discard_count);
  renderDiscard(document.getElementById("opponent-discard"), ai.discard_top, ai.discard_count);

  renderPokemonZone(
    document.getElementById("player-active"),
    human.active,
    {
      clickable: isBoardRefClickable({
        state,
        uiState,
        context,
        aiIsRunning,
        ref: playerActiveTargetRef,
      }),
      selectedTarget: uiState.selectedBoardTarget,
      context,
      targetRef: playerActiveTargetRef,
      previousPokemon: previousSnapshot?.players?.[0]?.active || null,
      setupPromptAction: mulliganAction,
    },
  );
  renderPokemonZone(
    document.getElementById("opponent-active"),
    ai.active,
    {
      clickable: isBoardRefClickable({
        state,
        uiState,
        context,
        aiIsRunning,
        ref: opponentActiveTargetRef,
      }),
      selectedTarget: uiState.selectedBoardTarget,
      context,
      targetRef: opponentActiveTargetRef,
      previousPokemon: previousSnapshot?.players?.[1]?.active || null,
    },
  );

  renderPokemonList(
    document.getElementById("player-bench"),
    human.bench,
    {
      isRefClickable: (ref) => isBoardRefClickable({
        state,
        uiState,
        context,
        aiIsRunning,
        ref,
      }),
      selectedTarget: uiState.selectedBoardTarget,
      context,
      previousBench: previousSnapshot?.players?.[0]?.bench || [],
    },
  );
  syncBenchZoneState(document.getElementById("player-bench-zone"), state);
  renderPokemonList(
    document.getElementById("opponent-bench"),
    ai.bench,
    {
      isRefClickable: (ref) => isBoardRefClickable({
        state,
        uiState,
        context,
        aiIsRunning,
        ref,
      }),
      selectedTarget: uiState.selectedBoardTarget,
      context,
      previousBench: previousSnapshot?.players?.[1]?.bench || [],
    },
  );

  renderHand(document.getElementById("player-hand"), human.hand, context, previousSnapshot?.players?.[0]?.hand || []);
  renderSelectedCardPreview(state);
  renderDeckBrowserOverlay(state);
  renderAttackDragIndicator(state);
  const contextActionsElement = document.getElementById("context-actions");
  if (contextActionsElement) {
    renderContextActions(contextActionsElement, context.actions);
  }
  renderDebugActions(document.getElementById("debug-actions"), state.legal_actions);
  const battleLogElement = document.getElementById("battle-log");
  const previousLogEntries = previousSnapshot?.log || [];
  if (didLogEntriesChange(state.log, previousLogEntries) || battleLogElement.childElementCount === 0) {
    renderLog(battleLogElement, state.log, previousLogEntries);
  }

  const contextInstructionsElement = document.getElementById("context-instructions");
  if (contextInstructionsElement) {
    contextInstructionsElement.textContent = context.instructions;
  }
  document.getElementById("new-game-button").disabled = aiIsRunning;

  updateStatus(makeStatusMessage(state, context));
  previousState = JSON.parse(JSON.stringify(state));
  syncAdaptiveActiveCardLayout();
  syncCursorForPointerPosition();
  if (state.current_player === 1 && state.winner === null) {
    queueAiTurn();
  }
}

function renderLobby(state) {
  renderAppIdentity(state);
  renderLobbyMatchMeta();
  renderStandardAiModeToggle(state);
  renderTrainerPicker(state);
  renderDeckPicker(state);
  renderOpponentIdentity(state);
  renderLobbyBoard(state);
  const selectedMode = resolveGameModeMetadata(state);
  const newGameButton = document.getElementById("new-game-button");
  if (newGameButton) {
    newGameButton.disabled = aiIsRunning || !selectedMode?.available;
  }
  syncAdaptiveActiveCardLayout();
  syncCursorForPointerPosition();
  updateStatus(
    selectedMode?.available
      ? "Choose your deck and gym leader, then start a new game."
      : "Standard ex Battle Deck data is loaded for preview, but Standard battles are not wired up yet.",
  );
}

function resolveFaceDownCardImageUrl(state) {
  return state?.shared_assets?.face_down_card_image_url || FACE_DOWN_CARD_IMAGE_URL;
}

function setSharedEnergyVisibility(enabled) {
  for (const pocketId of ["player-shared-energy-pocket", "opponent-shared-energy-pocket"]) {
    const pocket = document.getElementById(pocketId);
    if (pocket) {
      pocket.hidden = !enabled;
    }
  }
}

function renderDeckPile(element, deckPile, options) {
  if (!element) {
    return;
  }
  element.innerHTML = "";
  const count = Number(deckPile?.count ?? 0);
  if (count <= 0) {
    element.appendChild(
      buildPileEmptyCard({
        kind: "deck",
        label: "Deck",
      }),
    );
    return;
  }
  const deckCard = buildFaceDownPileCard({
    imageUrl: options.imageUrl,
    title: "Deck pile",
    description: `Deck pile • ${count} card${count === 1 ? "" : "s"} remaining`,
    badgeText: count,
    kind: "deck",
    stacked: count > 1,
  });
  element.appendChild(deckCard);
}

function renderSharedEnergySpot(element, energyCards, options) {
  if (!element) {
    return;
  }

  element.innerHTML = "";
  const cards = energyCards || [];
  const energySpot = buildSharedEnergyHolderCard({
    energyCard: cards[cards.length - 1] || null,
    count: cards.length,
    clickable: options.clickable,
    selected: !!options.selectedTarget && refsMatch(options.selectedTarget, options.targetRef),
    targetable: options.context.highlightedTargets.some((target) =>
      refsMatch(target, options.targetRef),
    ),
  });
  if (options.clickable) {
    energySpot.addEventListener("click", () => toggleSelectedBoardTarget(options.targetRef));
  }
  element.appendChild(energySpot);
}

function renderPrizePile(element, prizePile, options) {
  if (!element) {
    return;
  }
  element.innerHTML = "";
  const count = Number(prizePile?.count ?? 0);
  const imageUrl = prizePile?.image_url || options.imageUrl;
  const usesCoinArtwork = !!prizePile?.image_url;
  element.classList.toggle("is-empty", count === 0);
  if (count <= 0) {
    element.appendChild(buildPrizeEmptyStamp());
    return;
  }

  if (options.compactPile) {
    const prizeCard = buildFaceDownPileCard({
      imageUrl,
      title: usesCoinArtwork ? "Prize coin pile" : "Prize card pile",
      description: usesCoinArtwork
        ? `Prize pile • ${count} coin${count === 1 ? "" : "s"} remaining`
        : `Prize pile • ${count} card${count === 1 ? "" : "s"} remaining`,
      kind: "prize",
      badgeText: count,
      stacked: count > 1,
    });
    prizeCard.classList.add("prize-pile-card");
    prizeCard.classList.toggle("prize-pile-card-coin", usesCoinArtwork);
    prizeCard.setAttribute("aria-hidden", "true");
    element.appendChild(prizeCard);
    return;
  }

  for (let index = 0; index < count; index += 1) {
    const prizeCard = buildFaceDownPileCard({
      imageUrl,
      title: usesCoinArtwork ? `Prize coin ${index + 1}` : `Prize card ${index + 1}`,
      description: usesCoinArtwork
        ? `Prize coin ${index + 1} of ${count}`
        : `Prize card ${index + 1} of ${count}`,
      kind: "prize",
    });
    prizeCard.classList.add("prize-card");
    prizeCard.classList.toggle("prize-card-coin", usesCoinArtwork);
    prizeCard.setAttribute("aria-hidden", "true");
    element.appendChild(prizeCard);
  }
}

function renderDiscard(element, discardTop, discardCount) {
  element.innerHTML = "";
  if (!discardTop || discardCount <= 0) {
    element.appendChild(
      buildPileEmptyCard({
        kind: "discard",
        label: "Discard",
      }),
    );
    return;
  }

  const card = buildMiniCard(discardTop, {
    hideAccent: true,
    hideCopy: true,
  });
  card.classList.add("pile-card", "discard-pile-card", "is-inert");
  card.title = `${discardTop.name} • ${discardCount} card${discardCount === 1 ? "" : "s"}`;
  card.setAttribute("aria-hidden", "true");
  if (discardCount > 1) {
    card.classList.add("is-stacked");
  }
  card.appendChild(buildPileCountBadge(discardCount));
  element.appendChild(card);
}

function renderPokemonZone(element, pokemon, options) {
  element.innerHTML = "";
  element.appendChild(buildPokemonCard(pokemon, options));
}

function renderPokemonList(element, pokemonList, options) {
  element.innerHTML = "";
  element.classList.toggle("is-empty", !pokemonList.length);
  if (!pokemonList.length) {
    element.appendChild(buildBenchEmptyStamp());
    return;
  }

  pokemonList.forEach((pokemon, index) => {
    const previousPokemon = options.previousBench[index] || null;
    element.appendChild(
      buildPokemonCard(pokemon, {
        ...options,
        clickable: options.isRefClickable ? options.isRefClickable(pokemon.ref) : options.clickable,
        compact: true,
        benchCard: true,
        previousPokemon,
      }),
    );
  });
}

function buildBenchEmptyStamp() {
  return buildEmptyStamp("Bench", "bench-empty-stamp");
}

function buildEmptyStamp(label, className = "bench-empty-stamp") {
  return buildStackedStamp([label, "Empty"], className);
}

function buildStackedStamp(lines, className) {
  const element = document.createElement("div");
  element.className = className;
  for (const line of lines) {
    const lineElement = document.createElement("span");
    lineElement.textContent = line;
    element.appendChild(lineElement);
  }
  return element;
}

function buildPrizeEmptyStamp() {
  const element = document.createElement("div");
  element.className = "prize-empty-stamp";
  element.textContent = "Cleared";
  return element;
}

function buildSharedEnergyHolderCard(options = {}) {
  const element = document.createElement("article");
  const classNames = ["mini-card", "pile-card", "shared-energy-spot"];
  if (options.count > 0) {
    classNames.push("is-filled");
  }
  if (options.clickable) {
    classNames.push("is-clickable");
  }
  if (options.selected) {
    classNames.push("is-selected");
  }
  if (options.targetable) {
    classNames.push("is-targetable");
  }
  element.className = classNames.join(" ");
  element.dataset.kind = "shared-energy";
  element.title = `Shared Energy • ${options.count} in pool`;
  element.setAttribute("aria-label", `Shared Energy ${options.count}`);
  const mediaMarkup = options.energyCard
    ? `
      <div class="shared-energy-spot__card">
        ${buildCardImageMarkup(options.energyCard.image_url, options.energyCard.name)}
      </div>
    `
    : "";
  element.innerHTML = `<div class="shared-energy-spot__media">${mediaMarkup}</div>`;
  if (!options.energyCard) {
    const well = document.createElement("div");
    well.className = "shared-energy-spot__well";
    well.appendChild(buildStackedStamp(["Shared", "Energy", "Pool"], "shared-energy-stamp"));
    element.querySelector(".shared-energy-spot__media")?.appendChild(well);
  }
  if (options.count > 0) {
    const badge = buildPileCountBadge(options.count);
    badge.classList.add("shared-energy-count-badge");
    element.appendChild(badge);
  }
  return element;
}

function renderHand(element, hand, context, previousHand) {
  element.innerHTML = "";
  if (!hand.length) {
    element.appendChild(buildPlaceholder("Your hand is empty."));
    return;
  }

  const previousIds = new Set(previousHand.map((card) => card.instance_id));
  for (const card of hand) {
    const miniCard = buildMiniCard(card, {
      clickable: true,
      playable: card.playable,
      hideCopy: true,
      selected:
        uiState.selectedCardId === card.instance_id ||
        uiState.selectedDiscardIds.includes(card.instance_id),
      targetable: context.highlightedHandIds.has(card.instance_id),
      animationClass: previousIds.has(card.instance_id) ? "" : "card-anim-draw",
    });
    miniCard.addEventListener("click", () => toggleSelectedCard(card.instance_id));
    element.appendChild(miniCard);
  }
}

function renderSelectedCardPreview(state) {
  const previewElement = document.getElementById("floating-card-preview");
  if (!previewElement) {
    return;
  }

  const preview = resolveSelectedPreview(state);
  if (!preview) {
    previewElement.hidden = true;
    previewElement.innerHTML = "";
    return;
  }

  previewElement.hidden = false;
  const card = document.createElement("article");
  card.className = "floating-card-preview__card";
  card.dataset.element = preview.element || "";
  card.dataset.kind = preview.kind || "";
  card.title = preview.name;
  card.setAttribute("aria-label", `${preview.name} enlarged preview`);
  card.innerHTML = buildCardImageMarkup(preview.imageUrl, preview.name);
  previewElement.innerHTML = "";
  previewElement.appendChild(card);
}

function boardRefKey(ref) {
  if (!ref) {
    return "";
  }
  return [
    String(ref.player_index ?? "x"),
    String(ref.zone || "unknown"),
    Number.isInteger(ref.bench_index) ? String(ref.bench_index) : "x",
    String(ref.instance_id || "x"),
  ].join(":");
}

function findPokemonForRef(state, ref) {
  if (!state || !ref) {
    return null;
  }

  const player = state.players?.[ref.player_index];
  if (!player) {
    return null;
  }
  if (ref.zone === "active") {
    return player.active || null;
  }
  if (ref.zone === "bench" && Number.isInteger(ref.bench_index)) {
    return player.bench?.[ref.bench_index] || null;
  }
  return null;
}

function buildDeckBrowseRequest({
  sourceKey,
  sourceKind = "hand",
  sourceCardId = null,
  sourceRef = null,
  attackIndex = null,
  sourceCardName,
  searchEffect,
  searchActions,
  selectionField = "search_deck_ids",
}) {
  if (!sourceKey || !sourceCardName || !searchEffect || !searchActions.length) {
    return null;
  }

  const actionByDeckCardId = new Map();
  const selectableDeckCardIds = new Set();
  for (const actionView of searchActions) {
    const selectedDeckIds = Array.isArray(actionView.action?.[selectionField])
      ? actionView.action[selectionField]
      : [];
    for (const selectedDeckId of selectedDeckIds) {
      selectableDeckCardIds.add(selectedDeckId);
      if (!actionByDeckCardId.has(selectedDeckId)) {
        actionByDeckCardId.set(selectedDeckId, actionView);
      }
    }
  }

  const visibleCount = Number.isInteger(searchEffect.count) && searchEffect.count > 0
    ? searchEffect.count
    : null;
  const maximumSelectableCount = searchActions.reduce(
    (maxCount, actionView) =>
      Math.max(maxCount, Array.isArray(actionView.action?.[selectionField]) ? actionView.action[selectionField].length : 0),
    0,
  );
  const configuredChooseCount = Number.isInteger(searchEffect.choose_count) ? searchEffect.choose_count : 1;
  const chooseCount = maximumSelectableCount > 0 ? maximumSelectableCount : configuredChooseCount;
  const actionBySelectionKey = buildActionSelectionMap(searchActions, selectionField);
  const minimumChooseCount = actionBySelectionKey.has(selectionKeyForDeckIds([])) ? 0 : chooseCount;

  return {
    requestKey: sourceKey,
    sourceKind,
    sourceCardId,
    sourceRef: sourceRef ? { ...sourceRef } : null,
    attackIndex,
    sourceCardName,
    sourceZone: searchEffect.source_zone || "deck",
    selectionField,
    scope: visibleCount ? "top_cards" : "full_deck",
    visibleCount,
    chooseCount,
    minimumChooseCount,
    destinationZone: searchEffect.destination_zone || "hand",
    searchFilters: Array.isArray(searchEffect.search_filters) ? searchEffect.search_filters : [],
    actionByDeckCardId,
    actionBySelectionKey,
    selectableDeckCardIds: [...selectableDeckCardIds],
  };
}

function resolveHandDeckBrowseRequest(state, sourceCardId = uiState.selectedCardId) {
  const player = state?.players?.[0];
  if (!player || !sourceCardId) {
    return null;
  }

  const handCard = player.hand?.find((card) => card.instance_id === sourceCardId);
  if (!handCard) {
    return null;
  }

  const searchEffect = (handCard.effect_specs || []).find(
    (effectSpec) =>
      effectSpec.effect_type === "search_deck" &&
      effectSpec.source_zone === "deck" &&
      effectSpec.destination_zone,
  );
  if (!searchEffect) {
    return null;
  }
  const discardRequirement = resolveDiscardFromHandRequirement(state, sourceCardId);
  if (
    discardRequirement &&
    uiState.selectedDiscardIds.length < discardRequirement.chooseCount
  ) {
    return null;
  }
  const searchActions = findSearchDeckActionsForSelection(state, sourceCardId);
  if (!searchActions.length) {
    return null;
  }
  return buildDeckBrowseRequest({
    sourceKey: `hand:${sourceCardId}`,
    sourceKind: "hand",
    sourceCardId,
    sourceCardName: handCard.name,
    searchEffect,
    searchActions,
  });
}

function findRecoverFromDiscardActionsForSelection(state, sourceCardId = uiState.selectedCardId) {
  if (!state || !sourceCardId) {
    return [];
  }
  return state.legal_actions.filter(
    (actionView) =>
      actionView.source?.zone === "hand" &&
      actionView.source.instance_id === sourceCardId &&
      !actionView.target &&
      Array.isArray(actionView.action?.recover_from_discard_ids) &&
      actionMatchesSelectedDiscardIds(actionView),
  );
}

function resolveHandDiscardRecoveryRequest(state, sourceCardId = uiState.selectedCardId) {
  const player = state?.players?.[0];
  if (!player || !sourceCardId) {
    return null;
  }

  const handCard = player.hand?.find((card) => card.instance_id === sourceCardId);
  if (!handCard) {
    return null;
  }

  const recoverEffect = (handCard.effect_specs || []).find(
    (effectSpec) =>
      effectSpec.effect_type === "recover_from_discard" &&
      effectSpec.source_zone === "discard" &&
      effectSpec.destination_zone === "hand",
  );
  if (!recoverEffect) {
    return null;
  }

  const recoveryActions = findRecoverFromDiscardActionsForSelection(state, sourceCardId);
  if (!recoveryActions.length) {
    return null;
  }

  return buildDeckBrowseRequest({
    sourceKey: `discard:${sourceCardId}`,
    sourceKind: "hand",
    sourceCardId,
    sourceCardName: handCard.name,
    searchEffect: recoverEffect,
    searchActions: recoveryActions,
    selectionField: "recover_from_discard_ids",
  });
}

function findAttackSearchActions(state, sourceRef, attackIndex = null) {
  if (!state || !sourceRef) {
    return [];
  }

  return state.legal_actions.filter(
    (actionView) =>
      actionView.type === "attack" &&
      refsMatch(actionView.source, sourceRef) &&
      (!Number.isInteger(attackIndex) || actionView.action?.attack_index === attackIndex) &&
      Array.isArray(actionView.action?.search_deck_ids),
  );
}

function resolveAttackDeckBrowseRequest(
  state,
  {
    sourceRef = uiState.selectedBoardTarget,
    attackIndex = null,
  } = {},
) {
  const pokemon = findPokemonForRef(state, sourceRef);
  if (!pokemon) {
    return null;
  }

  const searchActions = findAttackSearchActions(state, sourceRef, attackIndex);
  if (!searchActions.length) {
    return null;
  }

  const resolvedAttackIndex = Number.isInteger(attackIndex)
    ? attackIndex
    : searchActions[0].action?.attack_index;
  const attack = Number.isInteger(resolvedAttackIndex) ? pokemon.attacks?.[resolvedAttackIndex] : null;
  if (!attack) {
    return null;
  }

  const searchEffect = (attack.effect_specs || []).find(
    (effectSpec) =>
      effectSpec.effect_type === "search_deck" &&
      effectSpec.source_zone === "deck" &&
      effectSpec.destination_zone,
  );
  if (!searchEffect) {
    return null;
  }

  return buildDeckBrowseRequest({
    sourceKey: `attack:${boardRefKey(sourceRef)}:${String(resolvedAttackIndex)}`,
    sourceKind: "attack",
    sourceRef,
    attackIndex: resolvedAttackIndex,
    sourceCardName: attack.name,
    searchEffect,
    searchActions,
  });
}

function resolveDeckBrowseRequest(state) {
  if (uiState.selectedCardId) {
    return (
      resolveHandDiscardRecoveryRequest(state, uiState.selectedCardId) ||
      resolveHandDeckBrowseRequest(state, uiState.selectedCardId)
    );
  }
  const activeRequest = uiState.deckBrowseRequest;
  if (activeRequest?.sourceKind === "attack" && activeRequest.sourceRef) {
    return resolveAttackDeckBrowseRequest(state, {
      sourceRef: activeRequest.sourceRef,
      attackIndex: activeRequest.attackIndex,
    });
  }
  if (uiState.selectedBoardTarget) {
    return resolveAttackDeckBrowseRequest(state, { sourceRef: uiState.selectedBoardTarget });
  }
  return null;
}

function resolveDiscardBrowseRequest(state, sourceCardId = uiState.selectedCardId) {
  const player = state?.players?.[0];
  if (!player || !sourceCardId) {
    return null;
  }

  const sourceCard = findHandCardById(state, sourceCardId);
  const discardRequirement = resolveDiscardFromHandRequirement(state, sourceCardId);
  if (!sourceCard || !discardRequirement) {
    return null;
  }

  const selectableHandCards = (player.hand || []).filter((card) =>
    discardRequirement.candidateIds.includes(card.instance_id),
  );
  if (selectableHandCards.length < discardRequirement.chooseCount) {
    return null;
  }

  return {
    sourceCardId,
    sourceCardName: sourceCard.name,
    chooseCount: discardRequirement.chooseCount,
    selectableHandCards,
    selectableHandCardIds: selectableHandCards.map((card) => card.instance_id),
  };
}

function buildActionSelectionMap(actionViews, selectionField = "search_deck_ids") {
  const actionMap = new Map();
  for (const actionView of actionViews) {
    const selectedIds = actionView.action?.[selectionField] || [];
    const key = selectionKeyForDeckIds(selectedIds);
    if (!key || actionMap.has(key)) {
      continue;
    }
    actionMap.set(key, actionView);
  }
  return actionMap;
}

function selectionKeyForDeckIds(deckIds) {
  if (!Array.isArray(deckIds)) {
    return "";
  }
  if (!deckIds.length) {
    return "__empty__";
  }
  return [...deckIds].sort().join("|");
}

function openDeckBrowserForSelection(state) {
  const sourceCardId = uiState.selectedCardId;
  const sourceCard = findHandCardById(state, sourceCardId);
  const request = resolveDeckBrowseRequest(state);
  const discardRequirement = resolveDiscardFromHandRequirement(state, sourceCardId);
  if (sourceCard && discardRequirement && !request) {
    const discardRequest = resolveDiscardBrowseRequest(state, sourceCardId);
    if (!discardRequest) {
      return false;
    }
    uiState.discardBrowseRequest = discardRequest;
    uiState.discardBrowseSelectedIds = [];
    updateStatus(`Choose ${discardRequest.chooseCount} cards to discard for ${sourceCard.name}.`);
    render(currentState);
    return true;
  }
  if (!request) {
    return false;
  }
  uiState.discardBrowseRequest = null;
  uiState.discardBrowseSelectedIds = [];
  uiState.deckBrowseRequest = request;
  uiState.deckBrowseSelectedIds = [];
  render(currentState);
  return true;
}

function findHandCardById(state, instanceId) {
  if (!state || !instanceId) {
    return null;
  }
  return state.players?.[0]?.hand?.find((card) => card.instance_id === instanceId) || null;
}

function resolveDiscardFromHandRequirement(state, sourceCardId = uiState.selectedCardId) {
  const sourceCard = findHandCardById(state, sourceCardId);
  if (!sourceCard) {
    return null;
  }
  const discardEffect = (sourceCard.effect_specs || []).find(
    (effectSpec) =>
      effectSpec.effect_type === "discard_from_hand" &&
      effectSpec.source_zone === "hand" &&
      effectSpec.destination_zone === "discard" &&
      Number.isInteger(effectSpec.choose_count) &&
      effectSpec.choose_count > 0,
  );
  if (!discardEffect) {
    return null;
  }
  const handIds = (state?.players?.[0]?.hand || []).map((card) => card.instance_id);
  const excludeSourceCard = !!discardEffect.exclude_source_card;
  return {
    chooseCount: discardEffect.choose_count,
    candidateIds: handIds.filter((instanceId) => !excludeSourceCard || instanceId !== sourceCardId),
  };
}

function actionMatchesSelectedDiscardIds(actionView) {
  const selectedDiscardIds = uiState.selectedDiscardIds || [];
  const requiredDiscardIds = actionView?.action?.discard_from_hand_ids;
  if (!Array.isArray(requiredDiscardIds) || !requiredDiscardIds.length) {
    return !selectedDiscardIds.length;
  }
  if (requiredDiscardIds.length !== selectedDiscardIds.length) {
    return false;
  }
  const selectedSet = new Set(selectedDiscardIds);
  return requiredDiscardIds.every((instanceId) => selectedSet.has(instanceId));
}

function findSearchDeckActionsForSelection(state, sourceCardId = uiState.selectedCardId) {
  if (!state || !sourceCardId) {
    return [];
  }
  return state.legal_actions.filter(
    (actionView) =>
      actionView.source?.zone === "hand" &&
      actionView.source.instance_id === sourceCardId &&
      !actionView.target &&
      Array.isArray(actionView.action?.search_deck_ids) &&
      actionMatchesSelectedDiscardIds(actionView),
  );
}

function cardMatchesSearchFilters(card, searchFilters) {
  if (!Array.isArray(searchFilters) || !searchFilters.length) {
    return true;
  }

  return searchFilters.every((filter) => {
    if (filter === "pokemon") {
      return card.kind === "pokemon";
    }
    if (filter === "basic_pokemon") {
      return card.kind === "pokemon" && card.is_basic;
    }
    if (filter === "evolution_pokemon") {
      return card.kind === "pokemon" && !card.is_basic;
    }
    if (filter === "supporter") {
      return card.kind === "trainer" && Array.isArray(card.card_tags) && card.card_tags.includes("supporter");
    }
    if (filter === "basic_energy") {
      return !!card.is_basic_energy;
    }
    return true;
  });
}

function attachDeckBrowserDragBehavior(track) {
  if (!track) {
    return;
  }

  track.addEventListener("pointerdown", (event) => {
    if (event.button !== 0) {
      return;
    }
    if (
      event.target instanceof Element &&
      event.target.closest(".deck-browser-card.is-selectable, .mini-card.is-clickable")
    ) {
      return;
    }
    deckBrowserDragState.pointerId = event.pointerId;
    deckBrowserDragState.startClientX = event.clientX;
    deckBrowserDragState.startClientY = event.clientY;
    deckBrowserDragState.startScrollLeft = track.scrollLeft;
    deckBrowserDragState.isDragging = false;
    track.classList.add("is-dragging");
    track.setPointerCapture?.(event.pointerId);
    event.preventDefault();
  });

  track.addEventListener("pointermove", (event) => {
    if (deckBrowserDragState.pointerId !== event.pointerId) {
      return;
    }
    const deltaY = Math.abs(event.clientY - deckBrowserDragState.startClientY);
    const deltaX = event.clientX - deckBrowserDragState.startClientX;
    if (!deckBrowserDragState.isDragging) {
      deckBrowserDragState.isDragging = Math.abs(deltaX) > 4 || deltaY > 4;
      if (!deckBrowserDragState.isDragging) {
        return;
      }
    }
    track.scrollLeft = deckBrowserDragState.startScrollLeft - deltaX;
  });

  const stopDragging = (event) => {
    if (deckBrowserDragState.pointerId !== event.pointerId) {
      return;
    }
    deckBrowserDragState.pointerId = null;
    deckBrowserDragState.isDragging = false;
    track.classList.remove("is-dragging");
    track.releasePointerCapture?.(event.pointerId);
  };

  track.addEventListener("pointerup", stopDragging);
  track.addEventListener("pointercancel", stopDragging);
}

function buildDeckBrowserSelectionMetaText(activeRequest) {
  const minimumChooseCount = Number.isInteger(activeRequest.minimumChooseCount)
    ? activeRequest.minimumChooseCount
    : activeRequest.chooseCount;
  const actionText = minimumChooseCount === 0 && activeRequest.chooseCount > 0
    ? `Choose up to ${String(activeRequest.chooseCount)} card${activeRequest.chooseCount === 1 ? "" : "s"}`
    : `Choose ${String(activeRequest.chooseCount)} card${activeRequest.chooseCount === 1 ? "" : "s"}`;
  return `${actionText} to place into ${String((activeRequest.destinationZone || "hand").replaceAll("_", " "))}. Selected: ${String(uiState.deckBrowseSelectedIds.length)}/${String(activeRequest.chooseCount)}.`;
}

function buildDiscardBrowserSelectionMetaText(activeRequest) {
  return `Choose ${String(activeRequest.chooseCount)} card${activeRequest.chooseCount === 1 ? "" : "s"} to discard. Selected: ${String(uiState.discardBrowseSelectedIds.length)}/${String(activeRequest.chooseCount)}.`;
}

function syncDeckBrowserSelectionUi(overlay, activeRequest) {
  const metaElement = overlay.querySelector(".deck-browser__meta");
  if (metaElement) {
    metaElement.textContent = buildDeckBrowserSelectionMetaText(activeRequest);
  }

  const confirmButton = overlay.querySelector(".deck-browser__confirm");
  if (confirmButton) {
    const minimumChooseCount = Number.isInteger(activeRequest.minimumChooseCount)
      ? activeRequest.minimumChooseCount
      : activeRequest.chooseCount;
    const selectedCount = uiState.deckBrowseSelectedIds.length;
    confirmButton.disabled = selectedCount < minimumChooseCount || selectedCount > activeRequest.chooseCount;
  }

  for (const cardElement of overlay.querySelectorAll(".deck-browser-card")) {
    const instanceId = cardElement.dataset.instanceId || "";
    cardElement.classList.toggle("is-selected", uiState.deckBrowseSelectedIds.includes(instanceId));
    cardElement.setAttribute(
      "aria-pressed",
      uiState.deckBrowseSelectedIds.includes(instanceId) ? "true" : "false",
    );
  }
}

function updateDeckBrowserSelection(selectedDeckCardId, overlay, activeRequest) {
  if (!selectedDeckCardId) {
    return;
  }
  const currentSelection = new Set(uiState.deckBrowseSelectedIds || []);
  if (currentSelection.has(selectedDeckCardId)) {
    currentSelection.delete(selectedDeckCardId);
  } else if (currentSelection.size < activeRequest.chooseCount) {
    currentSelection.add(selectedDeckCardId);
  } else if (activeRequest.chooseCount === 1) {
    currentSelection.clear();
    currentSelection.add(selectedDeckCardId);
  }
  uiState.deckBrowseSelectedIds = [...currentSelection];
  syncDeckBrowserSelectionUi(overlay, activeRequest);
}

function syncDiscardBrowserSelectionUi(overlay, activeRequest) {
  const metaElement = overlay.querySelector(".deck-browser__meta");
  if (metaElement) {
    metaElement.textContent = buildDiscardBrowserSelectionMetaText(activeRequest);
  }

  const confirmButton = overlay.querySelector(".deck-browser__confirm");
  if (confirmButton) {
    confirmButton.disabled = uiState.discardBrowseSelectedIds.length !== activeRequest.chooseCount;
  }

  for (const cardElement of overlay.querySelectorAll(".deck-browser-card")) {
    const instanceId = cardElement.dataset.instanceId || "";
    cardElement.classList.toggle("is-selected", uiState.discardBrowseSelectedIds.includes(instanceId));
    cardElement.setAttribute(
      "aria-pressed",
      uiState.discardBrowseSelectedIds.includes(instanceId) ? "true" : "false",
    );
  }
}

function updateDiscardBrowserSelection(selectedHandCardId, overlay, activeRequest) {
  if (!selectedHandCardId) {
    return;
  }
  const currentSelection = new Set(uiState.discardBrowseSelectedIds || []);
  if (currentSelection.has(selectedHandCardId)) {
    currentSelection.delete(selectedHandCardId);
  } else if (currentSelection.size < activeRequest.chooseCount) {
    currentSelection.add(selectedHandCardId);
  } else if (activeRequest.chooseCount === 1) {
    currentSelection.clear();
    currentSelection.add(selectedHandCardId);
  }
  uiState.discardBrowseSelectedIds = [...currentSelection];
  syncDiscardBrowserSelectionUi(overlay, activeRequest);
}

function renderDeckBrowserOverlay(state) {
  const overlay = document.getElementById("deck-browser-overlay");
  if (!overlay) {
    return;
  }

  const attackOptionRequest = uiState.attackOptionRequest;
  if (attackOptionRequest) {
    const actionViews = (state?.legal_actions || []).filter((actionView) =>
      attackOptionRequest.actionIds.includes(actionView.action_id),
    );
    if (actionViews.length) {
      overlay.hidden = false;
      overlay.innerHTML = "";

      const browser = document.createElement("div");
      browser.className = "deck-browser";

      const header = document.createElement("div");
      header.className = "deck-browser__header";

      const copy = document.createElement("div");
      copy.className = "deck-browser__copy";

      const eyebrow = document.createElement("p");
      eyebrow.className = "deck-browser__eyebrow";
      eyebrow.textContent = "Choose Attack Effect";

      const title = document.createElement("h3");
      title.textContent = attackOptionRequest.attackName;

      const meta = document.createElement("p");
      meta.className = "deck-browser__meta";
      meta.textContent = "Choose which opposing attack to block.";

      copy.appendChild(eyebrow);
      copy.appendChild(title);
      copy.appendChild(meta);

      const controls = document.createElement("div");
      controls.className = "deck-browser__controls";

      const closeButton = document.createElement("button");
      closeButton.type = "button";
      closeButton.className = "deck-browser__close";
      closeButton.setAttribute("aria-label", "Close attack effect chooser");
      closeButton.textContent = "Close";
      controls.appendChild(closeButton);

      header.appendChild(copy);
      header.appendChild(controls);

      const track = document.createElement("div");
      track.className = "deck-browser__track";
      track.setAttribute("aria-label", "Attack effect options");

      for (const actionView of actionViews) {
        const button = document.createElement("button");
        button.type = "button";
        button.className = "action-button";
        button.textContent = actionView.label;
        button.addEventListener("click", () => {
          uiState.attackOptionRequest = null;
          submitAction(actionView);
        });
        track.appendChild(button);
      }

      browser.appendChild(header);
      browser.appendChild(track);
      overlay.appendChild(browser);

      closeButton.addEventListener("click", () => {
        uiState.attackOptionRequest = null;
        render(currentState);
      });
      return;
    }
    uiState.attackOptionRequest = null;
  }

  const activeDiscardRequest = uiState.discardBrowseRequest;
  const selectedDiscardRequest = resolveDiscardBrowseRequest(state);
  if (
    activeDiscardRequest &&
    selectedDiscardRequest &&
    activeDiscardRequest.sourceCardId === selectedDiscardRequest.sourceCardId
  ) {
    overlay.hidden = false;
    overlay.innerHTML = "";

    const browser = document.createElement("div");
    browser.className = "deck-browser";

    const header = document.createElement("div");
    header.className = "deck-browser__header";

    const copy = document.createElement("div");
    copy.className = "deck-browser__copy";

    const eyebrow = document.createElement("p");
    eyebrow.className = "deck-browser__eyebrow";
    eyebrow.textContent = "Discard Cards";

    const title = document.createElement("h3");
    title.textContent = activeDiscardRequest.sourceCardName;

    const meta = document.createElement("p");
    meta.className = "deck-browser__meta";
    meta.textContent = buildDiscardBrowserSelectionMetaText(activeDiscardRequest);

    copy.appendChild(eyebrow);
    copy.appendChild(title);
    copy.appendChild(meta);

    const controls = document.createElement("div");
    controls.className = "deck-browser__controls";

    const confirmButton = document.createElement("button");
    confirmButton.type = "button";
    confirmButton.className = "deck-browser__confirm";
    confirmButton.setAttribute("aria-label", "Confirm selected discard cards");
    confirmButton.textContent = "Confirm";
    confirmButton.disabled = true;

    const closeButton = document.createElement("button");
    closeButton.type = "button";
    closeButton.className = "deck-browser__close";
    closeButton.setAttribute("aria-label", "Close discard browser");
    closeButton.textContent = "Close";

    controls.appendChild(confirmButton);
    controls.appendChild(closeButton);
    header.appendChild(copy);
    header.appendChild(controls);

    const track = document.createElement("div");
    track.className = "deck-browser__track";
    track.setAttribute("aria-label", "Discard cards carousel");

    for (const card of selectedDiscardRequest.selectableHandCards) {
      const isSelected = uiState.discardBrowseSelectedIds.includes(card.instance_id);
      const button = document.createElement("button");
      button.type = "button";
      button.className = [
        "mini-card",
        "deck-browser-card",
        "is-match",
        "is-selectable",
        isSelected ? "is-selected" : "",
      ].filter(Boolean).join(" ");
      button.dataset.kind = card.kind || "";
      button.dataset.element = card.element || "";
      button.dataset.instanceId = card.instance_id || "";
      button.title = card.name;
      button.setAttribute("aria-pressed", isSelected ? "true" : "false");
      button.innerHTML = buildCardImageMarkup(card.image_url, card.name);
      button.addEventListener("click", (event) => {
        event.preventDefault();
        event.stopPropagation();
        if (deckBrowserDragState.isDragging) {
          return;
        }
        updateDiscardBrowserSelection(card.instance_id, overlay, activeDiscardRequest);
      });
      track.appendChild(button);
    }

    browser.appendChild(header);
    browser.appendChild(track);
    overlay.appendChild(browser);

    closeButton.addEventListener("click", () => {
      uiState.discardBrowseRequest = null;
      uiState.discardBrowseSelectedIds = [];
      render(currentState);
    });
    confirmButton.addEventListener("click", () => {
      if (uiState.discardBrowseSelectedIds.length !== activeDiscardRequest.chooseCount) {
        return;
      }
      uiState.selectedDiscardIds = [...uiState.discardBrowseSelectedIds];
      uiState.discardBrowseRequest = null;
      uiState.discardBrowseSelectedIds = [];
      openDeckBrowserForSelection(currentState);
    });
    syncDiscardBrowserSelectionUi(overlay, activeDiscardRequest);
    attachDeckBrowserDragBehavior(track);
    return;
  }

  const activeRequest = uiState.deckBrowseRequest;
  const selectedRequest = resolveDeckBrowseRequest(state);
  if (!activeRequest || !selectedRequest || activeRequest.requestKey !== selectedRequest.requestKey) {
    overlay.hidden = true;
    overlay.innerHTML = "";
    return;
  }

  const player = state?.players?.[0];
  const sourceCards = activeRequest.sourceZone === "discard"
    ? (Array.isArray(player?.discard_cards) ? player.discard_cards : [])
    : (Array.isArray(player?.deck_cards) ? player.deck_cards : []);
  const visibleCards = activeRequest.visibleCount
    ? sourceCards.slice(0, activeRequest.visibleCount)
    : sourceCards;
  const scopeLabel = activeRequest.sourceZone === "discard"
    ? "Discard Pile"
    : activeRequest.scope === "top_cards"
      ? `Top ${activeRequest.visibleCount}`
      : "Full Deck";

  overlay.hidden = false;
  overlay.innerHTML = "";

  const browser = document.createElement("div");
  browser.className = "deck-browser";

  const header = document.createElement("div");
  header.className = "deck-browser__header";

  const copy = document.createElement("div");
  copy.className = "deck-browser__copy";

  const eyebrow = document.createElement("p");
  eyebrow.className = "deck-browser__eyebrow";
  eyebrow.textContent = activeRequest.sourceZone === "discard"
    ? scopeLabel
    : `${scopeLabel} Search`;

  const title = document.createElement("h3");
  title.textContent = activeRequest.sourceCardName;

  const meta = document.createElement("p");
  meta.className = "deck-browser__meta";
  meta.textContent = buildDeckBrowserSelectionMetaText(activeRequest);

  copy.appendChild(eyebrow);
  copy.appendChild(title);
  copy.appendChild(meta);

  const controls = document.createElement("div");
  controls.className = "deck-browser__controls";

  const confirmButton = document.createElement("button");
  confirmButton.type = "button";
  confirmButton.className = "deck-browser__confirm";
  confirmButton.setAttribute("aria-label", "Confirm selected cards");
  confirmButton.textContent = "Confirm";
  confirmButton.disabled = true;

  const closeButton = document.createElement("button");
  closeButton.type = "button";
  closeButton.className = "deck-browser__close";
  closeButton.setAttribute("aria-label", "Close deck browser");
  closeButton.textContent = "Close";

  controls.appendChild(confirmButton);
  controls.appendChild(closeButton);

  header.appendChild(copy);
  header.appendChild(controls);

  const track = document.createElement("div");
  track.className = "deck-browser__track";
  track.setAttribute(
    "aria-label",
    activeRequest.sourceZone === "discard" ? "Discard pile cards carousel" : "Deck cards carousel",
  );

  for (const card of visibleCards) {
    const isMatch = cardMatchesSearchFilters(card, activeRequest.searchFilters);
    const isSelectable = !!activeRequest.actionByDeckCardId?.has(card.instance_id);
    const isSelected = uiState.deckBrowseSelectedIds.includes(card.instance_id);
    const button = document.createElement("button");
    button.type = "button";
    button.className = [
      "mini-card",
      "deck-browser-card",
      isMatch ? "is-match" : "",
      isSelectable ? "is-selectable" : "",
      isSelected ? "is-selected" : "",
    ].filter(Boolean).join(" ");
    button.dataset.kind = card.kind || "";
    button.dataset.element = card.element || "";
    button.dataset.instanceId = card.instance_id || "";
    button.title = card.name;
    button.disabled = !isSelectable;
    button.setAttribute("aria-pressed", isSelected ? "true" : "false");
    button.innerHTML = buildCardImageMarkup(card.image_url, card.name);

    if (isSelectable) {
      button.addEventListener("click", (event) => {
        event.preventDefault();
        event.stopPropagation();
        if (deckBrowserDragState.isDragging) {
          return;
        }
        updateDeckBrowserSelection(card.instance_id, overlay, activeRequest);
      });
    }

    track.appendChild(button);
  }

  browser.appendChild(header);
  browser.appendChild(track);
  overlay.appendChild(browser);

  closeButton.addEventListener("click", () => {
    uiState.deckBrowseRequest = null;
    uiState.deckBrowseSelectedIds = [];
    render(currentState);
  });
  confirmButton?.addEventListener("click", () => {
    const selectedIds = uiState.deckBrowseSelectedIds || [];
    const actionView = activeRequest.actionBySelectionKey?.get(selectionKeyForDeckIds(selectedIds));
    if (!actionView) {
      return;
    }
    uiState.deckBrowseRequest = null;
    uiState.deckBrowseSelectedIds = [];
    submitAction(actionView);
  });
  syncDeckBrowserSelectionUi(overlay, activeRequest);
  attachDeckBrowserDragBehavior(track);
}

function renderAttackDragIndicator(state) {
  const indicator = document.getElementById("attack-drag-indicator");
  if (!indicator) {
    return;
  }

  const info = resolvePendingAttackTargetingInfo(state);
  if (!info || pointerState.clientX === null || pointerState.clientY === null) {
    indicator.hidden = true;
    indicator.innerHTML = "";
    return;
  }

  indicator.hidden = false;
  indicator.innerHTML = `
    <div class="attack-drag-indicator__damage">${escapeHtml(info.damageLabel)}</div>
    <div class="attack-drag-indicator__label">${escapeHtml(info.attackName)}</div>
  `;
  syncAttackDragIndicatorPosition();
}

function syncAttackDragIndicatorPosition() {
  const indicator = document.getElementById("attack-drag-indicator");
  if (!indicator) {
    return;
  }
  if (indicator.hidden || pointerState.clientX === null || pointerState.clientY === null) {
    indicator.style.transform = "";
    return;
  }
  indicator.style.transform = `translate(${pointerState.clientX + 18}px, ${pointerState.clientY + 18}px)`;
}

function resolveSelectedPreview(state) {
  const player = state?.players?.[0];
  if (!player) {
    return null;
  }

  if (uiState.selectedCardId) {
    const handCard = player.hand.find((card) => card.instance_id === uiState.selectedCardId);
    if (handCard) {
      return {
        name: handCard.name,
        imageUrl: handCard.image_url,
        element: handCard.element,
        kind: handCard.kind,
      };
    }
  }

  const selectedTarget = uiState.selectedBoardTarget;
  if (!selectedTarget) {
    return null;
  }

  if (player.active?.ref && refsMatch(player.active.ref, selectedTarget)) {
    return {
      name: player.active.name,
      imageUrl: player.active.image_url,
      element: player.active.element,
      kind: player.active.kind,
    };
  }

  const benchedPokemon = player.bench.find((pokemon) => refsMatch(pokemon.ref, selectedTarget));
  if (benchedPokemon) {
    return {
      name: benchedPokemon.name,
      imageUrl: benchedPokemon.image_url,
      element: benchedPokemon.element,
      kind: benchedPokemon.kind,
    };
  }

  if (selectedTarget.zone === "energy") {
    const topEnergyCard = player.energy_zone?.[player.energy_zone.length - 1] || null;
    if (topEnergyCard) {
      return {
        name: topEnergyCard.name,
        imageUrl: topEnergyCard.image_url,
        element: topEnergyCard.element,
        kind: topEnergyCard.kind,
      };
    }
  }

  return null;
}

function renderContextActions(element, actions) {
  element.innerHTML = "";
  if (!actions.length) {
    element.appendChild(buildPlaceholder("No action is ready to submit yet."));
    return;
  }

  for (const actionView of actions) {
    element.appendChild(buildActionButton(actionView));
  }
}

function renderDebugActions(element, actions) {
  element.innerHTML = "";
  if (!actions.length) {
    element.appendChild(buildPlaceholder("No legal actions."));
    return;
  }

  for (const actionView of actions) {
    const source = actionView.source?.name ? `${actionView.source.name} -> ` : "";
    element.appendChild(buildActionButton(actionView, {
      debug: true,
      label: `${source}${actionView.label}`,
    }));
  }
}

function renderLog(element, logEntries, previousLogEntries) {
  element.innerHTML = "";
  const previousTexts = new Set(previousLogEntries.map((entry) => entry.text));
  const entries = [...logEntries].reverse();
  if (!entries.length) {
    element.appendChild(buildPlaceholder("The battle log will appear here once turns begin."));
    return;
  }
  for (const entry of entries) {
    const item = document.createElement("div");
    item.className = `log-entry side-${entry.side} kind-${entry.kind}`;
    if (!previousTexts.has(entry.text)) {
      item.classList.add("is-new");
    }
    item.textContent = entry.text;
    element.appendChild(item);
  }
}

function didLogEntriesChange(logEntries, previousLogEntries) {
  if (logEntries.length !== previousLogEntries.length) {
    return true;
  }

  for (let index = 0; index < logEntries.length; index += 1) {
    const currentEntry = logEntries[index];
    const previousEntry = previousLogEntries[index];
    if (
      currentEntry.text !== previousEntry.text ||
      currentEntry.side !== previousEntry.side ||
      currentEntry.kind !== previousEntry.kind
    ) {
      return true;
    }
  }

  return false;
}

function buildPokemonCard(pokemon, options) {
  if (!pokemon) {
    return buildActiveSlotPlaceholder(options);
  }
  if (pokemon.face_down) {
    return buildFaceDownPokemonCard(pokemon, options);
  }

  const card = document.createElement("article");
  const classNames = ["board-card"];
  const isCompact = !!options.compact;
  const isBenchCard = !!options.benchCard;
  if (options.clickable) {
    classNames.push("is-clickable");
  }
  if (isCompact) {
    classNames.push("is-compact");
  }
  if (isBenchCard) {
    classNames.push("is-bench-card");
  }
  if (pokemon.attached_energy?.length) {
    classNames.push("has-attached-energy");
  }
  if (options.selectedTarget && refsMatch(options.selectedTarget, pokemon.ref)) {
    classNames.push("is-selected");
  }
  if (options.context.highlightedTargets.some((target) => refsMatch(target, pokemon.ref))) {
    classNames.push("is-targetable");
  }
  if (pokemon.requires_promotion) {
    classNames.push("is-promote");
  }

  const previousPokemon = options.previousPokemon;
  if (previousPokemon) {
    if (previousPokemon.damage !== pokemon.damage) {
      classNames.push("card-anim-impact");
    }
    if (previousPokemon.instance_id !== pokemon.instance_id) {
      classNames.push("card-anim-promotion");
    }
  }

  card.className = classNames.join(" ");
  if (isBenchCard) {
    const benchCardWidth = pokemon.ref?.player_index === 1 ? 108 : 124;
    const attachedEnergyOverhang = (pokemon.attached_energy?.length || 0) * benchCardWidth * 0.1;
    card.style.setProperty("--bench-card-width", `${benchCardWidth}px`);
    card.style.setProperty("--attached-energy-overhang", `${attachedEnergyOverhang}px`);
  }
  card.dataset.element = pokemon.element || "";
  card.dataset.kind = pokemon.kind || "";
  card.title = pokemon.name;
  card.setAttribute("aria-label", pokemon.name);
  const hpPercent = pokemon.hp ? Math.max(0, Math.round((pokemon.remaining_hp / pokemon.hp) * 100)) : 0;
  const healthClass = hpPercent <= 33 ? "is-danger" : hpPercent <= 66 ? "is-warning" : "";
  const primaryStageLabel = isCompact
    ? `HP ${pokemon.remaining_hp}/${pokemon.hp}`
    : `${formatStageLabel(pokemon.stage)} • ${formatElementLabel(pokemon.element)}`;
  const compactTags = isCompact ? describeCompactPokemonTags(pokemon) : "";
  const compactStatPills = [];
  if (pokemon.damage > 0) {
    compactStatPills.push(`<span class="info-pill">Damage ${pokemon.damage}</span>`);
  }
  if (pokemon.requires_promotion) {
    compactStatPills.push('<span class="info-pill">Promote</span>');
  }
  if (!compactStatPills.length && isCompact) {
    compactStatPills.push('<span class="info-pill">Benched</span>');
  }
  if (isBenchCard) {
    card.innerHTML = `${buildPokemonImageMarkup(pokemon)}`;
    if (options.clickable) {
      card.addEventListener("click", () => toggleSelectedBoardTarget(pokemon.ref));
    }
    return card;
  }

  card.innerHTML = `
    ${buildPokemonImageMarkup(pokemon)}
    <div class="card-copy">
      <div class="card-title-row">
        <div class="card-title-group">
          <div class="card-title">${escapeHtml(pokemon.name)}</div>
          <p class="card-subtitle">${escapeHtml(primaryStageLabel)}</p>
        </div>
        <div class="stage-pill">${escapeHtml(formatStagePillLabel(pokemon.stage))}</div>
      </div>
      ${isCompact ? "" : `
        <div class="health-track" aria-hidden="true">
          <span class="health-track-fill ${healthClass}" style="width: ${hpPercent}%"></span>
        </div>
      `}
      <div class="card-stat-row">
        ${isCompact ? compactStatPills.join("") : `
          <span class="info-pill">Damage ${pokemon.damage}</span>
          <span class="info-pill">HP ${pokemon.remaining_hp}/${pokemon.hp}</span>
        `}
      </div>
      ${isCompact ? "" : `
        <div class="attack-list">
          ${pokemon.attacks.map((attack, attackIndex) => renderAttackChip(pokemon, attack, attackIndex, options)).join("")}
        </div>
      `}
      ${isCompact && compactTags !== "Benched" ? `<p class="card-tags">${escapeHtml(compactTags)}</p>` : ""}
    </div>
  `;

  const attackButtons = card.querySelectorAll(".attack-chip-button[data-attack-index]");
  for (const attackButton of attackButtons) {
    attackButton.addEventListener("click", (event) => {
      event.stopPropagation();
      const attackIndex = Number(attackButton.dataset.attackIndex);
      handleAttackButtonClick(pokemon, attackIndex);
    });
  }

  if (options.clickable) {
    card.addEventListener("click", () => toggleSelectedBoardTarget(pokemon.ref));
  }
  return card;
}

function buildFaceDownPokemonCard(pokemon, options) {
  const card = document.createElement("article");
  const classNames = ["board-card", "board-card-facedown"];
  if (options.compact) {
    classNames.push("is-compact");
  }
  if (options.benchCard) {
    classNames.push("is-bench-card");
  }
  card.className = classNames.join(" ");
  card.dataset.element = "";
  card.dataset.kind = "pokemon";
  card.title = pokemon.name || "Face-down Pokemon";
  card.setAttribute("aria-label", pokemon.name || "Face-down Pokemon");
  card.innerHTML = `
    <div class="card-visual">
      <img src="${escapeHtml(pokemon.image_url || resolveFaceDownCardImageUrl(currentState))}" alt="${escapeHtml(pokemon.name || "Face-down Pokemon")}" loading="lazy" />
    </div>
    <div class="card-copy card-copy-facedown">
      <div class="card-title-row">
        <div class="card-title-group">
          <div class="card-title">Face-down Pokemon</div>
          <p class="card-subtitle">Reveals after setup is complete</p>
        </div>
        <div class="stage-pill">Hidden</div>
      </div>
    </div>
  `;
  return card;
}

function buildActiveSlotPlaceholder(options = {}) {
  const element = document.createElement("article");
  const classNames = ["board-card", "active-slot-placeholder"];
  if (options.clickable) {
    classNames.push("is-clickable");
  }
  if (options.selectedTarget && refsMatch(options.selectedTarget, options.targetRef)) {
    classNames.push("is-selected");
  }
  if (options.context?.highlightedTargets.some((target) => refsMatch(target, options.targetRef))) {
    classNames.push("is-targetable");
  }
  element.className = classNames.join(" ");
  element.setAttribute("aria-label", "Active empty");
  const copy = document.createElement("div");
  copy.className = "active-slot-placeholder-copy";
  if (options.setupPromptAction) {
    const prompt = document.createElement("div");
    prompt.className = "active-slot-setup-prompt";
    prompt.innerHTML = `
      <p class="active-slot-setup-prompt__title">No Basic Pokemon in hand</p>
      <p class="active-slot-setup-prompt__copy">Return your opening hand to the deck, shuffle, and draw 7 new cards.</p>
    `;
    const button = document.createElement("button");
    button.type = "button";
    button.className = "primary active-slot-setup-prompt__button";
    button.textContent = options.setupPromptAction.label || "Okay";
    button.disabled = aiIsRunning;
    button.addEventListener("click", (event) => {
      event.preventDefault();
      event.stopPropagation();
      submitAction(options.setupPromptAction);
    });
    prompt.appendChild(button);
    copy.appendChild(prompt);
  } else {
    copy.appendChild(buildStackedStamp(["Active", "Empty"], "bench-empty-stamp active-slot-empty-stamp"));
  }
  element.appendChild(copy);
  if (!options.setupPromptAction && options.clickable && options.targetRef) {
    element.addEventListener("click", () => toggleSelectedBoardTarget(options.targetRef));
  }
  return element;
}

function buildMiniCard(card, options = {}) {
  const element = document.createElement("article");
  const classNames = ["mini-card"];
  if (options.clickable) {
    classNames.push("is-clickable");
  }
  if (options.playable) {
    classNames.push("is-playable");
  }
  if (options.selected) {
    classNames.push("is-selected");
  }
  if (options.targetable) {
    classNames.push("is-targetable");
  }
  if (options.animationClass) {
    classNames.push(options.animationClass);
  }
  element.className = classNames.join(" ");
  element.dataset.element = card.element || "";
  element.dataset.kind = card.kind || "";
  const copyMarkup = options.hideCopy
    ? ""
    : (() => {
        const typeLabel = formatCardAccentLabel(card);
        const meta = options.metaText ??
          [formatKindLabel(card.kind), card.stage ? formatStagePillLabel(card.stage) : ""]
            .filter(Boolean)
            .join(" • ");
        const accentMarkup = options.hideAccent
          ? ""
          : `<div class="type-pill">${escapeHtml(typeLabel)}</div>`;
        return `
          <div class="card-copy">
            <div class="card-title-row">
              <div class="card-title-group">
                <div class="card-title">${escapeHtml(card.name)}</div>
                <p class="card-meta">${escapeHtml(meta)}</p>
              </div>
              ${accentMarkup}
            </div>
          </div>
        `;
      })();
  element.innerHTML = `
    <div class="mini-card-media">
      ${buildCardImageMarkup(card.image_url, card.name)}
      ${options.playable ? '<div class="mini-card-badge">Ready</div>' : ""}
    </div>
    ${copyMarkup}
  `;
  return element;
}

function buildFaceDownPileCard(options = {}) {
  const element = document.createElement("article");
  const classNames = ["mini-card", "pile-card", "facedown-pile-card"];
  if (options.kind) {
    classNames.push(`facedown-pile-card-${options.kind}`);
  }
  if (options.clickable) {
    classNames.push("is-clickable");
  }
  if (options.selected) {
    classNames.push("is-selected");
  }
  if (options.targetable) {
    classNames.push("is-targetable");
  }
  if (options.stacked) {
    classNames.push("is-stacked");
  }
  element.className = classNames.join(" ");
  element.dataset.kind = options.kind || "facedown";
  element.title = options.description || options.title || "Face-down pile";
  element.setAttribute("aria-label", element.title);
  element.innerHTML = `
    <div class="mini-card-media">
      ${buildCardImageMarkup(options.imageUrl, options.title || "Face-down card")}
    </div>
  `;
  if (options.badgeText !== undefined && options.badgeText !== null) {
    element.appendChild(buildPileCountBadge(options.badgeText));
  }
  return element;
}

function buildPileEmptyCard(options = {}) {
  const element = document.createElement("article");
  const classNames = ["mini-card", "pile-card", "pile-card-empty"];
  if (options.kind) {
    classNames.push(`pile-card-empty-${options.kind}`);
  }
  element.className = classNames.join(" ");
  element.dataset.kind = options.kind || "pile";
  element.title = `${options.label || "Pile"} pile empty`;
  element.setAttribute("aria-label", element.title);
  element.appendChild(buildEmptyStamp(options.label || "Pile", "bench-empty-stamp pile-empty-stamp"));
  return element;
}

function buildPileCountBadge(count) {
  const badge = document.createElement("div");
  badge.className = "pile-count-badge";
  badge.textContent = String(count);
  return badge;
}

function buildPlaceholder(text) {
  const element = document.createElement("div");
  element.className = "placeholder-card";
  element.textContent = text;
  return element;
}

function buildCardImageMarkup(imageUrl, altText) {
  if (!imageUrl) {
    return `<div class="placeholder-card">${escapeHtml(altText)}</div>`;
  }
  return `<img src="${escapeHtml(imageUrl)}" alt="${escapeHtml(altText)}" loading="lazy" draggable="false" />`;
}

function buildAttachedEnergyMarkup(attachedEnergy) {
  if (!Array.isArray(attachedEnergy) || !attachedEnergy.length) {
    return "";
  }

  return `
    <div class="attached-energy-stack" aria-hidden="true">
      ${attachedEnergy
        .map(
          (energyCard, index) => `
            <div class="attached-energy-card" style="--attached-energy-index: ${index + 1}; --attached-energy-z: ${attachedEnergy.length - index};">
              ${buildCardImageMarkup(energyCard.image_url, energyCard.name)}
            </div>
          `,
        )
        .join("")}
    </div>
  `;
}

function buildPokemonImageMarkup(pokemon) {
  const attachedEnergy = Array.isArray(pokemon.attached_energy) ? pokemon.attached_energy : [];
  const attachedEnergyMarkup = buildAttachedEnergyMarkup(attachedEnergy);
  const classNames = ["card-visual"];
  if (attachedEnergy.length) {
    classNames.push("has-attached-energy");
  }
  const overlay =
    pokemon.remaining_hp !== pokemon.hp
      ? `
        <div class="card-hp-badge">
          <span class="card-hp-value">${pokemon.remaining_hp}<span class="card-hp-total">/${pokemon.hp}</span></span>
        </div>
      `
      : "";

  if (!pokemon.image_url) {
    return `
      <div class="${classNames.join(" ")} card-visual-fallback">
        ${attachedEnergyMarkup}
        <div class="pokemon-visual-card">
          <div class="placeholder-card">${escapeHtml(pokemon.name)}</div>
          ${overlay}
        </div>
      </div>
    `;
  }

  return `
    <div class="${classNames.join(" ")}">
      ${attachedEnergyMarkup}
      <div class="pokemon-visual-card">
        <img src="${escapeHtml(pokemon.image_url)}" alt="${escapeHtml(pokemon.name)}" loading="lazy" />
        ${overlay}
      </div>
    </div>
  `;
}

function renderAttackChip(pokemon, attack, attackIndex, options) {
  const attackActions = findAttackActionsForPokemon(pokemon, attackIndex);
  const isPlayersActivePokemon =
    options.clickable &&
    pokemon.ref?.player_index === 0 &&
    pokemon.ref?.zone === "active";
  const tagName = isPlayersActivePokemon ? "button" : "div";
  const classes = ["attack-chip"];
  if (isPlayersActivePokemon) {
    classes.push("attack-chip-button");
    if (attackActions.length) {
      classes.push("is-ready");
    }
  }
  const effect =
    attack.effect && attack.effect !== "none"
      ? `<span class="attack-effect">${escapeHtml(formatEffectLabel(attack.effect))}</span>`
      : "";
  const tagAttributes = [
    `class="${classes.join(" ")}"`,
    `data-attack-index="${attackIndex}"`,
  ];
  if (tagName === "button") {
    tagAttributes.push('type="button"');
    if (!attackActions.length || aiIsRunning) {
      tagAttributes.push("disabled");
    }
  }
  return `
    <${tagName} ${tagAttributes.join(" ")}>
      <div class="attack-topline">
        <span class="attack-name">${escapeHtml(attack.name)}</span>
        <span class="attack-stats">Cost ${attack.cost} • Damage ${attack.damage}</span>
      </div>
      ${effect}
    </${tagName}>
  `;
}

function findAttackActionsForPokemon(pokemon, attackIndex) {
  if (!currentState || !pokemon?.ref) {
    return [];
  }

  return currentState.legal_actions.filter(
    (actionView) =>
      actionView.type === "attack" &&
      actionView.source?.zone === pokemon.ref.zone &&
      actionView.source?.player_index === pokemon.ref.player_index &&
      actionView.source?.instance_id === pokemon.ref.instance_id &&
      actionView.action?.attack_index === attackIndex,
  );
}

function handleAttackButtonClick(pokemon, attackIndex) {
  const attackActions = findAttackActionsForPokemon(pokemon, attackIndex);
  if (!attackActions.length) {
    return;
  }

  if (attackActions.length === 1) {
    uiState.pendingAttackActionIds = [];
    uiState.attackOptionRequest = null;
    submitAction(attackActions[0]);
    return;
  }

  const searchActions = attackActions.filter((actionView) =>
    Array.isArray(actionView.action?.search_deck_ids),
  );
  if (searchActions.length) {
    const searchEffect = (pokemon.attacks?.[attackIndex]?.effect_specs || []).find(
      (effectSpec) =>
        effectSpec.effect_type === "search_deck" &&
        effectSpec.source_zone === "deck" &&
        effectSpec.destination_zone,
    );
    const chooseCount = Number.isInteger(searchEffect?.choose_count) ? searchEffect.choose_count : 1;
    const playerBench = currentState?.players?.[pokemon.ref?.player_index || 0]?.bench || [];
    const remainingBenchSlots = Math.max(0, BENCH_LIMIT - playerBench.length);
    if (
      searchEffect?.destination_zone === "bench" &&
      remainingBenchSlots === 0
    ) {
      const emptySearchAction = searchActions.find(
        (actionView) => (actionView.action?.search_deck_ids || []).length === 0,
      ) || searchActions[0];
      uiState.selectedCardId = null;
      uiState.selectedDiscardIds = [];
      uiState.selectedBoardTarget = null;
      uiState.pendingAttackActionIds = [];
      uiState.attackOptionRequest = null;
      submitAction(emptySearchAction);
      return;
    }

    const searchRequest = resolveAttackDeckBrowseRequest(currentState, {
      sourceRef: pokemon.ref ? { ...pokemon.ref } : null,
      attackIndex,
    });
    if (!searchRequest) {
      uiState.pendingAttackActionIds = [];
      uiState.attackOptionRequest = null;
      submitAction(searchActions[0]);
      return;
    }

    uiState.selectedCardId = null;
    uiState.selectedDiscardIds = [];
    uiState.discardBrowseRequest = null;
    uiState.discardBrowseSelectedIds = [];
    uiState.selectedBoardTarget = null;
    uiState.pendingAttackActionIds = [];
    uiState.attackOptionRequest = null;
    uiState.deckBrowseRequest = searchRequest;
    uiState.deckBrowseSelectedIds = [];
    render(currentState);
    return;
  }

  const optionActions = attackActions.filter((actionView) =>
    Number.isInteger(actionView.action?.blocked_attack_index),
  );
  if (optionActions.length) {
    uiState.selectedCardId = null;
    uiState.selectedBoardTarget = null;
    uiState.pendingAttackActionIds = [];
    uiState.attackOptionRequest = {
      attackName: pokemon.attacks?.[attackIndex]?.name || optionActions[0].label,
      actionIds: optionActions.map((actionView) => actionView.action_id),
    };
    render(currentState);
    return;
  }

  const targetedActions = attackActions.filter(
    (actionView) =>
      actionView.target &&
      !refsMatch(actionView.target, actionView.source),
  );

  if (targetedActions.length) {
    uiState.selectedCardId = null;
    uiState.selectedBoardTarget = pokemon.ref ? { ...pokemon.ref } : null;
    uiState.pendingAttackActionIds = targetedActions.map((actionView) => actionView.action_id);
    uiState.attackOptionRequest = null;
    render(currentState);
    return;
  }

  uiState.pendingAttackActionIds = [];
  if (attackActions.length > 1) {
    uiState.attackOptionRequest = {
      attackName: pokemon.attacks?.[attackIndex]?.name || attackActions[0].label,
      actionIds: attackActions.map((actionView) => actionView.action_id),
    };
    render(currentState);
    return;
  }
  uiState.attackOptionRequest = null;
  submitAction(attackActions[0]);
}

function findAttackActionForPokemon(pokemon, attackIndex) {
  return (
    findAttackActionsForPokemon(pokemon, attackIndex).find(
      (actionView) =>
        !actionView.target ||
        refsMatch(actionView.target, actionView.source),
    ) || null
  );
}

function describeCompactPokemonTags(pokemon) {
  const tags = [];
  if (pokemon.requires_promotion) {
    tags.push("Must promote");
  } else if (pokemon.can_attack) {
    tags.push("Ready");
  }
  if (pokemon.target_action_types.length) {
    tags.push("Target");
  }
  return tags.length ? tags.join(" • ") : "Benched";
}

function renderMatchMeta(state) {
  const sessionChip = document.getElementById("session-chip");
  const phaseChip = document.getElementById("phase-chip");

  sessionChip.textContent = `Session ${shortSessionId(state.session_id)}`;
  sessionChip.title = state.session_id;
  document.getElementById("turn-chip").textContent = `Turn ${state.turn_number}`;

  phaseChip.textContent = makePhaseLabel(state);
  phaseChip.className =
    state.winner !== null
      ? "session-chip session-chip-muted"
      : state.current_player === 0 || state.pending_promotion_for === 0
        ? "session-chip session-chip-accent"
        : "session-chip";
}

function renderStandardAiModeToggle(state) {
  const toggle = document.getElementById("standard-ai-mode-toggle");
  const toggleChip = document.getElementById("standard-ai-mode-chip");
  const toggleText = document.getElementById("standard-ai-mode-text");
  if (!toggle || !toggleChip || !toggleText) {
    return;
  }

  const isStandardMode = resolveGameMode(state) === "standard";
  const selectedMode = isStandardMode ? resolveStandardAiMode(state) : "local";
  const remoteEnabled = selectedMode === "remote";
  const status = uiState.standardMlStatus;
  let description = remoteEnabled ? "Remote NN" : "Local AI";
  if (!isStandardMode) {
    description = "Standard only";
  } else if (uiState.standardAiModePending) {
    description = "Checking...";
  } else if (remoteEnabled && status?.model_loaded) {
    description = "Remote ready";
  } else if (remoteEnabled) {
    description = "Remote NN";
  } else if (status && !status.ready) {
    description = "Remote offline";
  }

  toggle.checked = remoteEnabled;
  toggle.disabled = aiIsRunning || uiState.standardAiModePending || !isStandardMode;
  toggleText.textContent = description;
  toggleChip.classList.toggle("is-enabled", remoteEnabled);
  toggleChip.classList.toggle("is-disabled", toggle.disabled);
  toggleChip.title =
    status?.error ||
    (remoteEnabled
      ? `Remote inference${status?.backend ? ` via ${status.backend}` : ""}`
      : "Use the local Standard AI.");
}

function renderTrainerPicker(state) {
  const trainerSelect = document.getElementById("trainer-select");
  const trainerMeta = document.getElementById("trainer-meta");
  const trainerProgress = document.getElementById("trainer-progress");
  const trainerProgressLabel = document.getElementById("trainer-progress-label");
  const trainerProgressValue = document.getElementById("trainer-progress-value");
  const trainerProgressFill = document.getElementById("trainer-progress-fill");
  if (
    !trainerSelect ||
    !trainerMeta ||
    !trainerProgress ||
    !trainerProgressLabel ||
    !trainerProgressValue ||
    !trainerProgressFill
  ) {
    return;
  }

  const trainers = state.available_trainers || [];
  const selectedTrainerId = resolveTrainerSelection(state);
  trainerSelect.innerHTML = trainers
    .map(
      (trainer) => `
        <option value="${escapeHtml(trainer.id)}">${escapeHtml(buildTrainerLabel(trainer))}</option>
      `,
    )
    .join("");
  trainerSelect.value = selectedTrainerId;
  trainerSelect.disabled = aiIsRunning;

  const selectedTrainer =
    trainers.find((trainer) => trainer.id === selectedTrainerId) || state.ai_trainer || null;
  if (!selectedTrainer) {
    trainerMeta.textContent = "";
    trainerProgress.hidden = true;
    return;
  }

  trainerMeta.textContent = "";

  const progress = getTrainerLevelProgress(selectedTrainer);
  trainerProgress.hidden = false;
  trainerProgressLabel.textContent = `${progress.xpToNextLevel} XP to Lv. ${progress.nextLevel}`;
  trainerProgressValue.textContent = `${progress.xpIntoLevel} / ${progress.totalLevelXp} XP`;
  trainerProgressFill.style.width = `${progress.progressPercent}%`;
}

function renderDeckPicker(state) {
  const deckSelect = document.getElementById("deck-select");
  if (!deckSelect) {
    return;
  }

  const decks = state.available_decks || [];
  const selectedHumanDeckId = resolveHumanDeckSelection(state);
  deckSelect.innerHTML = decks
    .map(
      (deck) => `
        <option value="${escapeHtml(deck.id)}">${escapeHtml(buildDeckLabel(deck))}</option>
      `,
    )
    .join("");
  if (selectedHumanDeckId) {
    deckSelect.value = selectedHumanDeckId;
  }
  deckSelect.disabled = aiIsRunning || !decks.length;
}

function renderLobbyMatchMeta() {
  const sessionChip = document.getElementById("session-chip");
  const phaseChip = document.getElementById("phase-chip");
  const turnChip = document.getElementById("turn-chip");

  turnChip.textContent = "No Game";
  sessionChip.textContent = "Session pending";
  sessionChip.title = "";
  phaseChip.textContent = "Lobby";
  phaseChip.className = "session-chip session-chip-muted";
}

function renderActiveView() {
  if (currentState) {
    render(currentState);
    return;
  }
  if (lobbyState) {
    renderLobby(lobbyState);
  }
}

function renderOpponentIdentity(state) {
  const opponentName = document.getElementById("opponent-name");
  const opponentKicker = document.getElementById("opponent-kicker");
  if (!opponentName || !opponentKicker) {
    return;
  }

  const trainer = state.ai_trainer;
  if (!trainer) {
    opponentName.textContent = "AI Board";
    opponentKicker.textContent = "Opponent";
    return;
  }

  opponentName.textContent = trainer.name;
  opponentKicker.textContent = `${trainer.specialty} Gym Leader • Lv. ${trainer.level}`;
}

function renderTurnHighlights(state) {
  applyBoardTurnState(
    {
      humanElement: state.players[0].element,
      aiElement: state.players[1].element,
    },
    state.current_player,
  );
}

function renderLobbyBoard(state) {
  const faceDownCardImageUrl = resolveFaceDownCardImageUrl(state);
  const sharedEnergyEnabled = usesSharedEnergyPool(state);
  const selectedDeckId = resolveHumanDeckSelection(state);
  const humanDeck =
    state.available_decks?.find((deck) => deck.id === selectedDeckId) || state.available_decks?.[0] || null;
  const aiDeck =
    state.available_decks?.find((deck) => deck.id === humanDeck?.paired_deck_id) || null;

  applyBoardTurnState(
    {
      humanElement: humanDeck?.element || "",
      aiElement: aiDeck?.element || "",
    },
    null,
  );
  setSharedEnergyVisibility(sharedEnergyEnabled);
  renderDeckPlaceholder(document.getElementById("player-deck"), faceDownCardImageUrl);
  renderDeckPlaceholder(document.getElementById("opponent-deck"), faceDownCardImageUrl);
  renderSharedEnergyPlaceholder(document.getElementById("player-shared-energy"), 0);
  renderSharedEnergyPlaceholder(document.getElementById("opponent-shared-energy"), 0);
  renderPrizePlaceholder(
    document.getElementById("player-prizes"),
    3,
    humanDeck?.prize_coin_image_url || faceDownCardImageUrl,
  );
  renderPrizePlaceholder(
    document.getElementById("opponent-prizes"),
    3,
    aiDeck?.prize_coin_image_url || faceDownCardImageUrl,
  );
  renderDiscard(document.getElementById("player-discard"), null, 0);
  renderDiscard(document.getElementById("opponent-discard"), null, 0);
  renderPokemonZone(document.getElementById("player-active"), null, {});
  renderPokemonZone(document.getElementById("opponent-active"), null, {});
  renderPokemonList(document.getElementById("player-bench"), [], { previousBench: [] });
  renderPokemonList(document.getElementById("opponent-bench"), [], { previousBench: [] });
  syncBenchZoneState(document.getElementById("player-bench-zone"), null);

  const playerHand = document.getElementById("player-hand");
  playerHand.innerHTML = "";
  playerHand.appendChild(buildPlaceholder("Choose a deck, then start a new game."));

  const battleLog = document.getElementById("battle-log");
  battleLog.innerHTML = "";
  battleLog.appendChild(buildPlaceholder("Battle log entries appear once a game starts."));

  const debugActions = document.getElementById("debug-actions");
  debugActions.innerHTML = "";
  debugActions.appendChild(buildPlaceholder("Start a new game to inspect legal actions."));

  renderPlayerEndTurnButtonLobby();
  renderSelectedCardPreview(null);
  renderDeckBrowserOverlay(null);
  previousState = null;
}

function applyBoardTurnState(elements, activePlayerIndex) {
  const boardPanels = [
    {
      element: document.getElementById("player-board-panel"),
      deckElement: elements.humanElement,
      isActive: activePlayerIndex === 0,
    },
    {
      element: document.getElementById("opponent-board-panel"),
      deckElement: elements.aiElement,
      isActive: activePlayerIndex === 1,
    },
  ];

  for (const boardPanel of boardPanels) {
    if (!boardPanel.element) {
      continue;
    }
    boardPanel.element.dataset.element = boardPanel.deckElement || "";
    boardPanel.element.classList.toggle("is-turn-active", boardPanel.isActive);
  }
}

function renderDeckPlaceholder(element, imageUrl) {
  if (!element) {
    return;
  }
  element.innerHTML = "";
  element.appendChild(
    buildFaceDownPileCard({
      imageUrl,
      title: "Deck pile",
      description: "Deck pile",
      badgeText: "-",
      kind: "deck",
      stacked: true,
    }),
  );
}

function renderSharedEnergyPlaceholder(element, count) {
  if (!element) {
    return;
  }
  element.innerHTML = "";
  element.appendChild(
    buildSharedEnergyHolderCard({
      energyCard: null,
      count,
      clickable: false,
      selected: false,
      targetable: false,
    }),
  );
}

function renderPrizePlaceholder(element, count, imageUrl) {
  renderPrizePile(
    element,
    {
      count,
      image_url: imageUrl,
    },
    {
      imageUrl,
    },
  );
}

function renderPlayerEndTurnButton(state) {
  const button = document.getElementById("player-end-turn-button");
  if (!button) {
    return;
  }

  const primaryAction = findPrimaryTurnAction(state);
  button.disabled = aiIsRunning || !primaryAction;
  button.classList.toggle("is-ready", !!primaryAction && !aiIsRunning);
  setPrimaryTurnButtonLabel(button, primaryAction);
  button.onclick = primaryAction ? () => submitAction(primaryAction) : null;
}

function renderPlayerEndTurnButtonLobby() {
  const button = document.getElementById("player-end-turn-button");
  if (!button) {
    return;
  }

  button.disabled = true;
  button.classList.remove("is-ready");
  button.onclick = null;
  setPrimaryTurnButtonLabel(button, null);
}

function findPrimaryTurnAction(state) {
  if (!state?.legal_actions?.length) {
    return null;
  }
  return (
    state.legal_actions.find((actionView) => actionView.type === "end_setup") ||
    state.legal_actions.find((actionView) => actionView.type === "end_turn") ||
    null
  );
}

function setPrimaryTurnButtonLabel(button, actionView) {
  const labelElement = button.querySelector(".end-turn-button__label");
  if (!labelElement) {
    return;
  }
  if (actionView?.type === "end_setup") {
    labelElement.innerHTML = "END<br />SETUP";
    return;
  }
  labelElement.innerHTML = "END<br />TURN";
}

function buildActionButton(actionView, options = {}) {
  const button = document.createElement("button");
  const classes = ["action-button", classifyAction(actionView)];
  if (options.debug) {
    classes.push("is-debug");
  }

  button.className = classes.filter(Boolean).join(" ");
  button.disabled = aiIsRunning;
  button.innerHTML = `
    <span class="action-copy">
      <span class="action-label">${escapeHtml(options.label || actionView.label)}</span>
      <span class="action-meta">${escapeHtml(options.debug ? `Debug • ${describeActionType(actionView.type)}` : describeActionType(actionView.type))}</span>
    </span>
  `;
  button.addEventListener("click", () => submitAction(actionView));
  return button;
}

function findOpeningMulliganAction(state) {
  if (!state || state.game_mode !== "standard") {
    return null;
  }
  return state.legal_actions.find((actionView) => actionView.type === "mulligan") || null;
}

function classifyAction(actionView) {
  if (actionView.type === "attack") {
    return "is-attack";
  }
  if (actionView.type === "end_turn" || actionView.type === "end_setup" || actionView.type === "promote") {
    return "is-primary";
  }
  return "is-support";
}

function describeActionType(actionType) {
  const labels = {
    attack: "Attack",
    bench_basic: "Bench play",
    end_setup: "Setup",
    end_turn: "Turn action",
    evolve: "Evolution",
    play_energy: "Energy attach",
    play_item: "Item",
    play_potion: "Support effect",
    play_supporter: "Supporter",
    play_switch: "Switch",
    promote: "Required choice",
    retreat: "Retreat",
  };
  return labels[actionType] || actionType.replaceAll("_", " ");
}

function formatStageLabel(stage) {
  if (!stage || stage === "basic") {
    return "Basic Pokemon";
  }
  const numberedStageMatch = stage.match(/^stage(\d+)$/);
  if (numberedStageMatch) {
    return `Stage ${numberedStageMatch[1]} Pokemon`;
  }
  return stage.replaceAll("_", " ");
}

function formatStagePillLabel(stage) {
  if (!stage || stage === "basic") {
    return "Basic";
  }
  const numberedStageMatch = stage.match(/^stage(\d+)$/);
  if (numberedStageMatch) {
    return `Stage ${numberedStageMatch[1]}`;
  }
  return stage.replaceAll("_", " ");
}

function formatElementLabel(element) {
  if (!element) {
    return "Neutral";
  }
  return `${element[0].toUpperCase()}${element.slice(1)}`;
}

function formatKindLabel(kind) {
  if (!kind) {
    return "Card";
  }
  return `${kind[0].toUpperCase()}${kind.slice(1)}`;
}

function formatCardAccentLabel(card) {
  if (card.kind === "trainer") {
    return "Trainer";
  }
  if (card.element) {
    return formatElementLabel(card.element);
  }
  return formatKindLabel(card.kind);
}

function formatEffectLabel(effect) {
  const labels = {
    coin_flip_bonus_30: "Coin flip for +30 bonus",
    coin_flip_fail: "Miss on tails",
  };
  return labels[effect] || effect.replaceAll("_", " ");
}

function shortSessionId(sessionId) {
  return sessionId.slice(0, 8);
}

function makePhaseLabel(state) {
  if (state.winner !== null) {
    return "Match over";
  }
  if (state.game_mode === "standard" && state.setup_phase === "choose_active") {
    return "Choose active";
  }
  if (state.game_mode === "standard" && state.setup_phase === "awaiting_end_setup") {
    return "End setup";
  }
  if (state.pending_promotion_for === 0) {
    return "Choose promotion";
  }
  return state.current_player === 0 ? "Your move" : "AI thinking";
}

function toggleSelectedCard(instanceId) {
  const result = resolveSelectedCardClick({
    state: currentState,
    uiState,
    instanceId,
    aiIsRunning,
  });

  if (result.autoAction) {
    uiState.selectedDiscardIds = [];
    submitAction(result.autoAction);
    return;
  }

  const previousSelectedCardId = uiState.selectedCardId;
  uiState.selectedCardId = result.nextUiState.selectedCardId;
  if (uiState.selectedCardId !== previousSelectedCardId) {
    uiState.selectedDiscardIds = [];
  }
  uiState.selectedBoardTarget = result.nextUiState.selectedBoardTarget;
  uiState.pendingAttackActionIds = [];
  render(currentState);
}

function toggleSelectedBoardTarget(targetRef) {
  const previousSelectedBoardTarget = uiState.selectedBoardTarget;
  const result = resolveSelectedBoardTargetClick({
    state: currentState,
    uiState,
    targetRef,
    aiIsRunning,
  });

  if (result.autoAction) {
    uiState.selectedDiscardIds = [];
    submitAction(result.autoAction);
    return;
  }

  uiState.selectedCardId = result.nextUiState.selectedCardId;
  uiState.selectedBoardTarget = result.nextUiState.selectedBoardTarget;
  uiState.pendingAttackActionIds =
    previousSelectedBoardTarget &&
    result.nextUiState.selectedBoardTarget &&
    refsMatch(previousSelectedBoardTarget, result.nextUiState.selectedBoardTarget)
      ? uiState.pendingAttackActionIds
      : [];
  render(currentState);
}

function tryAutoSubmitBenchPlay() {
  if (!currentState) {
    return false;
  }
  const actionView = findBenchPlayAction(currentState);
  if (!actionView) {
    return false;
  }

  submitAction(actionView);
  return true;
}

function findBenchPlayAction(state) {
  if (!state) {
    return null;
  }
  return findBenchPlayActionForSelection(state, uiState.selectedCardId, aiIsRunning);
}

function findBackgroundPlayAction(state) {
  if (!state) {
    return null;
  }
  return findBackgroundPlayActionForSelection(
    state,
    uiState.selectedCardId,
    aiIsRunning,
    uiState.selectedDiscardIds,
  );
}

function syncBenchZoneState(element, state) {
  if (!element) {
    return;
  }

  element.classList.toggle("is-actionable", !!findBenchPlayAction(state));
}

function deriveContext(state) {
  return deriveInteractionContext(state, uiState);
}

function describeSelection(state, context) {
  const fragments = [];
  const discardRequirement = resolveDiscardFromHandRequirement(state, uiState.selectedCardId);
  const pendingAttackInfo = resolvePendingAttackTargetingInfo(state);
  if (pendingAttackInfo) {
    fragments.push(`Attack: ${pendingAttackInfo.attackName}`);
  }
  if (uiState.selectedCardId) {
    const card = state.players[0].hand.find((item) => item.instance_id === uiState.selectedCardId);
    if (card) {
      fragments.push(`Hand: ${card.name}`);
    }
  }
  if (uiState.selectedBoardTarget) {
    const target = findBoardTargetLabel(state.players[0], uiState.selectedBoardTarget);
    if (target) {
      fragments.push(`Board: ${target}`);
    }
  }
  if (discardRequirement) {
    fragments.push(
      `Discard: ${uiState.selectedDiscardIds.length}/${discardRequirement.chooseCount}`,
    );
  }
  if (!fragments.length) {
    if (context.actions.length) {
      return "System actions ready.";
    }
    return "No selection yet.";
  }
  return fragments.join(" • ");
}

function resolvePendingAttackTargetingInfo(state) {
  if (!state || !uiState.pendingAttackActionIds.length || !uiState.selectedBoardTarget) {
    return null;
  }

  const pendingActions = state.legal_actions.filter((actionView) =>
    uiState.pendingAttackActionIds.includes(actionView.action_id),
  );
  if (!pendingActions.length) {
    return null;
  }

  const actionView = pendingActions[0];
  const sourceRef = actionView.source;
  if (!sourceRef) {
    return null;
  }

  const player = state.players?.[sourceRef.player_index];
  if (!player) {
    return null;
  }

  const pokemon =
    sourceRef.zone === "active"
      ? player.active
      : sourceRef.zone === "bench"
        ? player.bench[sourceRef.bench_index]
        : null;
  const attackIndex = actionView.action?.attack_index;
  const attack = Number.isInteger(attackIndex) ? pokemon?.attacks?.[attackIndex] : null;
  if (!attack) {
    return null;
  }

  const targetedEffect = (attack.effect_specs || []).find(
    (effectSpec) =>
      effectSpec.effect_type === "damage_target" &&
      effectSpec.target_player === "opponent" &&
      Number(effectSpec.selection_count || 0) === 1,
  );
  if (!targetedEffect) {
    return null;
  }
  const damageAmount = targetedEffect.amount ?? attack.damage ?? "";
  const damageLabel = String(damageAmount || "?");
  return {
    attackName: attack.name,
    damageLabel: `${damageLabel} DMG`,
  };
}

function findBoardTargetLabel(player, ref) {
  if (!currentState || !ref) {
    return null;
  }

  const targetPlayer = currentState.players?.[ref.player_index];
  if (!targetPlayer) {
    return null;
  }

  const label = findBoardTargetLabelForPlayer(targetPlayer, ref);
  if (!label) {
    return null;
  }
  return ref.player_index === 0 ? label : `Opponent ${label}`;
}

function makeStatusMessage(state, context) {
  const battleLabel = `${state.players[0].deck_name} vs ${state.ai_trainer?.name || state.players[1].deck_name}`;
  const mulliganAction = findOpeningMulliganAction(state);
  const chooseActiveAction =
    state?.legal_actions?.find((actionView) => actionView.type === "play_basic_to_active") || null;
  if (state.winner === 0) {
    return `${battleLabel} • Game over: you won.`;
  }
  if (state.winner === 1) {
    return `${battleLabel} • Game over: the AI won.`;
  }
  if (aiAutoRunPaused && state.current_player === 1) {
    return `${battleLabel} • AI auto-run is paused. Refresh or restart the backend server, then try again.`;
  }
  if (state.pending_promotion_for === 0) {
    return `${battleLabel} • Promotion required. ${context.instructions}`;
  }
  if (mulliganAction) {
    return `${battleLabel} • No Basic Pokemon in hand.`;
  }
  if (state.game_mode === "standard" && state.setup_phase === "choose_active" && chooseActiveAction) {
    return `${battleLabel} • Choose your Active Pokemon.`;
  }
  if (
    state.game_mode === "standard" &&
    state.setup_phase === "choose_active" &&
    !state.players[0].active &&
    !state.legal_actions.length
  ) {
    return `${battleLabel} • Opening hands are drawn.`;
  }
  if (state.game_mode === "standard" && state.setup_phase === "awaiting_end_setup") {
    return `${battleLabel} • Setup is ready. End setup to begin Turn 1.`;
  }
  if (state.current_player === 0) {
    return `${battleLabel} • Your turn.`;
  }
  return `${battleLabel} • AI turn in progress.`;
}

function updateStatus(message) {
  const statusBanner = document.getElementById("status-banner");
  if (statusBanner) {
    statusBanner.textContent = message;
  }
}

function handlePlayerBenchZoneClick(event) {
  if (!tryAutoSubmitBenchPlay()) {
    return;
  }

  event.preventDefault();
  event.stopPropagation();
}

function handleBoardBackgroundClick(event) {
  if (!currentState || aiIsRunning) {
    return;
  }

  const clickedInteractiveElement = event.target.closest(
    ".mini-card, .board-card, .attack-chip-button, .end-turn-button, .resource-panel, .deck-browser, button, select, input, label",
  );
  if (clickedInteractiveElement) {
    return;
  }

  const clickedBoardSearchArea = !event.target.closest(
    "#player-bench-zone, .hand-tray, .player-side-pocket",
  );
  if (clickedBoardSearchArea && openDeckBrowserForSelection(currentState)) {
    event.preventDefault();
    event.stopPropagation();
    return;
  }

  const clickedBoardPlayArea = !event.target.closest(
    "#player-bench-zone, .hand-tray, .player-side-pocket",
  );
  if (clickedBoardPlayArea) {
    const backgroundPlayAction = findBackgroundPlayAction(currentState);
    if (backgroundPlayAction) {
      event.preventDefault();
      event.stopPropagation();
      submitAction(backgroundPlayAction);
      return;
    }
  }

  if (!uiState.selectedCardId && !uiState.selectedBoardTarget) {
    return;
  }

  resetSelections();
  render(currentState);
}

function syncDevPanelToggleLabel() {
  const devPanel = document.querySelector(".dev-panel");
  const collapseHint = devPanel?.querySelector(".collapse-hint");
  if (!collapseHint) {
    return;
  }

  collapseHint.textContent = devPanel.open ? "Hide" : "Show";
}

function escapeHtml(text) {
  return String(text)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

document.getElementById("new-game-button").addEventListener("click", () => {
  void newGame();
});
document
  .getElementById("player-bench-zone")
  .addEventListener("click", handlePlayerBenchZoneClick, true);
document.getElementById("player-board-panel").addEventListener("click", handleBoardBackgroundClick);
document.getElementById("trainer-select").addEventListener("change", handleTrainerChange);
document.getElementById("deck-select").addEventListener("change", handleDeckChange);
document.getElementById("standard-ai-mode-toggle").addEventListener("change", (event) => {
  void handleStandardAiModeToggle(event);
});
document.querySelector(".dev-panel")?.addEventListener("toggle", syncDevPanelToggleLabel);
window.addEventListener("pointermove", handlePointerMove);
window.addEventListener("pointerleave", handlePointerLeaveWindow);
syncDevPanelToggleLabel();
window.addEventListener("load", refreshGame);

globalThis.__TCG_APP_TEST_API__ = {
  handleStandardAiModeToggle,
  newGame,
  openDeckBrowserForSelection,
  render,
  renderActiveView,
  setCurrentState(state) {
    currentState = state;
  },
  setLobbyState(state) {
    lobbyState = state;
  },
  setSubmitActionOverride(override) {
    submitActionOverride = override;
  },
  uiState,
};
