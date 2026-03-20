const SESSION_STORAGE_KEY = "tcg_ai_session_id";
const AI_HUMAN_DELAY_MIN_MS = 5000;
const AI_HUMAN_DELAY_MAX_MS = 8000;
const FALLBACK_AI_STEP_DELAY_MS = 6500;

let currentState = null;
let lobbyState = null;
let previousState = null;
let aiIsRunning = false;
let aiAutoRunQueued = false;
let aiAutoRunPaused = false;

const uiState = {
  selectedCardId: null,
  selectedBoardTarget: null,
  availableContextActions: [],
  selectedTrainerId: null,
  selectedHumanDeckId: null,
};

const pointerState = {
  clientX: null,
  clientY: null,
};

const {
  deriveInteractionContext,
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
}

function handlePointerLeaveWindow() {
  pointerState.clientX = null;
  pointerState.clientY = null;
  document.body.style.cursor = "";
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

async function refreshGame() {
  const sessionId = getStoredSessionId();
  if (!sessionId) {
    await loadLobby();
    return;
  }

  try {
    currentState = await requestJson(`/api/game?session_id=${encodeURIComponent(sessionId)}`);
    lobbyState = null;
    aiAutoRunPaused = false;
    uiState.selectedTrainerId = uiState.selectedTrainerId || currentState.ai_trainer?.id || null;
    uiState.selectedHumanDeckId = uiState.selectedHumanDeckId || currentState.human_deck_id || null;
    sanitizeSelections(currentState);
    render(currentState);
    maybeRunAiTurn(currentState);
  } catch (error) {
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

async function loadLobby() {
  try {
    lobbyState = await requestJson("/api/lobby");
    currentState = null;
    previousState = null;
    aiAutoRunPaused = false;
    resetSelections();
    uiState.selectedTrainerId = uiState.selectedTrainerId || lobbyState.ai_trainer?.id || null;
    uiState.selectedHumanDeckId = uiState.selectedHumanDeckId || lobbyState.human_deck_id || null;
    renderLobby(lobbyState);
  } catch (error) {
    updateStatus(error.message);
  }
}

async function newGame() {
  try {
    const payloadBody = {};
    const selectionState = currentState || lobbyState;
    const trainerId = resolveTrainerSelection(selectionState);
    if (trainerId) {
      payloadBody.trainer_id = trainerId;
    }
    const humanDeckId = resolveHumanDeckSelection(selectionState);
    if (humanDeckId) {
      payloadBody.human_deck_id = humanDeckId;
    }
    const payload = await requestJson("/api/new-game", {
      method: "POST",
      body: JSON.stringify(payloadBody),
    });
    currentState = payload;
    lobbyState = null;
    previousState = null;
    aiAutoRunPaused = false;
    uiState.selectedTrainerId = payload.ai_trainer?.id || uiState.selectedTrainerId;
    uiState.selectedHumanDeckId = payload.human_deck_id || uiState.selectedHumanDeckId;
    resetSelections();
    setStoredSessionId(payload.session_id);
    render(currentState);
    maybeRunAiTurn(currentState);
  } catch (error) {
    updateStatus(error.message);
  }
}

async function submitAction(actionView) {
  if (!currentState || aiIsRunning) {
    return;
  }

  try {
    aiAutoRunPaused = false;
    if (actionView.type === "attack") {
      resetSelections();
    }
    currentState = await requestJson("/api/action", {
      method: "POST",
      body: JSON.stringify({
        session_id: currentState.session_id,
        action: actionView.action,
      }),
    });
    sanitizeSelections(currentState);
    render(currentState);
    maybeRunAiTurn(currentState);
  } catch (error) {
    updateStatus(error.message);
  }
}

async function runAiTurn() {
  if (!currentState || aiIsRunning) {
    return;
  }

  aiAutoRunPaused = false;
  aiAutoRunQueued = false;
  aiIsRunning = true;
  render(currentState);
  updateStatus("AI is thinking...");
  try {
    await sleep(randomDelay(AI_HUMAN_DELAY_MIN_MS, AI_HUMAN_DELAY_MAX_MS));
    while (currentState && currentState.current_player === 1 && currentState.winner === null) {
      const payload = await requestJson("/api/ai-step", {
        method: "POST",
        body: JSON.stringify({ session_id: currentState.session_id }),
      });
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

      await sleep(step.delay_ms || FALLBACK_AI_STEP_DELAY_MS);
    }
  } catch (error) {
    aiAutoRunPaused = true;
    updateStatus(`AI auto-run stopped: ${error.message} Refresh or restart the backend server, then try again.`);
  } finally {
    aiIsRunning = false;
    if (currentState) {
      render(currentState);
    }
  }
}

async function runAiTurnReplayFallback() {
  if (!currentState) {
    return;
  }

  const payload = await requestJson("/api/ai-turn", {
    method: "POST",
    body: JSON.stringify({ session_id: currentState.session_id }),
  });

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

    await sleep(replayStep.delay_ms || FALLBACK_AI_STEP_DELAY_MS);
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
  uiState.selectedBoardTarget = null;
  uiState.availableContextActions = [];
}

function sanitizeSelections(state) {
  const sanitized = sanitizeSelectionState(uiState, state);
  uiState.selectedCardId = sanitized.selectedCardId;
  uiState.selectedBoardTarget = sanitized.selectedBoardTarget;
}

function render(state) {
  const previousSnapshot = previousState;
  const human = state.players[0];
  const ai = state.players[1];
  const context = deriveContext(state);
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

  renderMatchMeta(state);
  renderTrainerPicker(state);
  renderDeckPicker(state);
  renderOpponentIdentity(state);
  renderTurnHighlights(state);
  renderPlayerEndTurnButton(state);
  const selectionSummaryElement = document.getElementById("selection-summary");
  if (selectionSummaryElement) {
    selectionSummaryElement.textContent = describeSelection(state, context);
  }
  renderPlayerSummary(document.getElementById("player-summary"), human);
  renderPlayerSummary(document.getElementById("opponent-summary"), ai);

  renderEnergyRow(document.getElementById("player-energy"), human.energy_zone, {
    clickable: isBoardRefClickable({
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
  renderEnergyRow(document.getElementById("opponent-energy"), ai.energy_zone, {
    clickable: false,
    selectedTarget: null,
    context,
    targetRef: opponentEnergyTargetRef,
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
    },
  );
  renderPokemonZone(
    document.getElementById("opponent-active"),
    ai.active,
    {
      clickable: false,
      selectedTarget: null,
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
      clickable: false,
      selectedTarget: null,
      context,
      previousBench: previousSnapshot?.players?.[1]?.bench || [],
    },
  );

  renderHand(document.getElementById("player-hand"), human.hand, context, previousSnapshot?.players?.[0]?.hand || []);
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
  syncCursorForPointerPosition();
  if (state.current_player === 1 && state.winner === null) {
    queueAiTurn();
  }
}

function renderLobby(state) {
  renderLobbyMatchMeta();
  renderTrainerPicker(state);
  renderDeckPicker(state);
  renderOpponentIdentity(state);
  renderLobbyBoard(state);
  updateStatus("Choose your deck and gym leader, then start a new game.");
}

function renderEnergyRow(element, energyCards, options) {
  element.innerHTML = "";
  const energyCard = buildEnergySpotCard(energyCards, {
    clickable: options.clickable,
    selected: !!options.selectedTarget && refsMatch(options.selectedTarget, options.targetRef),
    targetable: options.context.highlightedTargets.some((target) =>
      refsMatch(target, options.targetRef),
    ),
  });

  if (options.clickable) {
    energyCard.addEventListener("click", () => toggleSelectedBoardTarget(options.targetRef));
  }
  element.appendChild(energyCard);
}

function renderDiscard(element, discardTop, discardCount) {
  element.innerHTML = "";
  if (!discardTop) {
    element.appendChild(buildPlaceholder("Empty."));
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
  const element = document.createElement("div");
  element.className = "bench-empty-stamp";
  element.textContent = "Bench Empty";
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
      selected: uiState.selectedCardId === card.instance_id,
      targetable: context.highlightedHandIds.has(card.instance_id),
      animationClass: previousIds.has(card.instance_id) ? "" : "card-anim-draw",
    });
    miniCard.addEventListener("click", () => toggleSelectedCard(card.instance_id));
    element.appendChild(miniCard);
  }
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
      const attackAction = findAttackActionForPokemon(pokemon, attackIndex);
      if (attackAction) {
        submitAction(attackAction);
      }
    });
  }

  if (options.clickable) {
    card.addEventListener("click", () => toggleSelectedBoardTarget(pokemon.ref));
  }
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
  element.setAttribute("aria-label", "Active Spot");
  element.innerHTML = `
    <div class="active-slot-placeholder-copy">
      <span class="active-slot-placeholder-label">Active Spot</span>
    </div>
  `;
  if (options.clickable && options.targetRef) {
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

function buildEnergySpotCard(energyCards, options = {}) {
  let element;
  if (energyCards.length) {
    const topCard = energyCards[energyCards.length - 1];
    element = buildMiniCard(topCard, {
      clickable: options.clickable,
      hideAccent: true,
      hideCopy: true,
    });
    element.classList.add("pile-card", "energy-pile-card");
    if (energyCards.length > 1) {
      element.classList.add("is-stacked");
      element.appendChild(buildPileCountBadge(energyCards.length));
    }
  } else {
    element = document.createElement("article");
    const classNames = ["mini-card", "pile-card", "energy-pile-card", "energy-empty-card"];
    if (options.clickable) {
      classNames.push("is-clickable");
    }
    element.className = classNames.join(" ");
    element.innerHTML = `
      <div class="energy-empty-state">Empty</div>
      <div class="card-copy">
        <p class="card-meta">No attachments</p>
      </div>
    `;
  }

  if (options.selected) {
    element.classList.add("is-selected");
  }
  if (options.targetable) {
    element.classList.add("is-targetable");
  }
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

function buildPokemonImageMarkup(pokemon) {
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
      <div class="card-visual card-visual-fallback">
        <div class="placeholder-card">${escapeHtml(pokemon.name)}</div>
        ${overlay}
      </div>
    `;
  }

  return `
    <div class="card-visual">
      <img src="${escapeHtml(pokemon.image_url)}" alt="${escapeHtml(pokemon.name)}" loading="lazy" />
      ${overlay}
    </div>
  `;
}

function renderAttackChip(pokemon, attack, attackIndex, options) {
  const attackAction = findAttackActionForPokemon(pokemon, attackIndex);
  const isPlayersActivePokemon =
    options.clickable &&
    pokemon.ref?.player_index === 0 &&
    pokemon.ref?.zone === "active";
  const tagName = isPlayersActivePokemon ? "button" : "div";
  const classes = ["attack-chip"];
  if (isPlayersActivePokemon) {
    classes.push("attack-chip-button");
    if (attackAction) {
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
    if (!attackAction || aiIsRunning) {
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

function findAttackActionForPokemon(pokemon, attackIndex) {
  if (!currentState || !pokemon?.ref) {
    return null;
  }

  return (
    currentState.legal_actions.find(
      (actionView) =>
        actionView.type === "attack" &&
        actionView.source?.zone === pokemon.ref.zone &&
        actionView.source?.player_index === pokemon.ref.player_index &&
        actionView.source?.instance_id === pokemon.ref.instance_id &&
        actionView.action?.attack_index === attackIndex,
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
    trainerMeta.textContent = "Choose a gym leader for your next battle.";
    trainerProgress.hidden = true;
    return;
  }

  const isCurrentOpponent = state.ai_trainer?.id === selectedTrainer.id;
  const prefix = isCurrentOpponent ? "Current opponent" : "Next battle";
  trainerMeta.textContent =
    `${prefix}: ${selectedTrainer.name} • Lv. ${selectedTrainer.level} • ` +
    `${selectedTrainer.experience} XP • ${selectedTrainer.specialty} specialist`;

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
  renderPlayerSummaryPlaceholder(document.getElementById("player-summary"));
  renderPlayerSummaryPlaceholder(document.getElementById("opponent-summary"));
  renderEnergyPlaceholder(document.getElementById("player-energy"));
  renderEnergyPlaceholder(document.getElementById("opponent-energy"));
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

function renderPlayerSummary(element, player) {
  element.innerHTML = [
    buildMetricPill("Prize", player.prize_tokens_remaining),
    buildMetricPill("Deck", player.deck_count),
    buildMetricPill("Discard", player.discard_count),
    buildMetricPill("Energy", player.energy_count),
  ].join("");
}

function buildMetricPill(label, value) {
  return `
    <div class="metric-pill">
      <span class="metric-pill-label">${escapeHtml(label)}</span>
      <span class="metric-pill-value">${escapeHtml(value)}</span>
    </div>
  `;
}

function renderPlayerSummaryPlaceholder(element) {
  if (!element) {
    return;
  }
  element.innerHTML = [
    buildMetricPill("Prize", "-"),
    buildMetricPill("Deck", "-"),
    buildMetricPill("Discard", "-"),
    buildMetricPill("Energy", "-"),
  ].join("");
}

function renderEnergyPlaceholder(element) {
  if (!element) {
    return;
  }
  element.innerHTML = "";
  element.appendChild(buildPlaceholder("Empty."));
}

function renderPlayerEndTurnButton(state) {
  const button = document.getElementById("player-end-turn-button");
  if (!button) {
    return;
  }

  const endTurnAction =
    state.legal_actions.find((actionView) => actionView.type === "end_turn") || null;
  button.disabled = aiIsRunning || !endTurnAction;
  button.classList.toggle("is-ready", !!endTurnAction && !aiIsRunning);
  button.onclick = endTurnAction ? () => submitAction(endTurnAction) : null;
}

function renderPlayerEndTurnButtonLobby() {
  const button = document.getElementById("player-end-turn-button");
  if (!button) {
    return;
  }

  button.disabled = true;
  button.classList.remove("is-ready");
  button.onclick = null;
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

function classifyAction(actionView) {
  if (actionView.type === "attack") {
    return "is-attack";
  }
  if (actionView.type === "end_turn" || actionView.type === "promote") {
    return "is-primary";
  }
  return "is-support";
}

function describeActionType(actionType) {
  const labels = {
    attack: "Attack",
    bench_basic: "Bench play",
    end_turn: "Turn action",
    evolve: "Evolution",
    play_energy: "Energy attach",
    play_potion: "Support effect",
    play_switch: "Switch",
    promote: "Required choice",
  };
  return labels[actionType] || actionType.replaceAll("_", " ");
}

function formatStageLabel(stage) {
  if (!stage || stage === "basic") {
    return "Basic Pokemon";
  }
  if (stage === "stage1") {
    return "Stage 1 Pokemon";
  }
  return stage.replaceAll("_", " ");
}

function formatStagePillLabel(stage) {
  if (!stage || stage === "basic") {
    return "Basic";
  }
  if (stage === "stage1") {
    return "Stage 1";
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
    submitAction(result.autoAction);
    return;
  }

  uiState.selectedCardId = result.nextUiState.selectedCardId;
  uiState.selectedBoardTarget = result.nextUiState.selectedBoardTarget;
  render(currentState);
}

function toggleSelectedBoardTarget(targetRef) {
  const result = resolveSelectedBoardTargetClick({
    state: currentState,
    uiState,
    targetRef,
    aiIsRunning,
  });

  if (result.autoAction) {
    submitAction(result.autoAction);
    return;
  }

  uiState.selectedCardId = result.nextUiState.selectedCardId;
  uiState.selectedBoardTarget = result.nextUiState.selectedBoardTarget;
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
  if (!fragments.length) {
    if (context.actions.length) {
      return "System actions ready.";
    }
    return "No selection yet.";
  }
  return fragments.join(" • ");
}

function findBoardTargetLabel(player, ref) {
  return findBoardTargetLabelForPlayer(player, ref);
}

function makeStatusMessage(state, context) {
  const battleLabel = `${state.players[0].deck_name} vs ${state.ai_trainer?.name || state.players[1].deck_name}`;
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
  if (state.current_player === 0) {
    return `${battleLabel} • Your turn.`;
  }
  return `${battleLabel} • AI turn in progress.`;
}

function updateStatus(message) {
  document.getElementById("status-banner").textContent = message;
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
    ".mini-card, .board-card, .attack-chip-button, .end-turn-button, .resource-panel, button, select, input, label",
  );
  if (clickedInteractiveElement) {
    return;
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

document.getElementById("new-game-button").addEventListener("click", newGame);
document
  .getElementById("player-bench-zone")
  .addEventListener("click", handlePlayerBenchZoneClick, true);
document.getElementById("player-board-panel").addEventListener("click", handleBoardBackgroundClick);
document.getElementById("trainer-select").addEventListener("change", handleTrainerChange);
document.getElementById("deck-select").addEventListener("change", handleDeckChange);
document.querySelector(".dev-panel")?.addEventListener("toggle", syncDevPanelToggleLabel);
window.addEventListener("pointermove", handlePointerMove);
window.addEventListener("pointerleave", handlePointerLeaveWindow);
syncDevPanelToggleLabel();
window.addEventListener("load", refreshGame);
