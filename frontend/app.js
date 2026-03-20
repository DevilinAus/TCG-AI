const SESSION_STORAGE_KEY = "tcg_ai_session_id";
const AI_REPLAY_THINK_DELAY_MS = 450;
const FALLBACK_AI_STEP_DELAY_MS = 900;

let currentState = null;
let previousState = null;
let aiIsRunning = false;

const uiState = {
  selectedCardId: null,
  selectedBoardTarget: null,
  availableContextActions: [],
};

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

function getStoredSessionId() {
  return window.localStorage.getItem(SESSION_STORAGE_KEY);
}

function setStoredSessionId(sessionId) {
  window.localStorage.setItem(SESSION_STORAGE_KEY, sessionId);
}

function clearStoredSessionId() {
  window.localStorage.removeItem(SESSION_STORAGE_KEY);
}

async function refreshGame() {
  const sessionId = getStoredSessionId();
  if (!sessionId) {
    await newGame();
    return;
  }

  try {
    currentState = await requestJson(`/api/game?session_id=${encodeURIComponent(sessionId)}`);
    sanitizeSelections(currentState);
    render(currentState);
    maybeRunAiTurn(currentState);
  } catch (error) {
    if (error.code === "session_not_found" || error.code === "missing_session_id") {
      clearStoredSessionId();
      await newGame();
      return;
    }
    updateStatus(error.message);
  }
}

async function newGame() {
  try {
    const payload = await requestJson("/api/new-game", {
      method: "POST",
      body: JSON.stringify({ human_first: true }),
    });
    currentState = payload;
    previousState = null;
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

  aiIsRunning = true;
  updateStatus("AI is thinking...");
  try {
    await sleep(AI_REPLAY_THINK_DELAY_MS);
    while (currentState && currentState.current_player === 1 && currentState.winner === null) {
      const payload = await requestJson("/api/ai-step", {
        method: "POST",
        body: JSON.stringify({ session_id: currentState.session_id }),
      });
      const step = payload.ai_step;
      currentState = payload;
      sanitizeSelections(currentState);
      render(currentState);

      if (!step?.action) {
        break;
      }

      updateStatus(buildAiReplayStatus(step));

      if (currentState.current_player === 1 && currentState.winner === null) {
        await sleep(step.delay_ms || FALLBACK_AI_STEP_DELAY_MS);
      }
    }
  } catch (error) {
    updateStatus(error.message);
  } finally {
    aiIsRunning = false;
    if (currentState) {
      render(currentState);
    }
  }
}

function buildAiReplayStatus(step) {
  const label = step.action?.label || "AI takes an action";
  return `AI action: ${label}`;
}

function maybeRunAiTurn(state) {
  if (state.winner !== null) {
    return;
  }
  if (state.current_player === 1) {
    window.setTimeout(runAiTurn, 400);
  }
}

function resetSelections() {
  uiState.selectedCardId = null;
  uiState.selectedBoardTarget = null;
  uiState.availableContextActions = [];
}

function sanitizeSelections(state) {
  const player = state.players[0];
  const handIds = new Set(player.hand.map((card) => card.instance_id));
  if (uiState.selectedCardId && !handIds.has(uiState.selectedCardId)) {
    uiState.selectedCardId = null;
  }

  if (uiState.selectedBoardTarget && !boardTargetExists(player, uiState.selectedBoardTarget)) {
    uiState.selectedBoardTarget = null;
  }
}

function boardTargetExists(player, target) {
  if (target.zone === "active") {
    return !!player.active && player.active.ref.instance_id === target.instance_id;
  }
  if (target.zone === "bench") {
    const pokemon = player.bench[target.bench_index];
    return !!pokemon && pokemon.ref.instance_id === target.instance_id;
  }
  if (target.zone === "energy") {
    return target.player_index === 0;
  }
  return false;
}

function render(state) {
  const previousSnapshot = previousState;
  const human = state.players[0];
  const ai = state.players[1];
  const context = deriveContext(state);
  uiState.availableContextActions = context.actions;

  renderMatchMeta(state);
  document.getElementById("selection-summary").textContent = describeSelection(state, context);
  renderPlayerSummary(document.getElementById("player-summary"), human);
  renderPlayerSummary(document.getElementById("opponent-summary"), ai);

  renderEnergyRow(document.getElementById("player-energy"), human.energy_zone, {
    clickable: true,
    selectedTarget: uiState.selectedBoardTarget,
    context,
    targetRef: { player_index: 0, zone: "energy", bench_index: null, instance_id: null },
  });
  renderEnergyRow(document.getElementById("opponent-energy"), ai.energy_zone, {
    clickable: false,
    selectedTarget: null,
    context,
    targetRef: { player_index: 1, zone: "energy", bench_index: null, instance_id: null },
  });
  renderDiscard(document.getElementById("player-discard"), human.discard_top, human.discard_count);
  renderDiscard(document.getElementById("opponent-discard"), ai.discard_top, ai.discard_count);

  renderPokemonZone(
    document.getElementById("player-active"),
    human.active,
    {
      clickable: true,
      selectedTarget: uiState.selectedBoardTarget,
      context,
      previousPokemon: previousSnapshot?.players?.[0]?.active || null,
    },
  );
  renderPokemonZone(
    document.getElementById("opponent-active"),
    ai.active,
    {
      clickable: false,
      compact: true,
      selectedTarget: null,
      context,
      previousPokemon: previousSnapshot?.players?.[1]?.active || null,
    },
  );

  renderPokemonList(
    document.getElementById("player-bench"),
    human.bench,
    {
      clickable: true,
      selectedTarget: uiState.selectedBoardTarget,
      context,
      previousBench: previousSnapshot?.players?.[0]?.bench || [],
    },
  );
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
  renderContextActions(document.getElementById("context-actions"), context.actions);
  renderDebugActions(document.getElementById("debug-actions"), state.legal_actions);
  renderLog(document.getElementById("battle-log"), state.log, previousSnapshot?.log || []);

  document.getElementById("context-instructions").textContent = context.instructions;
  document.getElementById("new-game-button").disabled = aiIsRunning;
  document.getElementById("clear-selection-button").disabled =
    aiIsRunning || (!uiState.selectedCardId && !uiState.selectedBoardTarget);
  document.getElementById("ai-turn-button").disabled =
    aiIsRunning || state.current_player !== 1 || state.winner !== null;

  updateStatus(makeStatusMessage(state, context));
  previousState = JSON.parse(JSON.stringify(state));
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
    element.appendChild(buildPlaceholder("Discard pile empty."));
    return;
  }

  const card = buildMiniCard(discardTop);
  const meta = card.querySelector(".card-meta");
  meta.textContent = `Top of discard • ${discardCount} total`;
  element.appendChild(card);
}

function renderPokemonZone(element, pokemon, options) {
  element.innerHTML = "";
  element.appendChild(buildPokemonCard(pokemon, options));
}

function renderPokemonList(element, pokemonList, options) {
  element.innerHTML = "";
  if (!pokemonList.length) {
    element.appendChild(buildPlaceholder("No benched Pokemon."));
    return;
  }

  pokemonList.forEach((pokemon, index) => {
    const previousPokemon = options.previousBench[index] || null;
    element.appendChild(
      buildPokemonCard(pokemon, {
        ...options,
        compact: true,
        previousPokemon,
      }),
    );
  });
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

function buildPokemonCard(pokemon, options) {
  if (!pokemon) {
    return buildPlaceholder("No Pokemon in this slot.");
  }

  const card = document.createElement("article");
  const classNames = ["board-card"];
  const isCompact = !!options.compact;
  if (options.clickable) {
    classNames.push("is-clickable");
  }
  if (isCompact) {
    classNames.push("is-compact");
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
  const hpPercent = pokemon.hp ? Math.max(0, Math.round((pokemon.remaining_hp / pokemon.hp) * 100)) : 0;
  const healthClass = hpPercent <= 33 ? "is-danger" : hpPercent <= 66 ? "is-warning" : "";
  const evolutionLine = pokemon.stack.map((cardItem) => cardItem.name).join(" -> ");
  const primaryStageLabel = isCompact
    ? `HP ${pokemon.remaining_hp}/${pokemon.hp}`
    : `${formatStageLabel(pokemon.stage)} • ${formatElementLabel(pokemon.element)}`;
  const compactTags = isCompact ? describeCompactPokemonTags(pokemon) : "";
  const compactStatPills = [];
  if (pokemon.damage > 0) {
    compactStatPills.push(`<span class="info-pill">Damage ${pokemon.damage}</span>`);
  }
  if (pokemon.can_attack) {
    compactStatPills.push('<span class="info-pill is-accent">Attack ready</span>');
  }
  if (pokemon.requires_promotion) {
    compactStatPills.push('<span class="info-pill">Promote</span>');
  }
  if (!compactStatPills.length && isCompact) {
    compactStatPills.push('<span class="info-pill">Benched</span>');
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
          ${pokemon.can_attack ? '<span class="info-pill is-accent">Attack ready</span>' : ""}
        `}
      </div>
      ${isCompact ? "" : `
        <div class="attack-list">
          ${pokemon.attacks.map(renderAttackChip).join("")}
        </div>
        <p class="card-meta card-stack">Evolution line: ${escapeHtml(evolutionLine)}</p>
      `}
      ${isCompact && compactTags === "Benched" ? "" : `<p class="card-tags">${escapeHtml(isCompact ? compactTags : describePokemonTags(pokemon))}</p>`}
    </div>
  `;

  if (options.clickable) {
    card.addEventListener("click", () => toggleSelectedBoardTarget(pokemon.ref));
  }
  return card;
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
  const typeLabel = formatCardAccentLabel(card);
  const meta = [formatKindLabel(card.kind), card.stage ? formatStagePillLabel(card.stage) : ""]
    .filter(Boolean)
    .join(" • ");
  element.innerHTML = `
    <div class="mini-card-media">
      ${buildCardImageMarkup(card.image_url, card.name)}
      ${options.playable ? '<div class="mini-card-badge">Ready</div>' : ""}
    </div>
    <div class="card-copy">
      <div class="card-title-row">
        <div class="card-title-group">
          <div class="card-title">${escapeHtml(card.name)}</div>
          <p class="card-meta">${escapeHtml(meta)}</p>
        </div>
        <div class="type-pill">${escapeHtml(typeLabel)}</div>
      </div>
    </div>
  `;
  return element;
}

function buildEnergySpotCard(energyCards, options = {}) {
  let element;
  if (energyCards.length) {
    const topCard = energyCards[energyCards.length - 1];
    element = buildMiniCard(topCard, { clickable: options.clickable });
    element.classList.add("energy-pile-card");
    if (energyCards.length > 1) {
      element.classList.add("is-stacked");
      const badge = document.createElement("div");
      badge.className = "energy-count-badge";
      badge.textContent = String(energyCards.length);
      element.appendChild(badge);
    }

    const meta = element.querySelector(".card-meta");
    if (meta) {
      meta.textContent =
        energyCards.length === 1 ? "1 energy in play" : `${energyCards.length} energy in play`;
    }
  } else {
    element = document.createElement("article");
    const classNames = ["mini-card", "energy-pile-card", "energy-empty-card"];
    if (options.clickable) {
      classNames.push("is-clickable");
    }
    element.className = classNames.join(" ");
    element.innerHTML = `
      <div class="energy-empty-state">Energy spot empty.</div>
      <div class="card-copy">
        <div class="card-title">Energy Spot</div>
        <p class="card-meta">0 energy in play</p>
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
  return `<img src="${escapeHtml(imageUrl)}" alt="${escapeHtml(altText)}" loading="lazy" />`;
}

function buildPokemonImageMarkup(pokemon) {
  const overlay =
    pokemon.remaining_hp !== pokemon.hp
      ? `
        <div class="card-hp-badge">
          <span class="card-hp-label">HP</span>
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

function renderAttackChip(attack) {
  const effect =
    attack.effect && attack.effect !== "none"
      ? `<span class="attack-effect">${escapeHtml(formatEffectLabel(attack.effect))}</span>`
      : "";
  return `
    <div class="attack-chip">
      <div class="attack-topline">
        <span class="attack-name">${escapeHtml(attack.name)}</span>
        <span class="attack-stats">Cost ${attack.cost} • Damage ${attack.damage}</span>
      </div>
      ${effect}
    </div>
  `;
}

function describePokemonTags(pokemon) {
  const tags = [];
  if (pokemon.can_attack) {
    tags.push("Attack ready");
  }
  if (pokemon.requires_promotion) {
    tags.push("Must promote");
  }
  if (pokemon.target_action_types.length) {
    tags.push(`Targeted by ${pokemon.target_action_types.join(", ")}`);
  }
  if (pokemon.source_action_types.length && !pokemon.can_attack && !pokemon.requires_promotion) {
    tags.push(`Can ${pokemon.source_action_types.join(", ")}`);
  }
  return tags.length ? tags.join(" • ") : "No immediate action";
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
  uiState.selectedCardId = uiState.selectedCardId === instanceId ? null : instanceId;
  render(currentState);
}

function toggleSelectedBoardTarget(targetRef) {
  const nextTarget =
    uiState.selectedBoardTarget && refsMatch(uiState.selectedBoardTarget, targetRef)
      ? null
      : { ...targetRef };

  if (nextTarget && tryAutoSubmitTargetedAction(nextTarget)) {
    return;
  }

  uiState.selectedBoardTarget = nextTarget;
  render(currentState);
}

function tryAutoSubmitTargetedAction(targetRef) {
  if (!currentState || !uiState.selectedCardId) {
    return false;
  }

  const matchingActions = currentState.legal_actions.filter(
    (action) =>
      action.source?.zone === "hand" &&
      action.source.instance_id === uiState.selectedCardId &&
      action.target &&
      refsMatch(action.target, targetRef),
  );

  if (matchingActions.length === 1 && matchingActions[0].type === "play_energy") {
    submitAction(matchingActions[0]);
    return true;
  }

  return false;
}

function clearSelectionsAndRender() {
  resetSelections();
  if (currentState) {
    render(currentState);
  }
}

function deriveContext(state) {
  const legalActions = state.legal_actions;
  const systemActions = legalActions.filter((action) => action.source?.zone === "system");
  const highlightedTargets = [];
  const highlightedHandIds = new Set();
  const instructions = [];
  let actions = [];

  if (state.winner !== null) {
    return {
      actions: [],
      instructions: "The game is over. Start a new game whenever you want to play again.",
      highlightedTargets,
      highlightedHandIds,
    };
  }

  if (state.pending_promotion_for === 0 && !uiState.selectedBoardTarget) {
    instructions.push("Choose one of your benched Pokemon to become your new Active Pokemon.");
  }

  if (uiState.selectedCardId) {
    const fromSelectedCard = legalActions.filter(
      (action) =>
        action.source?.zone === "hand" &&
        action.source.instance_id === uiState.selectedCardId,
    );
    const directActions = fromSelectedCard.filter((action) => !action.target);
    const targetedActions = fromSelectedCard.filter((action) => !!action.target);

    if (targetedActions.length && !uiState.selectedBoardTarget) {
      instructions.push("Choose a highlighted target on your board to finish this play.");
      collectUniqueRefs(targetedActions.map((action) => action.target), highlightedTargets);
      actions = directActions;
    } else if (targetedActions.length && uiState.selectedBoardTarget) {
      actions = fromSelectedCard.filter((action) =>
        refsMatch(action.target, uiState.selectedBoardTarget),
      );
      if (!actions.length) {
        instructions.push("That target does not work with the selected hand card.");
        collectUniqueRefs(targetedActions.map((action) => action.target), highlightedTargets);
      }
    } else {
      actions = fromSelectedCard;
    }
  } else if (uiState.selectedBoardTarget) {
    const sourceActions = legalActions.filter(
      (action) =>
        action.source &&
        action.source.zone !== "hand" &&
        refsMatch(action.source, uiState.selectedBoardTarget),
    );
    const targetActions = legalActions.filter(
      (action) =>
        action.source?.zone === "hand" &&
        action.target &&
        refsMatch(action.target, uiState.selectedBoardTarget),
    );

    actions = sourceActions;
    for (const action of targetActions) {
      highlightedHandIds.add(action.source.instance_id);
    }
    if (targetActions.length) {
      instructions.push("This target is valid. Now choose one of the highlighted hand cards.");
    }
    if (!sourceActions.length && !targetActions.length) {
      instructions.push("No legal actions are tied to this board card right now.");
    }
  } else {
    actions = systemActions;
    instructions.push("Click a card in your hand to begin a play, or click your Active Pokemon to attack.");
  }

  if (!actions.length && !instructions.length) {
    instructions.push("No action is ready yet. Try selecting a hand card or your Active Pokemon.");
  }

  return {
    actions,
    instructions: instructions.join(" "),
    highlightedTargets,
    highlightedHandIds,
  };
}

function collectUniqueRefs(refs, collection) {
  for (const ref of refs) {
    if (!collection.some((existing) => refsMatch(existing, ref))) {
      collection.push(ref);
    }
  }
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
  if (ref.zone === "active" && player.active && refsMatch(player.active.ref, ref)) {
    return `Active ${player.active.name}`;
  }
  if (ref.zone === "bench") {
    const pokemon = player.bench[ref.bench_index];
    if (pokemon && refsMatch(pokemon.ref, ref)) {
      return `Bench ${pokemon.name}`;
    }
  }
  if (ref.zone === "energy") {
    return "Energy Spot";
  }
  return null;
}

function refsMatch(left, right) {
  if (!left || !right) {
    return false;
  }
  return (
    left.player_index === right.player_index &&
    left.zone === right.zone &&
    (left.bench_index ?? null) === (right.bench_index ?? null) &&
    (left.instance_id ?? null) === (right.instance_id ?? null)
  );
}

function makeStatusMessage(state, context) {
  if (state.winner === 0) {
    return `${state.matchup_label} • Game over: you won.`;
  }
  if (state.winner === 1) {
    return `${state.matchup_label} • Game over: the AI won.`;
  }
  if (state.pending_promotion_for === 0) {
    return `${state.matchup_label} • Promotion required. ${context.instructions}`;
  }
  if (state.current_player === 0) {
    return `${state.matchup_label} • Your turn. ${context.instructions}`;
  }
  return `${state.matchup_label} • AI turn in progress.`;
}

function updateStatus(message) {
  document.getElementById("status-banner").textContent = message;
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
document.getElementById("ai-turn-button").addEventListener("click", runAiTurn);
document
  .getElementById("clear-selection-button")
  .addEventListener("click", clearSelectionsAndRender);
window.addEventListener("load", refreshGame);
