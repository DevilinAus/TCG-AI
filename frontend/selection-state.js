(function initSelectionHelpers(global) {
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

  function cloneRef(ref) {
    return ref ? { ...ref } : null;
  }

  function handCardCanTargetRef(state, sourceCardId, targetRef) {
    if (!state || !sourceCardId || !targetRef) {
      return false;
    }

    return state.legal_actions.some(
      (action) =>
        action.source?.zone === "hand" &&
        action.source.instance_id === sourceCardId &&
        action.target &&
        refsMatch(action.target, targetRef),
    );
  }

  function boardTargetExists(player, target) {
    if (!player || !target) {
      return false;
    }
    if (target.zone === "active") {
      if (!player.active) {
        return (target.instance_id ?? null) === null;
      }
      return player.active.ref.instance_id === target.instance_id;
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

  function sanitizeSelectionState(uiState, state) {
    const player = state?.players?.[0];
    if (!player) {
      return {
        selectedCardId: null,
        selectedBoardTarget: null,
      };
    }

    const handIds = new Set((player.hand || []).map((card) => card.instance_id));
    const selectedCardId =
      uiState?.selectedCardId && handIds.has(uiState.selectedCardId)
        ? uiState.selectedCardId
        : null;
    const selectedBoardTarget = uiState?.selectedBoardTarget;
    const keepEnergyTargetSelected =
      selectedBoardTarget?.zone === "energy" &&
      !!selectedCardId &&
      state.legal_actions.some(
        (action) =>
          action.source?.zone === "hand" &&
          action.source.instance_id === selectedCardId &&
          action.target &&
          refsMatch(action.target, selectedBoardTarget),
      );

    return {
      selectedCardId,
      selectedBoardTarget:
        selectedBoardTarget &&
        boardTargetExists(player, selectedBoardTarget) &&
        (selectedBoardTarget.zone !== "energy" || keepEnergyTargetSelected)
          ? cloneRef(selectedBoardTarget)
          : null,
    };
  }

  function collectUniqueRefs(refs, collection) {
    for (const ref of refs) {
      if (!collection.some((existing) => refsMatch(existing, ref))) {
        collection.push(ref);
      }
    }
  }

  function findMatchingHandTargetedAction(state, sourceCardId, targetRef, aiIsRunning) {
    if (!state || !sourceCardId || aiIsRunning) {
      return null;
    }

    const matchingActions = state.legal_actions.filter(
      (action) =>
        action.source?.zone === "hand" &&
        action.source.instance_id === sourceCardId &&
        action.target &&
        refsMatch(action.target, targetRef),
    );

    return matchingActions.length === 1 ? matchingActions[0] : null;
  }

  function findMatchingBoardTargetedAction(state, sourceRef, targetRef, aiIsRunning) {
    if (!state || aiIsRunning) {
      return null;
    }

    const matchingActions = state.legal_actions.filter(
      (action) =>
        action.source &&
        action.source.zone !== "hand" &&
        refsMatch(action.source, sourceRef) &&
        action.target &&
        refsMatch(action.target, targetRef),
    );

    return matchingActions.length === 1 ? matchingActions[0] : null;
  }

  function hasSourceAction(state, ref) {
    if (!state || !ref) {
      return false;
    }

    return state.legal_actions.some(
      (action) =>
        action.source &&
        action.source.zone !== "hand" &&
        refsMatch(action.source, ref),
    );
  }

  function resolveSelectedCardClick({ state, uiState, instanceId, aiIsRunning }) {
    const nextSelectedCardId = uiState.selectedCardId === instanceId ? null : instanceId;

    if (nextSelectedCardId && uiState.selectedBoardTarget) {
      const autoAction = findMatchingHandTargetedAction(
        state,
        nextSelectedCardId,
        uiState.selectedBoardTarget,
        aiIsRunning,
      );
      if (autoAction) {
        return {
          autoAction,
          nextUiState: null,
        };
      }
    }

    return {
      autoAction: null,
      nextUiState: {
        selectedCardId: nextSelectedCardId,
        selectedBoardTarget:
          nextSelectedCardId && handCardCanTargetRef(state, nextSelectedCardId, uiState.selectedBoardTarget)
            ? cloneRef(uiState.selectedBoardTarget)
            : null,
      },
    };
  }

  function resolveSelectedBoardTargetClick({ state, uiState, targetRef, aiIsRunning }) {
    const currentTarget = uiState.selectedBoardTarget;
    const nextTarget = currentTarget && refsMatch(currentTarget, targetRef) ? null : cloneRef(targetRef);

    if (uiState.selectedCardId) {
      if (!nextTarget) {
        return {
          autoAction: null,
          nextUiState: {
            selectedCardId: uiState.selectedCardId,
            selectedBoardTarget: null,
          },
        };
      }

      const handAction = findMatchingHandTargetedAction(
        state,
        uiState.selectedCardId,
        nextTarget,
        aiIsRunning,
      );
      if (handAction) {
        return {
          autoAction: handAction,
          nextUiState: null,
        };
      }

      if (hasSourceAction(state, nextTarget)) {
        return {
          autoAction: null,
          nextUiState: {
            selectedCardId: null,
            selectedBoardTarget: cloneRef(nextTarget),
          },
        };
      }

      return {
        autoAction: null,
        nextUiState: {
          selectedCardId: uiState.selectedCardId,
          selectedBoardTarget: cloneRef(currentTarget),
        },
      };
    }

    if (nextTarget && currentTarget && !refsMatch(currentTarget, nextTarget)) {
      const boardAction = findMatchingBoardTargetedAction(
        state,
        currentTarget,
        nextTarget,
        aiIsRunning,
      );
      if (boardAction) {
        return {
          autoAction: boardAction,
          nextUiState: null,
        };
      }
    }

    return {
      autoAction: null,
      nextUiState: {
        selectedCardId: uiState.selectedCardId ?? null,
        selectedBoardTarget: nextTarget,
      },
    };
  }

  function findBenchPlayActionForSelection(state, selectedCardId, aiIsRunning) {
    if (!state || !selectedCardId || aiIsRunning) {
      return null;
    }

    const matchingActions = state.legal_actions.filter(
      (action) =>
        action.type === "bench_basic" &&
        action.source?.zone === "hand" &&
        action.source.instance_id === selectedCardId,
    );

    return matchingActions.length === 1 ? matchingActions[0] : null;
  }

  function findBackgroundPlayActionForSelection(state, selectedCardId, aiIsRunning) {
    if (!state || !selectedCardId || aiIsRunning) {
      return null;
    }

    const matchingActions = state.legal_actions.filter(
      (action) =>
        action.type === "play_supporter" &&
        action.source?.zone === "hand" &&
        action.source.instance_id === selectedCardId &&
        !action.target,
    );

    return matchingActions.length === 1 ? matchingActions[0] : null;
  }

  function isBoardRefClickable({ state, uiState, context, aiIsRunning, ref }) {
    if (!ref || aiIsRunning) {
      return false;
    }

    if (ref.zone === "energy") {
      if (!uiState.selectedCardId) {
        return false;
      }
      return context.highlightedTargets.some((target) => refsMatch(target, ref));
    }

    if (!uiState.selectedCardId) {
      return true;
    }

    if (uiState.selectedBoardTarget && refsMatch(uiState.selectedBoardTarget, ref)) {
      return true;
    }

    if (hasSourceAction(state, ref)) {
      return true;
    }

    return context.highlightedTargets.some((target) => refsMatch(target, ref));
  }

  function deriveInteractionContext(state, uiState) {
    const legalActions = state.legal_actions || [];
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
      const directSourceActions = sourceActions.filter(
        (action) =>
          !action.target ||
          refsMatch(action.target, uiState.selectedBoardTarget) ||
          action.target.player_index !== 0,
      );
      const targetedBoardActions = sourceActions.filter(
        (action) =>
          action.target &&
          action.target.player_index === 0 &&
          !refsMatch(action.target, uiState.selectedBoardTarget),
      );
      const targetActions = legalActions.filter(
        (action) =>
          action.source?.zone === "hand" &&
          action.target &&
          refsMatch(action.target, uiState.selectedBoardTarget),
      );

      actions = directSourceActions;
      const visibleBoardTargets = targetedBoardActions.map((action) => action.target);
      if (visibleBoardTargets.length) {
        instructions.push("This Pokemon is selected. Now choose one of the highlighted board targets.");
        collectUniqueRefs(visibleBoardTargets, highlightedTargets);
      }
      for (const action of targetActions) {
        highlightedHandIds.add(action.source.instance_id);
      }
      if (targetActions.length) {
        instructions.push("This target is valid. Now choose one of the highlighted hand cards.");
      }
      if (!directSourceActions.length && !targetActions.length && !targetedBoardActions.length) {
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

  function findBoardTargetLabelForPlayer(player, ref) {
    if (ref.zone === "active") {
      if (player.active && refsMatch(player.active.ref, ref)) {
        return `Active ${player.active.name}`;
      }
      if (!player.active && (ref.instance_id ?? null) === null) {
        return "Active Spot";
      }
    }
    if (ref.zone === "bench") {
      const pokemon = player.bench[ref.bench_index];
      if (pokemon && refsMatch(pokemon.ref, ref)) {
        return `Bench ${pokemon.name}`;
      }
    }
    if (ref.zone === "energy") {
      return "Shared Energy";
    }
    return null;
  }

  const api = {
    boardTargetExists,
    deriveInteractionContext,
    findBackgroundPlayActionForSelection,
    findBenchPlayActionForSelection,
    findBoardTargetLabelForPlayer,
    isBoardRefClickable,
    refsMatch,
    resolveSelectedBoardTargetClick,
    resolveSelectedCardClick,
    sanitizeSelectionState,
  };

  if (typeof module !== "undefined" && module.exports) {
    module.exports = api;
  }
  global.TcgUiSelection = api;
})(typeof globalThis !== "undefined" ? globalThis : this);
