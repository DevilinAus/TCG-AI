const POLL_INTERVAL_MS = 2000;
const STATUS_URL = "/api/standard-self-play/status";

const overviewGrid = document.getElementById("overview-grid");
const throughputChart = document.getElementById("throughput-chart");
const throughputBreakdown = document.getElementById("throughput-breakdown");
const leaderboard = document.getElementById("leaderboard");
const workersGrid = document.getElementById("workers-grid");
const runIdEl = document.getElementById("run-id");
const lastRefreshEl = document.getElementById("last-refresh");
const leaderboardSummaryEl = document.getElementById("leaderboard-summary");
const workerSummaryEl = document.getElementById("worker-summary");

let pollTimer = null;

async function fetchStatus() {
  const response = await fetch(STATUS_URL, { cache: "no-store" });
  if (!response.ok) {
    throw new Error(`Status request failed: ${response.status}`);
  }
  return response.json();
}

function startPolling() {
  const run = async () => {
    try {
      const status = await fetchStatus();
      renderDashboard(status);
    } catch (error) {
      renderError(error);
    } finally {
      pollTimer = window.setTimeout(run, POLL_INTERVAL_MS);
    }
  };
  run();
}

function renderDashboard(status) {
  const workers = Object.values(status.workers ?? {});
  const machines = aggregateMachines(workers);
  const onlineMachines = machines.filter((machine) => machine.is_online);
  const busyMachines = machines.filter((machine) => machine.status === "busy");
  const staleMachines = machines.filter((machine) => machine.status === "stalled" || machine.status === "offline");
  const reported = status.reported ?? {};
  const aggregate = status.aggregate ?? {};
  const throughput = status.throughput ?? {};

  runIdEl.textContent = status.run_id ?? "unknown";
  lastRefreshEl.textContent = `Updated ${formatClock(new Date())}`;

  renderOverview([
    metricCard("Machines Online", `${onlineMachines.length}/${machines.length}`, `${busyMachines.length} busy · ${staleMachines.length} dropped/stale`),
    metricCard("Reported Games", formatNumber(reported.games ?? 0), `${formatNumber(aggregate.games ?? 0)} committed to shards`),
    metricCard("Throughput", `${formatDecimal(throughput.games_per_minute_5m ?? 0)} gpm`, `${formatDecimal(throughput.games_per_minute_overall ?? 0)} overall`),
    metricCard("Samples", formatNumber(reported.samples ?? 0), `${formatDecimal(throughput.samples_per_minute_5m ?? 0)} samples/min`),
    metricCard("ETA", formatDuration(throughput.eta_seconds), `Target ${formatNumber(status.run_config?.games ?? status.total_games_target ?? 0) || "run config"}`),
    metricCard("Win Split", `${formatNumber(reported.deck_wins?.["ampharos-ex-battle-deck"] ?? 0)} A`, `${formatNumber(reported.deck_wins?.["lucario-ex-battle-deck"] ?? 0)} L`),
  ]);

  renderThroughputChart(status.throughput_series ?? []);
  renderThroughputBreakdown(throughput);
  renderLeaderboard(machines);
  renderWorkers(machines);
  leaderboardSummaryEl.textContent = machines.length
    ? `${busyMachines.length} busy · ${onlineMachines.length} online · ${staleMachines.length} stale/offline`
    : "No workers yet";
  workerSummaryEl.textContent = machines.length
    ? `${machines.length} machines · ${workers.length} worker processes · ${formatDecimal(throughput.games_per_minute_5m ?? 0)} games/min over the last 5 minutes`
    : "No workers connected";
}

function renderOverview(cards) {
  overviewGrid.innerHTML = cards.join("");
}

function renderThroughputChart(series) {
  if (!series.length) {
    throughputChart.innerHTML = `<div class="empty-state">No throughput data yet</div>`;
    return;
  }
  const maxGames = Math.max(...series.map((entry) => entry.games), 1);
  throughputChart.innerHTML = series.map((entry) => {
    const height = Math.max((entry.games / maxGames) * 100, entry.games > 0 ? 6 : 1.5);
    const title = `${entry.minute}: ${entry.games} games`;
    return `<div class="chart-bar" style="height:${height}%" title="${escapeHtml(title)}"></div>`;
  }).join("");
}

function renderThroughputBreakdown(throughput) {
  throughputBreakdown.innerHTML = [
    miniMetric("1m GPM", formatDecimal(throughput.games_per_minute_1m ?? 0)),
    miniMetric("5m GPM", formatDecimal(throughput.games_per_minute_5m ?? 0)),
    miniMetric("Actions / min", formatDecimal(throughput.actions_per_minute_5m ?? 0)),
    miniMetric("Avg Game", formatDuration(throughput.average_game_duration_seconds_5m)),
  ].join("");
}

function renderLeaderboard(machines) {
  if (!machines.length) {
    leaderboard.innerHTML = `<div class="empty-state">Machines will appear here once workers lease chunks.</div>`;
    return;
  }
  const rows = [...machines].sort((left, right) => {
    if (right.completed_games !== left.completed_games) {
      return right.completed_games - left.completed_games;
    }
    return (right.active_workers ?? 0) - (left.active_workers ?? 0);
  });
  leaderboard.innerHTML = rows.map((worker, index) => `
    <div class="leaderboard-row">
      <div>
        <span class="leaderboard-rank">${index + 1}</span>
        <strong>${escapeHtml(worker.machine_name)}</strong>
      </div>
      <div>${formatNumber(worker.completed_games ?? 0)} games</div>
      <div>${formatDecimal(worker.recent_games_per_minute_5m ?? 0)} gpm</div>
      <div>${formatNumber(worker.active_workers ?? 0)} workers</div>
    </div>
  `).join("");
}

function renderWorkers(machines) {
  if (!machines.length) {
    workersGrid.innerHTML = `<div class="empty-state">No machines have connected yet. Start a worker and it will show up here automatically.</div>`;
    return;
  }
  workersGrid.innerHTML = [...machines].sort((left, right) => {
    const statusOrder = { busy: 0, idle: 1, stalled: 2, offline: 3 };
    return (statusOrder[left.status] ?? 9) - (statusOrder[right.status] ?? 9);
  }).map((worker) => renderWorkerCard(worker)).join("");
}

function renderWorkerCard(worker) {
  const progressPct = Math.round((worker.current_task_progress ?? 0) * 100);
  const onlineLabel = worker.status === "busy" ? "Busy" : worker.status === "idle" ? "Idle" : worker.status === "stalled" ? "Stalled" : "Offline";
  return `
    <article class="worker-card">
      <div class="worker-head">
        <div>
          <h3 class="worker-title">${escapeHtml(worker.machine_name)}</h3>
          <p class="worker-platform">${escapeHtml(worker.hostname ?? "unknown host")} · ${escapeHtml(shortPlatform(worker.platform))} · ${formatNumber(worker.active_workers ?? 0)} workers</p>
        </div>
        <span class="status-pill status-${escapeHtml(worker.status)}">${onlineLabel}</span>
      </div>

      <div class="worker-progress" title="Current chunk progress">
        <div class="worker-progress-bar" style="width:${progressPct}%"></div>
      </div>

      <div class="worker-meta">
        ${worker.active_shards > 0
          ? `${formatNumber(worker.active_shards)} active shards · ${worker.current_task_completed_games}/${worker.current_task_game_count} games`
          : "No chunk currently leased"}
      </div>

      <div class="worker-stats">
        <div class="worker-stat">
          <div class="metric-label">Games Completed</div>
          <div class="mini-metric-value">${formatNumber(worker.completed_games ?? 0)}</div>
        </div>
        <div class="worker-stat">
          <div class="metric-label">Recent Pace</div>
          <div class="mini-metric-value">${formatDecimal(worker.recent_games_per_minute_5m ?? 0)} gpm</div>
        </div>
        <div class="worker-stat">
          <div class="metric-label">Avg Game Size</div>
          <div class="mini-metric-value">${formatDecimal(worker.average_actions_per_game ?? 0)} actions</div>
        </div>
        <div class="worker-stat">
          <div class="metric-label">Win Split</div>
          <div class="mini-metric-value">${formatNumber(worker.deck_wins?.["ampharos-ex-battle-deck"] ?? 0)} / ${formatNumber(worker.deck_wins?.["lucario-ex-battle-deck"] ?? 0)}</div>
        </div>
      </div>

      <div class="tiny-chart">
        ${(worker.throughput_series ?? []).map((entry) => {
          const maxGames = Math.max(...(worker.throughput_series ?? []).map((item) => item.games), 1);
          const height = Math.max((entry.games / maxGames) * 100, entry.games > 0 ? 10 : 4);
          return `<div class="tiny-bar" style="height:${height}%"></div>`;
        }).join("")}
      </div>

      <div class="worker-footer">
        <span>Seen ${formatRelativeSeconds(worker.seconds_since_seen)}</span>
        <span>${formatLastProgress(worker)}</span>
      </div>
    </article>
  `;
}

function aggregateMachines(workers) {
  const machinesByName = new Map();

  for (const worker of workers) {
    const machineName = worker.machine_name ?? worker.hostname ?? worker.worker_id ?? "unknown-machine";
    const machine = machinesByName.get(machineName) ?? createMachineAggregate(machineName, worker);
    mergeWorkerIntoMachine(machine, worker);
    machinesByName.set(machineName, machine);
  }

  return [...machinesByName.values()];
}

function createMachineAggregate(machineName, worker) {
  return {
    machine_name: machineName,
    hostname: worker.hostname ?? machineName,
    platform: worker.platform ?? null,
    status: "offline",
    is_online: false,
    active_workers: 0,
    active_shards: 0,
    completed_games: 0,
    completed_actions: 0,
    completed_turns: 0,
    completed_samples: 0,
    truncated_games: 0,
    deck_wins: {
      "ampharos-ex-battle-deck": 0,
      "lucario-ex-battle-deck": 0,
    },
    current_task_game_count: 0,
    current_task_completed_games: 0,
    current_task_progress: 0,
    last_duration_seconds: null,
    seconds_since_seen: null,
    recent_games_per_minute_5m: 0,
    average_actions_per_game: 0,
    throughput_series: [],
    _seriesMap: new Map(),
    _lastProgressRank: Number.POSITIVE_INFINITY,
    _lastProgressValue: null,
  };
}

function mergeWorkerIntoMachine(machine, worker) {
  machine.completed_games += worker.completed_games ?? 0;
  machine.completed_actions += worker.completed_actions ?? 0;
  machine.completed_turns += worker.completed_turns ?? 0;
  machine.completed_samples += worker.completed_samples ?? 0;
  machine.truncated_games += worker.truncated_games ?? 0;
  machine.current_task_game_count += worker.current_task_game_count ?? 0;
  machine.current_task_completed_games += worker.current_task_completed_games ?? 0;
  machine.recent_games_per_minute_5m += worker.recent_games_per_minute_5m ?? 0;
  machine.active_workers += 1;

  if (worker.leased_task_index !== null && worker.leased_task_index !== undefined) {
    machine.active_shards += 1;
  }

  if (worker.is_online) {
    machine.is_online = true;
  }

  machine.status = mergeMachineStatus(machine.status, worker.status);

  for (const [deckId, wins] of Object.entries(worker.deck_wins ?? {})) {
    machine.deck_wins[deckId] = (machine.deck_wins[deckId] ?? 0) + Number(wins ?? 0);
  }

  if (machine.seconds_since_seen === null || (worker.seconds_since_seen ?? Number.POSITIVE_INFINITY) < machine.seconds_since_seen) {
    machine.seconds_since_seen = worker.seconds_since_seen ?? null;
  }

  const workerProgressRank = worker.seconds_since_progress ?? Number.POSITIVE_INFINITY;
  if (workerProgressRank < machine._lastProgressRank) {
    machine._lastProgressRank = workerProgressRank;
    machine._lastProgressValue = worker.last_duration_seconds ?? null;
    machine.hostname = worker.hostname ?? machine.hostname;
    machine.platform = worker.platform ?? machine.platform;
  }

  if (worker.throughput_series?.length) {
    for (const entry of worker.throughput_series) {
      const existing = machine._seriesMap.get(entry.minute) ?? { ...entry };
      existing.games = (existing.games ?? 0) + Number(entry.games ?? 0);
      existing.actions = (existing.actions ?? 0) + Number(entry.actions ?? 0);
      existing.turns = (existing.turns ?? 0) + Number(entry.turns ?? 0);
      existing.samples = (existing.samples ?? 0) + Number(entry.samples ?? 0);
      existing.duration_seconds = (existing.duration_seconds ?? 0) + Number(entry.duration_seconds ?? 0);
      machine._seriesMap.set(entry.minute, existing);
    }
  }

  machine.average_actions_per_game = machine.completed_games
    ? machine.completed_actions / machine.completed_games
    : 0;
  machine.current_task_progress = machine.current_task_game_count
    ? machine.current_task_completed_games / machine.current_task_game_count
    : 0;
  machine.last_duration_seconds = machine._lastProgressValue;
  machine.throughput_series = [...machine._seriesMap.values()].sort((left, right) => left.minute.localeCompare(right.minute));
}

function mergeMachineStatus(currentStatus, workerStatus) {
  const statusOrder = { busy: 0, idle: 1, stalled: 2, offline: 3 };
  return (statusOrder[workerStatus] ?? 9) < (statusOrder[currentStatus] ?? 9)
    ? workerStatus
    : currentStatus;
}

function renderError(error) {
  lastRefreshEl.textContent = "Coordinator unavailable";
  overviewGrid.innerHTML = metricCard("Dashboard Error", "Waiting", escapeHtml(error.message ?? String(error)));
  throughputChart.innerHTML = `<div class="empty-state">Status fetch failed. The coordinator may still be starting.</div>`;
  throughputBreakdown.innerHTML = "";
  leaderboard.innerHTML = `<div class="empty-state">No live stats available right now.</div>`;
  workersGrid.innerHTML = `<div class="empty-state">The dashboard will recover automatically once the coordinator responds again.</div>`;
}

function metricCard(label, value, subtext) {
  return `
    <article class="overview-card">
      <div class="metric-label">${escapeHtml(label)}</div>
      <div class="metric-value">${escapeHtml(value)}</div>
      <div class="metric-subtext">${escapeHtml(subtext)}</div>
    </article>
  `;
}

function miniMetric(label, value) {
  return `
    <div class="mini-metric">
      <div class="metric-label">${escapeHtml(label)}</div>
      <div class="mini-metric-value">${escapeHtml(value)}</div>
    </div>
  `;
}

function formatClock(date) {
  return date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

function formatDuration(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return "n/a";
  }
  const seconds = Math.max(0, Number(value));
  if (seconds < 60) {
    return `${seconds.toFixed(1)}s`;
  }
  const minutes = Math.floor(seconds / 60);
  const remainder = Math.round(seconds % 60);
  if (minutes < 60) {
    return `${minutes}m ${remainder}s`;
  }
  const hours = Math.floor(minutes / 60);
  return `${hours}h ${minutes % 60}m`;
}

function formatDecimal(value) {
  return Number(value ?? 0).toFixed(1);
}

function formatNumber(value) {
  return Number(value ?? 0).toLocaleString();
}

function formatRelativeSeconds(value) {
  if (value === null || value === undefined) {
    return "never";
  }
  if (value < 1) {
    return "just now";
  }
  if (value < 60) {
    return `${Math.round(value)}s ago`;
  }
  const minutes = Math.floor(value / 60);
  return `${minutes}m ago`;
}

function formatLastProgress(worker) {
  if (worker.last_duration_seconds === null || worker.last_duration_seconds === undefined) {
    return "No games finished yet";
  }
  return `Last game ${formatDuration(worker.last_duration_seconds)}`;
}

function shortPlatform(platform) {
  if (!platform) {
    return "unknown platform";
  }
  return platform.replaceAll("-64bit", "").replaceAll("-ARM64", "").replaceAll("-x86_64", "");
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

startPolling();
