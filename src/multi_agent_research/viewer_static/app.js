const state = {
  summaries: [],
  run: null,
  selected: null,
  tab: "rendered",
};

const $ = (selector) => document.querySelector(selector);
const escapeHtml = (value = "") =>
  String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");

const compact = new Intl.NumberFormat("en-US", { notation: "compact", maximumFractionDigits: 1 });

async function boot() {
  state.summaries = await fetch("/api/runs").then((response) => response.json());
  renderRunList();
  if (state.summaries.length) {
    await selectRun(state.summaries[0].run_id);
  } else {
    $("#run-summary").innerHTML = '<div class="metric"><strong>No runs found</strong><span>Point mar view at a results directory</span></div>';
  }
}

function renderRunList(filter = "") {
  const needle = filter.trim().toLowerCase();
  const rows = state.summaries.filter((run) =>
    [run.run_id, run.experiment_id, run.task_id, run.workflow]
      .filter(Boolean)
      .some((value) => value.toLowerCase().includes(needle)),
  );
  $("#run-list").innerHTML = rows.map((run) => `
    <button class="run-row ${state.run?.summary.run_id === run.run_id ? "active" : ""}" data-run-id="${run.run_id}">
      <div class="run-topline">
        <span class="run-id">${escapeHtml(run.experiment_id)}</span>
        <span class="status-dot ${escapeHtml(run.status)}"></span>
      </div>
      <div class="run-detail">
        <span>${escapeHtml(run.workflow.replaceAll("_", " "))}</span>
        <span>${compact.format(run.total_tokens || 0)} tok</span>
      </div>
      <div class="run-detail">
        <span>${formatDate(run.started_at)}</span>
        <span>${escapeHtml(run.final_answer ?? "—")}</span>
      </div>
    </button>
  `).join("");
  document.querySelectorAll("[data-run-id]").forEach((button) => {
    button.addEventListener("click", () => selectRun(button.dataset.runId));
  });
}

async function selectRun(runId) {
  state.run = await fetch(`/api/runs/${runId}`).then((response) => response.json());
  state.selected = null;
  renderRunList($("#run-filter").value);
  renderSummary();
  renderBoard();
  renderExchanges();
  renderLegend();
  renderInspector();
}

function renderSummary() {
  const run = state.run.summary;
  $("#run-summary").innerHTML = [
    metric("Run", run.experiment_id),
    metric("Workflow", run.workflow.replaceAll("_", " ")),
    metric("Status", run.status),
    metric("Final answer", run.final_answer ?? "—", "answer"),
    metric("Tokens", Number(run.total_tokens || 0).toLocaleString()),
    metric("Wall time", formatDuration(run.wall_time_ms)),
  ].join("");
  $("#board-caption").textContent = `${run.model_calls} model calls · ${run.task_id}`;
}

function metric(label, value, className = "") {
  return `<div class="metric"><span>${escapeHtml(label)}</span><strong class="${className}">${escapeHtml(value)}</strong></div>`;
}

function renderBoard() {
  const primaryAgents = state.run.agents.filter((agent) => {
    if (agent.role === "judge") return false;
    return state.run.stage_cards.some((card) => card.agent_id === agent.id);
  });
  const phases = state.run.phase_order.filter((phase) => phase.id !== "cross_examination");
  const columns = `155px repeat(${phases.length}, minmax(130px, 1fr))`;
  const cells = [
    '<div class="board-cell board-header" style="grid-column:1;grid-row:1">Agent</div>',
    ...phases.map((phase, phaseIndex) => `
      <div class="board-cell board-header" style="grid-column:${phaseIndex + 2};grid-row:1">
        ${escapeHtml(phase.label)}
      </div>
    `),
  ];
  primaryAgents.forEach((agent, agentIndex) => {
    const gridRow = agentIndex + 2;
    const agentCards = state.run.stage_cards.filter((card) => card.agent_id === agent.id);
    cells.push(`
      <div class="board-cell agent-label" style="grid-column:1;grid-row:${gridRow}">
        ${agentMarkup(agent, agentIndex)}
      </div>
    `);
    phases.forEach((phase, phaseIndex) => {
      if (phase.id === "aggregate") return;
      const card = agentCards.find((candidate) => candidate.phase === phase.id);
      cells.push(`
        <div class="board-cell" style="grid-column:${phaseIndex + 2};grid-row:${gridRow}">
          ${card ? stageCard(card, agentCards) : '<div class="empty-cell">·</div>'}
        </div>
      `);
    });
  });
  const aggregateCards = state.run.stage_cards.filter((card) => card.kind === "aggregate");
  const aggregatePhaseIndex = phases.findIndex((phase) => phase.id === "aggregate");
  if (aggregateCards.length && aggregatePhaseIndex >= 0) {
    const card = aggregateCards[aggregateCards.length - 1];
    cells.push(`
      <div
        class="board-cell aggregate-cell"
        style="grid-column:${aggregatePhaseIndex + 2};grid-row:2 / span ${Math.max(primaryAgents.length, 1)}"
      >
        ${stageCard(card, aggregateCards, "aggregate")}
      </div>
    `);
  }
  $("#debate-board").innerHTML = `
    <div
      class="board-grid"
      style="grid-template-columns:${columns};grid-template-rows:42px repeat(${Math.max(primaryAgents.length, 1)}, 92px)"
    >
      ${cells.join("")}
    </div>
  `;
  document.querySelectorAll("[data-card-step]").forEach((button) => {
    button.addEventListener("click", () => selectStage(button.dataset.cardStep));
  });
  renderAgentSigils();
}

function stageCard(card, trajectory, variant = "candidate") {
  const prior = trajectory
    .filter((candidate) => candidate.sequence < card.sequence)
    .sort((a, b) => b.sequence - a.sequence)[0];
  const changed = prior && prior.answer !== card.answer;
  return `
    <button class="stage-card ${variant} ${state.selected?.key === `stage:${card.step}` ? "active" : ""}" data-card-step="${escapeHtml(card.step)}">
      <div class="stage-answer">
        <strong>${escapeHtml(card.answer || "—")}</strong>
        <span>${card.confidence == null ? "" : `${card.confidence}% conf`}</span>
      </div>
      <div class="stage-meta">
        <span>${compact.format(card.total_tokens || 0)} tok</span>
        ${changed ? '<span class="answer-change" title="Answer changed"></span>' : "<span>—</span>"}
      </div>
    </button>
  `;
}

function renderExchanges() {
  const panel = $("#exchange-panel");
  panel.hidden = !state.run.exchanges.length;
  if (!state.run.exchanges.length) {
    $("#exchange-list").innerHTML = '<div class="empty-exchanges">This workflow has no directed cross-examination exchanges.</div>';
    return;
  }
  $("#exchange-list").innerHTML = state.run.exchanges.map((exchange, index) => `
    <article class="exchange-row">
      <div class="exchange-cell">
        <label>Round ${exchange.round} · Exchange ${exchange.exchange_index + 1}</label>
        <div class="exchange-route">
          ${agentMini(exchange.challenger_id, agentIndex(exchange.challenger_id))}
          <span class="route-arrow">→</span>
          ${agentMini(exchange.target_id, agentIndex(exchange.target_id))}
        </div>
      </div>
      <div class="exchange-cell">
        <label>Challenge</label>
        <p>${escapeHtml(exchange.challenge)}</p>
      </div>
      <div class="exchange-cell">
        <label>Response</label>
        <p>${escapeHtml(exchange.response)}</p>
      </div>
      <div class="exchange-cell">
        <label>Verdict</label>
        <button class="verdict-button ${exchange.verdict_status}" data-exchange-id="${exchange.id}">${escapeHtml(exchange.verdict_status)}</button>
      </div>
      <div class="exchange-cell exchange-stats">
        <span>${compact.format(exchange.total_tokens)} tok</span>
        <span>${formatDuration(exchange.latency_ms)}</span>
        <button class="inspect-exchange" data-exchange-id="${exchange.id}">open</button>
      </div>
    </article>
  `).join("");
  document.querySelectorAll("[data-exchange-id]").forEach((button) => {
    button.addEventListener("click", () => selectExchange(button.dataset.exchangeId));
  });
  renderAgentSigils();
}

function renderLegend() {
  $("#agent-legend").innerHTML = state.run.agents.map((agent, index) => agentMarkup(agent, index)).join("");
  renderAgentSigils();
}

function agentMarkup(agent, index) {
  return `
    ${agentSigil(agent.id, index)}
    <span><strong>${escapeHtml(agent.id)}</strong><small>${escapeHtml(agent.role || agent.model || "")}</small></span>
  `;
}

function agentMini(agentId, index) {
  return `
    <span class="route-agent">
      ${agentSigil(agentId, index)}
      <strong>${escapeHtml(agentId)}</strong>
    </span>
  `;
}

function agentSigil(agentId, index) {
  return `
    <canvas
      class="agent-icon"
      width="28"
      height="28"
      data-agent-sigil="${escapeHtml(agentId)}"
      data-agent-color="${index % 4}"
      aria-hidden="true"
    ></canvas>
  `;
}

function renderAgentSigils() {
  const palette = ["#42cef5", "#71e65f", "#f4bb35", "#bd80ef"];
  document.querySelectorAll("canvas[data-agent-sigil]").forEach((canvas) => {
    const context = canvas.getContext("2d");
    const colorIndex = Number(canvas.dataset.agentColor || 0) % palette.length;
    drawPixelSigil(
      context,
      canvas.dataset.agentSigil || "agent",
      palette[colorIndex],
    );
  });
}

const CURATED_SEEDS = [
  "agent-4", "agent-68", "agent-7", "agent-99", "agent-30", "agent-31",
  "agent-0", "agent-64", "agent-83", "agent-93", "agent-49", "agent-57",
  "agent-54", "agent-72", "agent-79", "agent-92"
];

function getCuratedSeed(seedText) {
  if (CURATED_SEEDS.includes(seedText)) {
    return seedText;
  }
  const hash = hashString(seedText);
  return CURATED_SEEDS[hash % CURATED_SEEDS.length];
}

function drawPixelSigil(context, seedText, color) {
  const curatedSeed = getCuratedSeed(seedText);
  const gridSize = 4;
  const pixel = 5;
  const offset = 4;

  context.clearRect(0, 0, 28, 28);
  context.fillStyle = "#0b1722";
  context.fillRect(0, 0, 28, 28);

  const hash = hashString(curatedSeed);
  const useRotational = (hash & 0x100) !== 0; // Bit 8 decides symmetry type
  const cellsState = [];
  let activeCount = 0;

  // We have 8 independent cell pairs in both symmetry types.
  // Generate states for 8 cells using bits of the hash.
  for (let i = 0; i < 8; i++) {
    const bit = (hash >> i) & 1;
    cellsState.push(bit);
    if (bit) activeCount++;
  }

  // Ensure density is balanced: if we have too few or too many active pairs,
  // deterministically force exactly 4 active pairs (50% density).
  if (activeCount < 3 || activeCount > 6) {
    const indices = [0, 1, 2, 3, 4, 5, 6, 7];
    let tempHash = hash;
    for (let i = indices.length - 1; i > 0; i--) {
      tempHash = xorshift(tempHash);
      const j = Math.floor(tempHash % (i + 1));
      const temp = indices[i];
      indices[i] = indices[j];
      indices[j] = temp;
    }
    cellsState.fill(0);
    for (let i = 0; i < 4; i++) {
      cellsState[indices[i]] = 1;
    }
  }

  // Draw the cells with horizontal or rotational symmetry
  context.fillStyle = color;
  const activePairs = [];
  for (let i = 0; i < 8; i++) {
    if (cellsState[i] === 1) {
      let cell1, cell2;

      if (useRotational) {
        // 180-degree rotational symmetry: Row 0..1 (y) and Column 0..3 (x)
        const cellY = Math.floor(i / 4);
        const cellX = i % 4;
        cell1 = { x: cellX, y: cellY };
        cell2 = { x: gridSize - 1 - cellX, y: gridSize - 1 - cellY };
      } else {
        // Horizontal symmetry: Row 0..3 (y) and Column 0..1 (x)
        const cellY = Math.floor(i / 2);
        const cellX = i % 2;
        cell1 = { x: cellX, y: cellY };
        cell2 = { x: gridSize - 1 - cellX, y: cellY };
      }

      // Draw cell 1
      context.fillRect(
        offset + cell1.x * pixel,
        offset + cell1.y * pixel,
        pixel,
        pixel,
      );
      // Draw cell 2
      context.fillRect(
        offset + cell2.x * pixel,
        offset + cell2.y * pixel,
        pixel,
        pixel,
      );

      activePairs.push({ cell1, cell2 });
    }
  }

  // Draw a symmetric highlight on one of the active cell pairs
  if (activePairs.length > 0) {
    context.globalAlpha = 0.5;
    context.fillStyle = "#ffffff";

    const highlightIdx = hashString(`${curatedSeed}:highlight`) % activePairs.length;
    const { cell1, cell2 } = activePairs[highlightIdx];

    // Highlight cell 1
    context.fillRect(
      offset + cell1.x * pixel,
      offset + cell1.y * pixel,
      pixel,
      pixel,
    );
    // Highlight mirrored cell 2
    context.fillRect(
      offset + cell2.x * pixel,
      offset + cell2.y * pixel,
      pixel,
      pixel,
    );
    context.globalAlpha = 1;
  }
}

function hashString(value) {
  let hash = 2166136261;
  for (let index = 0; index < value.length; index += 1) {
    hash ^= value.charCodeAt(index);
    hash = Math.imul(hash, 16777619);
  }
  return hash >>> 0;
}

function xorshift(value) {
  let next = value || 1;
  next ^= next << 13;
  next ^= next >>> 17;
  next ^= next << 5;
  return next >>> 0;
}

function agentIndex(agentId) {
  return Math.max(0, state.run.agents.findIndex((agent) => agent.id === agentId));
}

function selectStage(step) {
  const card = state.run.stage_cards.find((candidate) => candidate.step === step);
  const call = state.run.calls.find((candidate) => candidate.id === card.call_id || candidate.step === card.step);
  state.selected = { key: `stage:${step}`, agent_id: card.agent_id, title: `${card.agent_id} · ${card.phase_label}`, rendered: card.raw_response, call, metadata: card.metadata };
  renderBoard();
  renderInspector();
}

function selectExchange(exchangeId) {
  const exchange = state.run.exchanges.find((candidate) => candidate.id === exchangeId);
  state.selected = {
    key: `exchange:${exchangeId}`,
    challenger_id: exchange.challenger_id,
    target_id: exchange.target_id,
    title: `${exchange.challenger_id} → ${exchange.target_id} · round ${exchange.round}`,
    rendered: `CHALLENGE\n${exchange.challenge}\n\nRESPONSE\n${exchange.response}\n\nVERDICT\n${exchange.verdict}`,
    call: exchange.calls.verdict || exchange.calls.response || exchange.calls.challenge,
    metadata: {
      round: exchange.round,
      exchange_index: exchange.exchange_index,
      challenger_id: exchange.challenger_id,
      target_id: exchange.target_id,
      verdict_status: exchange.verdict_status,
      phase_calls: Object.fromEntries(Object.entries(exchange.calls).map(([phase, call]) => [phase, call?.id])),
    },
  };
  renderInspector();
}

function renderMarkdownAndMath(rawText) {
  if (typeof marked === "undefined") {
    return `<div class="rendered" style="white-space: pre-wrap;">${escapeHtml(rawText)}</div>`;
  }

  const mathBlocks = [];
  let tempText = rawText;

  function protect(regex) {
    tempText = tempText.replace(regex, (match) => {
      const placeholder = `@@MATHBLOCK_${mathBlocks.length}@@`;
      mathBlocks.push({ placeholder, original: match });
      return placeholder;
    });
  }

  // Protect LaTeX blocks from being mangled by marked's parser
  protect(/\$\$[\s\S]*?\$\$/g);
  protect(/\\\[[\s\S]*?\\\]/g);
  protect(/\\\([\s\S]*?\\\)/g);
  protect(/\$[^\s$](?:[^\$]*?[^\s$])?\$/g);

  // Parse markdown
  let html = marked.parse(tempText);

  // Restore LaTeX blocks
  for (let i = 0; i < mathBlocks.length; i++) {
    html = html.replace(mathBlocks[i].placeholder, mathBlocks[i].original);
  }

  return `<div class="rendered">${html}</div>`;
}

function renderInspector() {
  const selected = state.selected;
  $("#inspection-title").textContent = selected?.title || "Select a stage";

  const sigilsContainer = $("#inspector-sigils");
  if (sigilsContainer) {
    if (!selected) {
      sigilsContainer.innerHTML = "";
    } else if (selected.key.startsWith("stage:")) {
      const idx = agentIndex(selected.agent_id);
      sigilsContainer.innerHTML = agentSigil(selected.agent_id, idx);
    } else if (selected.key.startsWith("exchange:")) {
      sigilsContainer.innerHTML = `
        ${agentSigil(selected.challenger_id, agentIndex(selected.challenger_id))}
        <span class="route-arrow" style="font-size: 11px;">→</span>
        ${agentSigil(selected.target_id, agentIndex(selected.target_id))}
      `;
    }
    renderAgentSigils();
  }

  if (!selected) {
    $("#inspection-body").textContent = "Choose an answer card or cross-examination exchange.";
    return;
  }
  const call = selected.call || {};

  const rawText = selected.rendered || "No rendered response.";
  const parsedContent = renderMarkdownAndMath(rawText);

  const content = {
    rendered: parsedContent,
    messages: `<pre>${escapeHtml(JSON.stringify(call.messages || [], null, 2))}</pre>`,
    metadata: `<pre>${escapeHtml(JSON.stringify({
      selection: selected.metadata,
      call: call.metadata,
      request_parameters: call.request_parameters,
      prompt_references: call.prompt_references,
      model: call.response_model || call.requested_model,
      started_at: call.started_at,
      ended_at: call.ended_at,
    }, null, 2))}</pre>`,
    usage: `<pre>${escapeHtml(JSON.stringify({
      usage: call.usage || {},
      cost_usd: call.cost_usd,
      latency_ms: call.latency_ms,
      status: call.status,
      error: call.error,
    }, null, 2))}</pre>`,
  };

  const body = $("#inspection-body");
  body.innerHTML = content[state.tab];

  if (state.tab === "rendered" && typeof renderMathInElement !== "undefined") {
    try {
      renderMathInElement(body, {
        delimiters: [
          { left: "$$", right: "$$", display: true },
          { left: "$", right: "$", display: false },
          { left: "\\(", right: "\\)", display: false },
          { left: "\\[", right: "\\]", display: true },
        ],
        throwOnError: false,
      });
    } catch (e) {
      console.warn("KaTeX rendering failed:", e);
    }
  }
}

function formatDate(value) {
  if (!value) return "unknown";
  return new Date(value).toLocaleString([], { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" });
}

function formatDuration(milliseconds = 0) {
  if (milliseconds < 1000) return `${Math.round(milliseconds)} ms`;
  if (milliseconds < 60000) return `${(milliseconds / 1000).toFixed(1)} s`;
  return `${Math.floor(milliseconds / 60000)}m ${Math.round((milliseconds % 60000) / 1000)}s`;
}

$("#run-filter").addEventListener("input", (event) => renderRunList(event.target.value));
document.querySelectorAll("[data-tab]").forEach((button) => {
  button.addEventListener("click", () => {
    state.tab = button.dataset.tab;
    document.querySelectorAll("[data-tab]").forEach((candidate) => candidate.classList.toggle("active", candidate === button));
    renderInspector();
  });
});

boot().catch((error) => {
  $("#run-list").textContent = `Unable to load runs: ${error.message}`;
});
