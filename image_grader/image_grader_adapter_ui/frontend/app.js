const state = {
  config: null,
  facets: null,
  templates: [],
  sessions: [],
  totalFiltered: 0,
  cursor: 0,
  nextCursor: null,
  pageSize: 50,
  selectedSessionKeys: new Set(),
  defaultRankExpression: "",
  busy: false,
};

const $ = (selector) => document.querySelector(selector);

const el = {
  totalSessions: $("#total-sessions"),
  selectedCount: $("#selected-count"),
  modelCount: $("#model-count"),
  tabs: document.querySelectorAll(".tab"),
  views: document.querySelectorAll(".view"),
  filterTemplate: $("#filter-template"),
  filterCkpt: $("#filter-ckpt"),
  filterAspect: $("#filter-aspect"),
  filterWorkflow: $("#filter-workflow"),
  runLimit: $("#run-limit"),
  clearFilters: $("#clear-filters"),
  applyFilters: $("#apply-filters"),
  modelChecks: $("#model-checks"),
  policyChecks: $("#policy-checks"),
  templateName: $("#template-name"),
  saveTemplate: $("#save-template"),
  reloadTemplates: $("#reload-templates"),
  savedTemplates: $("#saved-templates"),
  browseStatus: $("#browse-status"),
  sessionTable: $("#session-table"),
  selectPage: $("#select-page"),
  selectAllSessions: $("#select-all-sessions"),
  clearSelection: $("#clear-selection"),
  prevPage: $("#prev-page"),
  nextPage: $("#next-page"),
  pageStatus: $("#page-status"),
  runPlayground: $("#run-playground"),
  rankExpression: $("#rank-expression"),
  rankExample: $("#rank-example"),
  rankPercent: $("#rank-percent"),
  rankKeyReference: $("#rank-key-reference"),
  playgroundSummary: $("#playground-summary"),
  playgroundRanking: $("#playground-ranking"),
  playgroundResults: $("#playground-results"),
  runReport: $("#run-report"),
  reportSummary: $("#report-summary"),
  flags: $("#flags"),
  pivot: $("#pivot"),
  dryRun: $("#dry-run"),
  writeLabels: $("#write-labels"),
  exportAligned: $("#export-aligned"),
  reviewerResults: $("#reviewer-results"),
  message: $("#message"),
};

async function api(path, options = {}) {
  const headers = new Headers(options.headers || {});
  if (options.body && !headers.has("Content-Type")) {
    headers.set("Content-Type", "application/json");
  }
  const response = await fetch(path, {
    method: options.method || "GET",
    headers,
    body: options.body,
    cache: "no-store",
  });
  const payload = await response.json();
  if (!response.ok || !payload.ok) {
    throw new Error(payload.error || `Request failed: ${response.status}`);
  }
  return payload.data;
}

function setBusy(value) {
  state.busy = value;
  for (const button of document.querySelectorAll("button:not(.tab)")) {
    button.disabled = value;
  }
  if (!value) {
    updatePagerState();
  }
}

function setMessage(text, error = false) {
  el.message.textContent = text || "";
  el.message.classList.toggle("error", Boolean(error));
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function option(value, label = value, selected = false) {
  return `<option value="${escapeHtml(value)}"${selected ? " selected" : ""}>${escapeHtml(label)}</option>`;
}

function checkpointAlias(checkpoint) {
  const raw = String(checkpoint ?? "");
  const configured = state.facets?.checkpoint_aliases?.[raw];
  if (configured) {
    return configured;
  }
  return raw.replaceAll("\\", "/").split("/").pop() || raw || "unknown";
}

function checkpointMarkup(checkpoint, className = "checkpoint-alias") {
  const raw = String(checkpoint ?? "");
  const alias = checkpointAlias(raw);
  const title = raw && raw !== alias ? `Original checkpoint: ${raw}` : raw;
  return `<span class="${escapeHtml(className)}" title="${escapeHtml(title)}">${escapeHtml(alias)}</span>`;
}

function safeDomId(value) {
  return String(value).replace(/[^A-Za-z0-9_-]+/g, "_");
}

function selectedCheckedValues(container) {
  return [...container.querySelectorAll("input[type='checkbox']:checked")].map((item) => item.value);
}

function currentFilters() {
  const filters = {};
  if (el.filterTemplate.value) {
    filters.prompt_template_keys = [el.filterTemplate.value];
  }
  if (el.filterCkpt.value) {
    filters.ckpts = [el.filterCkpt.value];
  }
  if (el.filterAspect.value) {
    filters.aspect_ratios = [el.filterAspect.value];
  }
  if (el.filterWorkflow.value) {
    filters.workflow_names = [el.filterWorkflow.value];
  }
  return filters;
}

function queryForFilters(cursor = 0, limit = state.pageSize) {
  const params = new URLSearchParams();
  params.set("cursor", String(Math.max(0, cursor)));
  params.set("limit", String(limit));
  const mapping = [
    ["prompt_template_keys", "prompt_template_key"],
    ["ckpts", "ckpt"],
    ["aspect_ratios", "aspect_ratio"],
    ["workflow_names", "workflow_name"],
  ];
  const filters = currentFilters();
  for (const [filterKey, queryKey] of mapping) {
    for (const value of filters[filterKey] || []) {
      params.append(queryKey, value);
    }
  }
  return params.toString();
}

function selectedSessionKeys() {
  return [...state.selectedSessionKeys].sort();
}

function checkedModels() {
  const values = selectedCheckedValues(el.modelChecks);
  if (!values.length) {
    throw new Error("Select at least one eval model.");
  }
  return values;
}

function checkedPolicies() {
  const values = selectedCheckedValues(el.policyChecks);
  if (!values.length) {
    throw new Error("Select at least one crop / resize policy.");
  }
  return values;
}

function currentTemplatePayload() {
  const models = checkedModels();
  const policies = checkedPolicies();
  const terms = [];
  for (const model of models) {
    for (const policy of policies) {
      terms.push({ model, preprocess_policy: policy, weight: 1 });
    }
  }
  return {
    name: el.templateName.value.trim() || "ai_eval_v1",
    models,
    preprocess_policies: policies,
    score_formula: {
      type: "weighted_sum",
      missing: "fail",
      terms,
    },
    report: {
      bad_absolute_threshold: 5,
      bad_relative_delta: 1,
      error_rate_threshold: 0.2,
      below_score_threshold: 5,
    },
    decision: {
      winner_min_score: 6,
      winner_min_delta: 0.75,
      tie_delta: 0.25,
      both_good_min: 7.5,
      both_bad_max: 3.5,
      on_error: "skip",
      on_ambiguous: "skip",
    },
  };
}

function requestPayload({ includeScores = false, useSavedTemplate = true } = {}) {
  const keys = selectedSessionKeys();
  const payload = {};
  if (keys.length) {
    payload.session_keys = keys;
  } else {
    payload.filters = currentFilters();
    const limit = Number.parseInt(el.runLimit.value, 10);
    if (Number.isFinite(limit) && limit > 0) {
      payload.limit = limit;
    }
  }
  if (includeScores) {
    payload.include_scores = true;
  }
  if (useSavedTemplate && el.savedTemplates.value) {
    payload.template_name = el.savedTemplates.value;
  } else {
    payload.template = currentTemplatePayload();
  }
  return payload;
}

function playgroundPayload() {
  const payload = requestPayload({ includeScores: true, useSavedTemplate: false });
  const expression = el.rankExpression.value.trim();
  payload.rank_expression = expression || state.defaultRankExpression || defaultRankExpression(selectedRankModels(), selectedRankPolicies());
  const percent = Number.parseFloat(el.rankPercent.value);
  payload.rank_percent = Number.isFinite(percent) ? percent : 10;
  return payload;
}

function scopeText() {
  const selectedCount = state.selectedSessionKeys.size;
  if (selectedCount) {
    return `selected ${selectedCount} session${selectedCount === 1 ? "" : "s"}`;
  }
  const limit = Number.parseInt(el.runLimit.value, 10) || "all";
  return `filtered subset, max ${limit} session${limit === 1 ? "" : "s"}`;
}

function renderConfig() {
  const models = state.config.enabled_models || [];
  const policies = state.config.preprocess_policies || ["native"];
  el.modelCount.textContent = `${models.length} eval model${models.length === 1 ? "" : "s"}`;
  el.modelChecks.innerHTML = models
    .map((model) => checkboxRow("model", model, model, true))
    .join("");
  el.policyChecks.innerHTML = policies
    .map((policy) => checkboxRow("policy", policy, policyLabel(policy), true))
    .join("");
  renderRankExamples();
}

function checkboxRow(prefix, value, label, checked) {
  const id = `${prefix}-${safeDomId(value)}`;
  return `
    <label class="check-row" for="${escapeHtml(id)}">
      <input id="${escapeHtml(id)}" type="checkbox" value="${escapeHtml(value)}"${checked ? " checked" : ""}>
      <span>${escapeHtml(label)}</span>
    </label>
  `;
}

function policyLabel(value) {
  const labels = {
    native: "Native",
    fit_pad_square: "Fit pad square",
    center_crop_square: "Center crop square",
  };
  return labels[value] || value;
}

function renderRankExamples() {
  const models = selectedRankModels();
  const policies = selectedRankPolicies();
  const previousDefault = state.defaultRankExpression;
  const nextDefault = defaultRankExpression(models, policies);
  const current = el.rankExpression.value.trim();
  if (!current || current === previousDefault || current === "avg(scores())") {
    el.rankExpression.value = nextDefault;
  }
  state.defaultRankExpression = nextDefault;
  el.rankExample.innerHTML = [
    option("", "Choose formula", false),
    ...rankExamples(models, policies).map((item) => option(item.value, item.label, item.value === el.rankExpression.value.trim())),
  ].join("");
  renderRankReference(models, policies, rankExamples(models, policies));
}

function selectedRankModels() {
  const checked = selectedCheckedValues(el.modelChecks);
  return checked.length ? checked : state.config.enabled_models || [];
}

function selectedRankPolicies() {
  const checked = selectedCheckedValues(el.policyChecks);
  return checked.length ? checked : state.config.preprocess_policies || ["native"];
}

function rankExamples(models, policies) {
  const model = models[0] || "model_id";
  const firstPolicy = policies[0] || "native";
  const secondPolicy = policies[1] || firstPolicy;
  const directFirst = scoreLookup(firstPolicy, model);
  const directSecond = scoreLookup(secondPolicy, model);
  const defaultExpression = defaultRankExpression(models, policies);
  const explicitValues = scoreTerms(models, policies).slice(0, 3);
  if (!explicitValues.length) {
    explicitValues.push("0");
  }
  const explicitList = explicitValues.join(", ");
  return [
    { label: "Average all selected", value: defaultExpression },
    { label: "Direct score", value: directFirst },
    { label: "Average two crops", value: `(${directFirst} + ${directSecond}) / 2` },
    { label: "Weighted crop blend", value: `${directFirst} * 0.7 + ${directSecond} * 0.3` },
    { label: "Penalize crop spread", value: `avg(scores()) - abs(${directFirst} - ${directSecond}) * 0.25` },
    { label: "Lower quartile", value: "percentile(scores(), 25)" },
    { label: "Upper quartile", value: "percentile(scores(), 75)" },
    { label: "Median score", value: "percentile(scores(), 50)" },
    { label: "Interquartile mean", value: "(percentile(scores(), 25) + percentile(scores(), 75)) / 2" },
    { label: "Direct-value p75", value: `percentile(${explicitList}, 75)` },
    { label: "Semicolon p75", value: `percentile(${explicitList}; 75)` },
    { label: "Max of min and p75", value: `max(min(${explicitList}), percentile(${explicitList}, 75))` },
    { label: "Clamp mean to quartiles", value: `max(p25(${explicitList}), min(avg(${explicitList}), p75(${explicitList})))` },
    { label: "Best method", value: "max(scores())" },
    { label: "Worst method", value: "min(scores())" },
  ];
}

function defaultRankExpression(models, policies) {
  const terms = scoreTerms(models, policies);
  if (!terms.length) {
    return "0";
  }
  return `(${terms.join(" + ")}) / ${terms.length}`;
}

function scoreTerms(models, policies) {
  const terms = [];
  for (const policy of policies) {
    for (const model of models) {
      terms.push(scoreLookup(policy, model));
    }
  }
  return terms;
}

function scoreLookup(policy, model) {
  return `score[${JSON.stringify(policy)}][${JSON.stringify(model)}]`;
}

function renderRankReference(models, policies, examples) {
  const keys = scoreTerms(models, policies);
  const exampleButtons = examples
    .map((item) => `
      <button class="formula-chip" type="button" data-expression="${escapeHtml(item.value)}">
        ${escapeHtml(item.label)}
      </button>
    `)
    .join("");
  el.rankKeyReference.innerHTML = `
    <div class="rank-reference-block">
      <strong>Usable keys</strong>
      <div class="code-list">
        ${keys.map((key) => `<code>${escapeHtml(key)}</code>`).join("") || "<span class=\"muted\">No selected keys.</span>"}
      </div>
    </div>
    <div class="rank-reference-block">
      <strong>Examples</strong>
      <div class="formula-chip-list">${exampleButtons}</div>
    </div>
  `;
}

function renderFacets() {
  const facets = state.facets?.facets || {};
  renderSelect(el.filterTemplate, facets.prompt_template_keys || [], "All prompt templates");
  renderSelect(el.filterCkpt, facets.ckpts || [], "All generation models", checkpointAlias);
  renderSelect(el.filterAspect, facets.aspect_ratios || [], "All aspect ratios");
  renderSelect(el.filterWorkflow, facets.workflow_names || [], "All workflows");
  el.totalSessions.textContent = `${state.facets?.total_sessions || 0} session${state.facets?.total_sessions === 1 ? "" : "s"}`;
}

function renderSelect(select, values, allLabel, labelForValue = (value) => value) {
  const current = select.value;
  const sorted = [...values].sort((a, b) => String(a).localeCompare(String(b)));
  select.innerHTML = option("", allLabel) + sorted.map((value) => option(value, labelForValue(value), value === current)).join("");
  if (current && !sorted.includes(current)) {
    select.value = "";
  }
}

function renderTemplates() {
  const current = el.savedTemplates.value;
  el.savedTemplates.innerHTML =
    option("", "Use current editor config") +
    state.templates.map((item) => option(item.name, item.name, item.name === current)).join("");
}

function fillTemplateForm(template) {
  if (!template) {
    return;
  }
  el.templateName.value = template.name || "ai_eval_v1";
  setCheckedValues(el.modelChecks, new Set(template.models || []));
  setCheckedValues(el.policyChecks, new Set(template.preprocess_policies || []));
  renderRankExamples();
}

function setCheckedValues(container, values) {
  for (const input of container.querySelectorAll("input[type='checkbox']")) {
    input.checked = values.has(input.value);
  }
}

function renderSessions() {
  updateSelectedCount();
  const start = state.totalFiltered ? state.cursor + 1 : 0;
  const end = state.cursor + state.sessions.length;
  el.browseStatus.textContent = `${state.totalFiltered} matching sessions. Showing ${start}-${end}.`;
  el.pageStatus.textContent = `${start}-${end} of ${state.totalFiltered}`;
  if (!state.sessions.length) {
    el.sessionTable.innerHTML = `<div class="empty">No generated sessions match the current filters.</div>`;
    updatePagerState();
    return;
  }
  el.sessionTable.innerHTML = `
    <div class="session-list">
      ${state.sessions.map(renderSessionRow).join("")}
    </div>
  `;
  updatePagerState();
}

function renderSessionRow(session) {
  const checked = state.selectedSessionKeys.has(session.session_key);
  const imageIndex = session.first_image_index ?? 0;
  return `
    <article class="session-card">
      <label class="session-select">
        <input class="session-check" type="checkbox" data-session-key="${escapeHtml(session.session_key)}"${checked ? " checked" : ""}>
        <span>${checked ? "Selected" : "Select"}</span>
      </label>
      <img class="thumb" alt="" src="${originalImageUrl(session.session_key, imageIndex)}">
      <div class="session-main">
        <strong>${escapeHtml(session.task_name || session.task_yaml_name || "untitled task")}</strong>
        <div class="muted">${escapeHtml(session.dataset_display_name || session.dataset_id || "")}</div>
        <div class="muted">${escapeHtml(session.image_count)} images · ${escapeHtml(session.workflow_name || "workflow n/a")}</div>
      </div>
      <div class="session-tags">
        ${pillList(session.prompt_template_keys)}
        ${checkpointPillList(session.checkpoints || session.ckpts)}
        ${pillList(session.aspect_ratios)}
      </div>
    </article>
  `;
}

function pillList(values) {
  const items = values || [];
  if (!items.length) {
    return `<span class="muted">n/a</span>`;
  }
  return `<div class="pill-list">${items.map((item) => `<span class="pill">${escapeHtml(item)}</span>`).join("")}</div>`;
}

function checkpointPillList(values) {
  const items = values || [];
  if (!items.length) {
    return `<span class="muted">n/a</span>`;
  }
  return `<div class="pill-list">${items.map((item) => {
    const raw = typeof item === "object" && item !== null ? item.name : item;
    const alias = typeof item === "object" && item !== null ? item.alias : checkpointAlias(raw);
    const title = raw && raw !== alias ? `Original checkpoint: ${raw}` : raw;
    return `<span class="pill" title="${escapeHtml(title)}">${escapeHtml(alias)}</span>`;
  }).join("")}</div>`;
}

function updateSelectedCount() {
  const count = state.selectedSessionKeys.size;
  const matching = Number(state.totalFiltered || 0);
  el.selectedCount.textContent = matching
    ? `${count} selected (${matching} matching)`
    : `${count} selected`;
}

function updatePagerState() {
  el.prevPage.disabled = state.busy || state.cursor <= 0;
  el.nextPage.disabled = state.busy || state.nextCursor === null;
}

function renderReport(report) {
  const summary = report.summary || {};
  el.reportSummary.innerHTML = renderMetricCards([
    ["Sessions", summary.session_count],
    ["Images", summary.image_count],
    ["Scores", summary.score_count],
    ["Score errors", summary.error_score_count],
  ]);

  const flags = report.bad_fit_flags || [];
  el.flags.innerHTML = flags.length
    ? flags.slice(0, 30).map(renderFlag).join("")
    : `<p class="muted">No bad-fit flags for this run.</p>`;

  const rows = report.tables?.prompt_template_model_aspect || [];
  el.pivot.innerHTML = renderTable(rows.slice(0, 200), [
    "prompt_template_key",
    "ckpt",
    "aspect_ratio",
    "eval_model",
    "preprocess_policy",
    "count",
    "mean",
    "median",
    "p10",
    "p90",
    "error_rate",
    "below_threshold_rate",
  ]);
}

function renderFlag(flag) {
  return `
    <div class="flag">
      <strong>${checkpointMarkup(flag.ckpt)} under ${escapeHtml(flag.prompt_template_key)} / ${escapeHtml(flag.aspect_ratio)}</strong>
      <div class="muted">${escapeHtml(flag.eval_model)} · ${escapeHtml(policyLabel(flag.preprocess_policy))}</div>
      <div class="muted">${escapeHtml((flag.reasons || []).join(", "))}</div>
      <div>mean ${formatNumber(flag.mean)} · peer median ${formatNumber(flag.context_median)} · errors ${formatPercent(flag.error_rate)}</div>
    </div>
  `;
}

function renderPlayground(data) {
  renderReport(data.report);
  const summary = data.report.summary || {};
  const ranking = data.playground_ranking || null;
  const metrics = [
    ["Sessions", summary.session_count],
    ["Images", summary.image_count],
    ["Scores", summary.score_count],
    ["Errors", summary.error_score_count],
  ];
  if (ranking) {
    metrics.push(["Ranked", ranking.ranked_count], ["Bucket", ranking.bucket_size]);
  }
  el.playgroundSummary.innerHTML = renderMetricCards(metrics);
  const scores = data.scores || [];
  if (!scores.length) {
    el.playgroundRanking.innerHTML = "";
    el.playgroundResults.innerHTML = `<div class="empty">No scores were returned.</div>`;
    return;
  }
  const groups = new Map();
  for (const score of scores) {
    const key = `${score.session_key}:${score.image_index}`;
    if (!groups.has(key)) {
      groups.set(key, []);
    }
    groups.get(key).push(score);
  }
  const usableScoreCount = scores.filter((score) => score.ok && typeof score.score === "number" && Number.isFinite(score.score)).length;
  el.playgroundRanking.innerHTML = usableScoreCount
    ? renderRankingBuckets(ranking, groups)
    : `<div class="empty error-state" role="alert">
        <strong>No usable scores were produced.</strong>
        <span>All ${escapeHtml(scores.length)} score attempts failed. Check the grader dependencies and model files, then review the errors below.</span>
      </div>`;
  const allCards = [...groups.values()]
    .sort((a, b) => groupSortKey(a).localeCompare(groupSortKey(b)))
    .map((rows) => renderPlaygroundCard(rows))
    .join("");
  el.playgroundResults.innerHTML = `
    <section class="rank-section">
      <div class="rank-section-head">
        <h3>All scored images</h3>
        <span>${escapeHtml(groups.size)} image${groups.size === 1 ? "" : "s"}</span>
      </div>
      <div class="result-grid">${allCards}</div>
    </section>
  `;
}

function groupSortKey(rows) {
  const first = rows[0] || {};
  return `${first.session_id || ""}:${String(first.image_index || 0).padStart(5, "0")}`;
}

function renderRankingBuckets(ranking, groups) {
  if (!ranking) {
    return "";
  }
  if (!ranking.ranked_count) {
    return `<div class="empty">No images produced a usable rank score.</div>`;
  }
  const percent = formatCompactPercent(ranking.percent);
  const errorNote = ranking.invalid_count
    ? `<div class="empty">${escapeHtml(ranking.invalid_count)} image${ranking.invalid_count === 1 ? "" : "s"} did not produce a usable rank score.</div>`
    : "";
  return `
    ${errorNote}
    ${renderRankSection(`Best ${percent}%`, ranking.top || [], groups, "Best")}
    ${renderRankSection(`Worst ${percent}%`, ranking.bottom || [], groups, "Worst")}
  `;
}

function renderRankSection(title, items, groups, rankLabel) {
  const cards = items
    .map((item) => renderPlaygroundCard(groups.get(item.key) || [], item, rankLabel))
    .filter(Boolean)
    .join("");
  return `
    <section class="rank-section">
      <div class="rank-section-head">
        <h3>${escapeHtml(title)}</h3>
        <span>${escapeHtml(items.length)} image${items.length === 1 ? "" : "s"}</span>
      </div>
      <div class="result-grid">${cards || `<div class="empty">No ranked images.</div>`}</div>
    </section>
  `;
}

function renderPlaygroundCard(rows, rankItem = null, rankLabel = "Mean") {
  if (!rows.length) {
    return "";
  }
  const hasRankItem = rankItem && typeof rankItem === "object";
  const first = rows[0];
  const policies = [...new Set(rows.map((row) => row.preprocess_policy))].sort();
  const policyAverages = averageScoresByPolicy(rows);
  const meanScore = averageScore(rows);
  const badgeScore = hasRankItem && typeof rankItem.score === "number" ? rankItem.score : meanScore;
  const rankNumber = hasRankItem
    ? rankItem.bucket === "bottom"
      ? rankItem.reverse_rank
      : rankItem.rank
    : null;
  const rankText = hasRankItem && Number.isFinite(Number(rankNumber))
    ? rankItem.bucket === "bottom"
      ? `#${rankNumber} low`
      : `#${rankNumber} high`
    : "";
  return `
    <article class="result-card">
      <div class="result-head">
        <img class="thumb" alt="" src="${originalImageUrl(first.session_key, first.image_index)}">
        <div>
          <strong>${escapeHtml(first.task_name)} · image ${escapeHtml(first.image_index)}</strong>
          <div class="muted">${escapeHtml(first.prompt_template_key)} · ${checkpointMarkup(first.ckpt)} · ${escapeHtml(first.aspect_ratio)}</div>
        </div>
        <div class="score-badge">
          <span>${escapeHtml(hasRankItem ? rankLabel : "Mean")}</span>
          <strong>${escapeHtml(formatNumber(badgeScore))}</strong>
          ${rankText ? `<small>${escapeHtml(rankText)}</small>` : ""}
        </div>
      </div>
      <div class="preview-strip">
        <figure>
          <img alt="" src="${originalImageUrl(first.session_key, first.image_index)}">
          <figcaption>Original</figcaption>
        </figure>
        ${policies.map((policy) => `
          <figure>
            <img alt="" src="${preprocessedImageUrl(first.session_key, first.image_index, policy)}">
            <figcaption>
              <span>${escapeHtml(policyLabel(policy))}</span>
              <strong>${escapeHtml(formatNumber(policyAverages.get(policy)?.mean))}</strong>
            </figcaption>
          </figure>
        `).join("")}
      </div>
      ${renderPolicyAverageGrid(policies, policyAverages)}
      ${renderScoreRows([...rows].sort(scoreSort))}
    </article>
  `;
}

function scoreSort(a, b) {
  return `${a.eval_model}:${a.preprocess_policy}`.localeCompare(`${b.eval_model}:${b.preprocess_policy}`);
}

function averageScore(rows) {
  const values = rows
    .filter((row) => row.ok && typeof row.score === "number" && Number.isFinite(Number(row.score)))
    .map((row) => Number(row.score));
  if (!values.length) {
    return null;
  }
  return values.reduce((total, value) => total + value, 0) / values.length;
}

function averageScoresByPolicy(rows) {
  const grouped = new Map();
  for (const row of rows) {
    const policy = String(row.preprocess_policy || "unknown");
    if (!grouped.has(policy)) {
      grouped.set(policy, []);
    }
    if (row.ok && typeof row.score === "number" && Number.isFinite(Number(row.score))) {
      grouped.get(policy).push(Number(row.score));
    }
  }
  return new Map([...grouped.entries()].map(([policy, values]) => [
    policy,
    {
      mean: values.length ? values.reduce((total, value) => total + value, 0) / values.length : null,
      count: values.length,
    },
  ]));
}

function renderPolicyAverageGrid(policies, averages) {
  if (policies.length <= 1) {
    return "";
  }
  return `
    <div class="crop-average-grid" aria-label="Average score by crop method">
      ${policies.map((policy) => {
        const average = averages.get(policy);
        return `
          <div class="crop-average">
            <span>${escapeHtml(policyLabel(policy))}</span>
            <strong>${escapeHtml(formatNumber(average?.mean))}</strong>
            <small>${escapeHtml(average?.count ?? 0)} score${average?.count === 1 ? "" : "s"}</small>
          </div>
        `;
      }).join("")}
    </div>
  `;
}

function renderReviewer(result, mode) {
  if (mode === "export") {
    el.reviewerResults.innerHTML = renderMetricCards([
      ["Events", result.event_count],
    ]) + `<p class="muted">${escapeHtml(result.label_events_path)}<br>${escapeHtml(result.labels_latest_path)}</p>`;
    return;
  }
  renderReport(result.report);
  const labels = result.labels || {};
  const items = labels.items || [];
  const rows = items.map((item) => ({
    session_id: item.session_id,
    skipped: Boolean(item.skipped),
    reason: item.reason || "",
    decision: item.decision?.decision || "",
    chosen_image_indices: (item.decision?.chosen_image_indices || []).join(", "),
    scores: item.decision ? JSON.stringify(item.decision.scores || {}) : "",
  }));
  el.reviewerResults.innerHTML = renderMetricCards([
    ["Candidate sessions", items.length],
    ["Written events", result.written_events ?? labels.event_count ?? 0],
    ["Skipped", labels.skipped_count ?? 0],
  ]) + renderTable(rows, ["session_id", "skipped", "reason", "decision", "chosen_image_indices", "scores"]);
}

function renderMetricCards(items) {
  return items
    .map(([label, value]) => `
      <div class="metric">
        <span>${escapeHtml(label)}</span>
        <strong>${escapeHtml(value ?? 0)}</strong>
      </div>
    `)
    .join("");
}

function renderTable(rows, fields) {
  if (!rows.length) {
    return `<p class="muted">No rows.</p>`;
  }
  return `
    <div class="table-scroll">
      <table class="score-table">
        <thead><tr>${fields.map((field) => `<th>${escapeHtml(headerLabel(field))}</th>`).join("")}</tr></thead>
        <tbody>
          ${rows.map((row) => `<tr>${fields.map((field) => `<td>${formatTableCell(field, row[field])}</td>`).join("")}</tr>`).join("")}
        </tbody>
      </table>
    </div>
  `;
}

function renderScoreRows(rows) {
  if (!rows.length) {
    return `<p class="muted">No score rows.</p>`;
  }
  return `
    <div class="score-stack">
      ${rows.map((row) => {
        const score = typeof row.score === "number" ? Math.max(0, Math.min(10, Number(row.score))) : null;
        const width = score === null ? 0 : score * 10;
        return `
          <div class="score-row">
            <div>
              <strong>${escapeHtml(row.eval_model)}</strong>
              <span>${escapeHtml(policyLabel(row.preprocess_policy))}</span>
            </div>
            <div class="score-track" aria-label="Score ${escapeHtml(formatNumber(row.score))}">
              <span style="width: ${escapeHtml(width)}%"></span>
            </div>
            <strong class="score-value">${escapeHtml(row.ok ? formatNumber(row.score) : "error")}</strong>
            ${row.error ? `<p class="score-error">${escapeHtml(row.error)}</p>` : ""}
          </div>
        `;
      }).join("")}
    </div>
  `;
}

function headerLabel(value) {
  return String(value).replaceAll("_", " ");
}

function formatCell(value) {
  if (typeof value === "number") {
    return escapeHtml(formatNumber(value));
  }
  if (typeof value === "boolean") {
    return value ? "yes" : "no";
  }
  return escapeHtml(value ?? "");
}

function formatTableCell(field, value) {
  return field === "ckpt" ? checkpointMarkup(value) : formatCell(value);
}

function formatNumber(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return "n/a";
  }
  return Number(value).toFixed(3);
}

function formatPercent(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return "n/a";
  }
  return `${(Number(value) * 100).toFixed(1)}%`;
}

function formatCompactPercent(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) {
    return "n/a";
  }
  return Number.isInteger(number) ? String(number) : number.toFixed(1);
}

function originalImageUrl(sessionKey, imageIndex) {
  return `/media/original/${encodeURIComponent(sessionKey)}/${encodeURIComponent(imageIndex ?? 0)}`;
}

function preprocessedImageUrl(sessionKey, imageIndex, policy) {
  return `/media/preprocessed/${encodeURIComponent(sessionKey)}/${encodeURIComponent(imageIndex ?? 0)}/${encodeURIComponent(policy)}`;
}

async function loadConfig() {
  state.config = await api("/api/v1/config");
  renderConfig();
}

async function loadFacets() {
  state.facets = await api("/api/v1/facets");
  renderFacets();
}

async function loadSessions(cursor = 0) {
  const data = await api(`/api/v1/sessions?${queryForFilters(cursor)}`);
  state.sessions = data.items || [];
  state.totalFiltered = data.total || 0;
  state.cursor = data.cursor || 0;
  state.nextCursor = data.next_cursor ?? null;
  state.pageSize = data.limit || state.pageSize;
  renderSessions();
}

async function selectAllMatchingSessions() {
  let cursor = 0;
  let added = 0;
  let total = 0;
  const pageLimit = 500;
  do {
    const data = await api(`/api/v1/sessions?${queryForFilters(cursor, pageLimit)}`);
    const items = data.items || [];
    total = Number(data.total || total || 0);
    for (const session of items) {
      if (session?.session_key && !state.selectedSessionKeys.has(session.session_key)) {
        state.selectedSessionKeys.add(session.session_key);
        added += 1;
      }
    }
    updateSelectedCount();
    cursor = data.next_cursor ?? null;
  } while (cursor !== null);
  renderSessions();
  return { added, total };
}

async function loadTemplates() {
  const data = await api("/api/v1/templates");
  state.templates = (data.templates || []).filter((item) => !item.error);
  renderTemplates();
}

async function withBusy(task) {
  try {
    setBusy(true);
    await task();
  } catch (error) {
    setMessage(error.message, true);
  } finally {
    setBusy(false);
  }
}

function showView(name) {
  for (const tab of el.tabs) {
    tab.classList.toggle("is-active", tab.dataset.view === name);
  }
  for (const view of el.views) {
    view.classList.toggle("is-active", view.id === `view-${name}`);
  }
}

for (const tab of el.tabs) {
  tab.addEventListener("click", () => showView(tab.dataset.view));
}

el.rankExample.addEventListener("change", () => {
  if (!el.rankExample.value) {
    return;
  }
  el.rankExpression.value = el.rankExample.value;
});

el.rankKeyReference.addEventListener("click", (event) => {
  if (!(event.target instanceof Element)) {
    return;
  }
  const button = event.target.closest(".formula-chip");
  if (!button) {
    return;
  }
  el.rankExpression.value = button.dataset.expression || "";
  renderRankExamples();
});

el.modelChecks.addEventListener("change", renderRankExamples);
el.policyChecks.addEventListener("change", renderRankExamples);

el.applyFilters.addEventListener("click", () => void withBusy(async () => {
  await loadSessions(0);
  setMessage(`Loaded ${state.totalFiltered} matching sessions. Selected subset remains ${state.selectedSessionKeys.size}.`);
}));

el.clearFilters.addEventListener("click", () => void withBusy(async () => {
  el.filterTemplate.value = "";
  el.filterCkpt.value = "";
  el.filterAspect.value = "";
  el.filterWorkflow.value = "";
  await loadSessions(0);
  setMessage("Filters cleared.");
}));

el.selectPage.addEventListener("click", () => {
  for (const session of state.sessions) {
    state.selectedSessionKeys.add(session.session_key);
  }
  renderSessions();
  setMessage(`Selected ${state.sessions.length} sessions from this page.`);
});

el.selectAllSessions.addEventListener("click", () => void withBusy(async () => {
  setMessage("Selecting every session that matches the current filters.");
  const result = await selectAllMatchingSessions();
  setMessage(`Selected all ${result.total} matching sessions. Total selected: ${state.selectedSessionKeys.size}. Added ${result.added}.`);
}));

el.clearSelection.addEventListener("click", () => {
  state.selectedSessionKeys.clear();
  renderSessions();
  setMessage("Selection cleared. Runs will use the current filters and max-session limit.");
});

el.prevPage.addEventListener("click", () => void withBusy(async () => {
  await loadSessions(Math.max(0, state.cursor - state.pageSize));
}));

el.nextPage.addEventListener("click", () => void withBusy(async () => {
  if (state.nextCursor !== null) {
    await loadSessions(state.nextCursor);
  }
}));

el.sessionTable.addEventListener("change", (event) => {
  const input = event.target;
  if (!(input instanceof HTMLInputElement) || !input.classList.contains("session-check")) {
    return;
  }
  const key = input.dataset.sessionKey;
  if (!key) {
    return;
  }
  if (input.checked) {
    state.selectedSessionKeys.add(key);
  } else {
    state.selectedSessionKeys.delete(key);
  }
  const label = input.closest(".session-select")?.querySelector("span");
  if (label) {
    label.textContent = input.checked ? "Selected" : "Select";
  }
  updateSelectedCount();
});

el.reloadTemplates.addEventListener("click", () => void withBusy(async () => {
  await loadTemplates();
  setMessage("Templates reloaded.");
}));

el.saveTemplate.addEventListener("click", () => void withBusy(async () => {
  const data = await api("/api/v1/templates", {
    method: "POST",
    body: JSON.stringify(currentTemplatePayload()),
  });
  await loadTemplates();
  el.savedTemplates.value = data.template.name;
  setMessage(`Saved template ${data.template.name}.`);
}));

el.savedTemplates.addEventListener("change", () => {
  const selected = state.templates.find((item) => item.name === el.savedTemplates.value);
  if (selected?.template) {
    fillTemplateForm(selected.template);
    setMessage(`Loaded template ${selected.name} into the editor.`);
  }
});

el.runPlayground.addEventListener("click", () => void withBusy(async () => {
  setMessage(`Running playground on ${scopeText()} with the current editor config.`);
  const data = await api("/api/v1/playground", {
    method: "POST",
    body: JSON.stringify(playgroundPayload()),
  });
  renderPlayground(data);
  showView("playground");
  const summary = data.report?.summary || {};
  const allScoresFailed = Number(summary.score_count || 0) > 0 && Number(summary.error_score_count || 0) >= Number(summary.score_count || 0);
  if (allScoresFailed) {
    setMessage(`Playground run ${data.run_id} completed, but all ${summary.score_count} score attempts failed. Review the grader errors below.`, true);
  } else {
    setMessage(`Playground run ${data.run_id} complete. Scores: ${data.score_count}.`);
  }
}));

el.runReport.addEventListener("click", () => void withBusy(async () => {
  setMessage(`Running report on ${scopeText()}.`);
  const data = await api("/api/v1/reports", {
    method: "POST",
    body: JSON.stringify(requestPayload({ useSavedTemplate: true })),
  });
  renderReport(data.report);
  showView("reports");
  setMessage(`Report ${data.run_id} complete. Scores: ${data.score_count}.`);
}));

el.dryRun.addEventListener("click", () => void withBusy(async () => {
  setMessage(`Dry-running AI reviewer on ${scopeText()}.`);
  const data = await api("/api/v1/labels/dry-run", {
    method: "POST",
    body: JSON.stringify(requestPayload({ useSavedTemplate: true })),
  });
  renderReviewer(data, "dry-run");
  showView("reviewer");
  setMessage(`Dry run complete. Candidate events: ${data.labels.items.filter((item) => !item.skipped).length}; skipped: ${data.labels.skipped_count}.`);
}));

el.writeLabels.addEventListener("click", () => void withBusy(async () => {
  setMessage(`Writing AI labels for ${scopeText()}.`);
  const data = await api("/api/v1/labels/write", {
    method: "POST",
    body: JSON.stringify(requestPayload({ useSavedTemplate: true })),
  });
  renderReviewer(data, "write");
  showView("reviewer");
  setMessage(`Wrote ${data.written_events} AI label events.\n${data.ai_label_events_path}\n${data.aligned_export.label_events_path}`);
}));

el.exportAligned.addEventListener("click", () => void withBusy(async () => {
  const data = await api("/api/v1/exports/aligned", {
    method: "POST",
    body: JSON.stringify({}),
  });
  renderReviewer(data, "export");
  showView("reviewer");
  setMessage(`Aligned exports written.\n${data.label_events_path}\n${data.labels_latest_path}`);
}));

void withBusy(async () => {
  await loadConfig();
  await loadFacets();
  await loadSessions(0);
  await loadTemplates();
  setMessage("Ready. Select sessions for an exact subset, or use filters plus the max-session limit.");
});
