const TABLE_ORDER = [
  "table0",
  "table1",
  "table2",
  "table3",
  "table4",
  "table5",
  "table6",
  "table7",
  "table8",
  "table9",
  "table10",
  "table11",
];

const state = {
  imports: [],
  selectedImportId: null,
  selectedImport: null,
  rows: [],
  currentCursor: 0,
  nextCursor: null,
  totalRows: 0,
  pageSize: 10,
  maxPageSize: 50,
  effectiveLimit: 10,
  loadingRows: false,
  activeView: "rows",
  analytics: null,
  analyticsError: "",
  loadingAnalytics: false,
};

const elements = {
  appVersion: document.querySelector("#app-version"),
  importCount: document.querySelector("#import-count"),
  uploadForm: document.querySelector("#upload-form"),
  uploadFile: document.querySelector("#upload-file"),
  uploadButton: document.querySelector("#upload-button"),
  uploadMessage: document.querySelector("#upload-message"),
  refreshImports: document.querySelector("#refresh-imports"),
  importsEmpty: document.querySelector("#imports-empty"),
  importsList: document.querySelector("#imports-list"),
  importTitle: document.querySelector("#import-title"),
  importDescription: document.querySelector("#import-description"),
  importCreated: document.querySelector("#import-created"),
  viewToggle: document.querySelector("#view-toggle"),
  viewRows: document.querySelector("#view-rows"),
  viewAnalytics: document.querySelector("#view-analytics"),
  warningsCard: document.querySelector("#warnings-card"),
  warningsList: document.querySelector("#warnings-list"),
  rowsToolbar: document.querySelector("#rows-toolbar"),
  rowsFooter: document.querySelector("#rows-footer"),
  rowsStatus: document.querySelector("#rows-status"),
  rowsStatusBottom: document.querySelector("#rows-status-bottom"),
  pageSize: document.querySelector("#page-size"),
  pageNumber: document.querySelector("#page-number"),
  prevPage: document.querySelector("#prev-page"),
  nextPage: document.querySelector("#next-page"),
  jumpPage: document.querySelector("#jump-page"),
  pageNumberBottom: document.querySelector("#page-number-bottom"),
  prevPageBottom: document.querySelector("#prev-page-bottom"),
  nextPageBottom: document.querySelector("#next-page-bottom"),
  jumpPageBottom: document.querySelector("#jump-page-bottom"),
  deleteImport: document.querySelector("#delete-import"),
  rowsList: document.querySelector("#rows-list"),
  analyticsPanel: document.querySelector("#analytics-panel"),
  analyticsMessage: document.querySelector("#analytics-message"),
  analyticsList: document.querySelector("#analytics-list"),
};

async function api(path, options = {}) {
  const headers = new Headers(options.headers || {});
  if (options.body && !headers.has("Content-Type")) {
    headers.set("Content-Type", "application/json");
  }
  const response = await fetch(path, {
    method: options.method || "GET",
    body: options.body,
    headers,
    cache: "no-store",
  });
  const payload = await response.json();
  if (!response.ok || !payload.ok) {
    throw new Error(payload.error || `Request failed: ${response.status}`);
  }
  return payload.data;
}

function setUploadMessage(message, isError = false) {
  elements.uploadMessage.textContent = message;
  elements.uploadMessage.style.color = isError ? "#8a2c1f" : "#5e5548";
}

function formatTimestamp(value) {
  if (!value) {
    return "Unknown time";
  }
  return String(value).replace("T", " ");
}

function buildPageSizeOptions(defaultSize, maxSize) {
  const values = new Set([defaultSize, 5, 10, 20, 50, maxSize]);
  return [...values]
    .filter((value) => Number.isFinite(value) && value >= 1 && value <= maxSize)
    .sort((left, right) => left - right);
}

function renderPageSizeOptions() {
  const options = buildPageSizeOptions(state.pageSize, state.maxPageSize);
  elements.pageSize.innerHTML = options
    .map((value) => `<option value="${value}"${value === state.pageSize ? " selected" : ""}>${value}</option>`)
    .join("");
}

function renderImports() {
  elements.importsList.innerHTML = "";
  elements.importsEmpty.style.display = state.imports.length ? "none" : "block";
  elements.importCount.textContent = `${state.imports.length} import${state.imports.length === 1 ? "" : "s"}`;
  for (const item of state.imports) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `import-item${item.import_id === state.selectedImportId ? " active" : ""}`;
    button.innerHTML = `
      <strong>${escapeHtml(item.filename)}</strong>
      <span>${escapeHtml(item.format)} · ${item.valid_rows} valid · ${item.invalid_rows} skipped</span>
      <span>${escapeHtml(formatTimestamp(item.created_at))}</span>
    `;
    button.addEventListener("click", () => void selectImport(item.import_id));
    elements.importsList.appendChild(button);
  }
}

function renderSelectedImport() {
  const selected = state.selectedImport;
  if (!selected) {
    elements.importTitle.textContent = "Select an upload to begin";
    elements.importDescription.textContent = "The viewer accepts preference, strict DPO, and latest-label exports.";
    elements.importCreated.textContent = "";
    elements.importCreated.classList.add("hidden");
    elements.viewToggle.classList.add("hidden");
    elements.warningsCard.classList.add("hidden");
    elements.rowsToolbar.classList.add("hidden");
    elements.rowsFooter.classList.add("hidden");
    elements.analyticsPanel.classList.add("hidden");
    elements.rowsList.innerHTML = "";
    elements.analyticsList.innerHTML = "";
    elements.analyticsMessage.textContent = "Choose an import to view analytics.";
    return;
  }

  elements.importTitle.textContent = selected.filename;
  elements.importDescription.textContent = `${selected.format} · ${selected.valid_rows} valid row(s) · ${selected.invalid_rows} skipped row(s)`;
  elements.importCreated.textContent = `Created ${formatTimestamp(selected.created_at)}`;
  elements.importCreated.classList.remove("hidden");
  elements.viewToggle.classList.remove("hidden");
  renderViewToggle();
  renderWarnings(selected);

  if (state.activeView === "analytics") {
    elements.rowsToolbar.classList.add("hidden");
    elements.rowsFooter.classList.add("hidden");
    elements.rowsList.innerHTML = "";
    elements.analyticsPanel.classList.remove("hidden");
    renderAnalytics();
    return;
  }

  elements.analyticsPanel.classList.add("hidden");
  renderRowsToolbar();
  renderRows();
}

function renderViewToggle() {
  elements.viewRows.classList.toggle("is-active", state.activeView === "rows");
  elements.viewAnalytics.classList.toggle("is-active", state.activeView === "analytics");
}

function renderWarnings(selected) {
  elements.warningsList.innerHTML = "";
  if (selected.warnings && selected.warnings.length) {
    elements.warningsCard.classList.remove("hidden");
    for (const warning of selected.warnings) {
      const item = document.createElement("div");
      item.className = "warning-item";
      item.textContent = `Line ${warning.line_number}: ${warning.message}`;
      elements.warningsList.appendChild(item);
    }
  } else {
    elements.warningsCard.classList.add("hidden");
  }
}

function renderRowsToolbar() {
  elements.rowsToolbar.classList.remove("hidden");
  elements.rowsFooter.classList.remove("hidden");
  const rangeStart = state.totalRows ? state.currentCursor + 1 : 0;
  const rangeEnd = state.totalRows ? Math.min(state.currentCursor + state.rows.length, state.totalRows) : 0;
  const pageNumber = state.totalRows ? Math.floor(state.currentCursor / Math.max(state.effectiveLimit, 1)) + 1 : 0;
  const totalPages = state.totalRows ? Math.ceil(state.totalRows / Math.max(state.effectiveLimit, 1)) : 0;
  const statusText = `Page ${pageNumber} · rows ${rangeStart}-${rangeEnd} of ${state.totalRows}`;
  const previousDisabled = state.loadingRows || state.currentCursor <= 0;
  const nextDisabled = state.loadingRows || state.nextCursor === null;
  const jumpDisabled = state.loadingRows || totalPages <= 1;

  elements.rowsStatus.textContent = statusText;
  elements.rowsStatusBottom.textContent = statusText;
  elements.prevPage.disabled = previousDisabled;
  elements.nextPage.disabled = nextDisabled;
  elements.prevPageBottom.disabled = previousDisabled;
  elements.nextPageBottom.disabled = nextDisabled;
  elements.pageSize.disabled = state.loadingRows;
  elements.pageNumber.disabled = jumpDisabled;
  elements.pageNumberBottom.disabled = jumpDisabled;
  elements.jumpPage.disabled = jumpDisabled;
  elements.jumpPageBottom.disabled = jumpDisabled;
  elements.pageNumber.min = totalPages > 0 ? "1" : "0";
  elements.pageNumberBottom.min = totalPages > 0 ? "1" : "0";
  elements.pageNumber.max = String(totalPages || 0);
  elements.pageNumberBottom.max = String(totalPages || 0);
  elements.pageNumber.value = totalPages ? String(pageNumber) : "";
  elements.pageNumberBottom.value = totalPages ? String(pageNumber) : "";
}

function renderRows() {
  elements.rowsList.innerHTML = "";
  if (!state.selectedImport) {
    return;
  }
  if (!state.rows.length) {
    elements.rowsList.innerHTML = `<div class="card empty-state">This import has no renderable rows.</div>`;
    return;
  }
  for (const row of state.rows) {
    const article = document.createElement("article");
    article.className = "card row-card";
    article.innerHTML = `
      <div class="row-head">
        <div>
          <p class="eyebrow">${escapeHtml(row.kind === "pair_export" ? "Pair Export" : "Latest Label")}</p>
          <h3>${escapeHtml(row.task_name || row.session_id || row.row_id)}</h3>
        </div>
        <div class="row-head-side">
          <strong>${escapeHtml(row.decision || "No decision")}</strong>
          <span class="muted">${escapeHtml(row.reviewer_username || "Unknown reviewer")}</span>
        </div>
      </div>
      <div class="row-meta">
        <span>Dataset: ${escapeHtml(row.dataset_id || "Unknown")}</span>
        <span>Session: ${escapeHtml(row.session_id || "Unknown")}</span>
        <span>Workflow: ${escapeHtml(row.workflow_name || "Unknown")}</span>
        <span>Checkpoint: ${escapeHtml(row.primary_ckpt || "Unknown")}</span>
        ${row.strict_dpo === null ? "" : `<span>Strict DPO: ${row.strict_dpo ? "Yes" : "No"}</span>`}
        <span>Updated: ${escapeHtml(formatTimestamp(row.created_at))}</span>
      </div>
      ${row.note ? `<p class="meta-line"><strong>Note:</strong> ${escapeHtml(row.note)}</p>` : ""}
      <div class="row-images"></div>
    `;
    const imagesRoot = article.querySelector(".row-images");
    for (const image of row.images) {
      const shell = document.createElement("section");
      const classes = ["image-shell"];
      if (image.is_good) {
        classes.push("good");
      }
      if (image.is_bad) {
        classes.push("bad");
      }
      if (image.has_defect) {
        classes.push("defect");
      }
      shell.className = classes.join(" ");
      shell.innerHTML = `
        <div class="image-card">
          <img loading="lazy" src="${image.media_url}" alt="${escapeHtml(image.slot_label)} image">
          <div class="image-body">
            <h4>${escapeHtml(image.slot_label)} · ${escapeHtml(image.image_name || `image_${image.image_index ?? ""}`)}</h4>
            <p class="meta-line">SHA256: ${escapeHtml(image.expected_sha256)}</p>
            <p class="meta-line">Checkpoint: ${escapeHtml(image.ckpt || "Unknown")}</p>
            <div class="prompt-block">
              <h5>Prompt</h5>
              <p>${escapeHtml(image.positive_prompt || "(empty)")}</p>
            </div>
            <div class="prompt-block">
              <h5>Negative</h5>
              <p>${escapeHtml(image.negative_prompt || "(empty)")}</p>
            </div>
          </div>
        </div>
      `;
      imagesRoot.appendChild(shell);
    }
    elements.rowsList.appendChild(article);
  }
}

function renderAnalytics() {
  elements.analyticsList.innerHTML = "";
  if (!state.selectedImport) {
    elements.analyticsMessage.textContent = "Choose an import to view analytics.";
    return;
  }
  if (state.loadingAnalytics) {
    elements.analyticsMessage.textContent = "Computing analytics from the persisted normalized rows...";
    elements.analyticsList.innerHTML = `<div class="card empty-state">Computing analytics...</div>`;
    return;
  }
  if (state.analyticsError) {
    elements.analyticsMessage.textContent = state.analyticsError;
    elements.analyticsList.innerHTML = `<div class="card empty-state">Analytics failed to load.</div>`;
    return;
  }
  if (!state.analytics) {
    elements.analyticsMessage.textContent = "Open Analytics to load the dashboard for this import.";
    return;
  }

  const summary = state.analytics.summary || {};
  const summaryText = [`${Number(summary.row_count || 0)} normalized row(s) merged as one dataset version.`];
  if (summary.note) {
    summaryText.push(summary.note);
  }
  elements.analyticsMessage.textContent = summaryText.join(" ");

  for (const tableId of TABLE_ORDER) {
    const table = state.analytics.tables?.[tableId];
    if (!table) {
      continue;
    }
    elements.analyticsList.appendChild(renderAnalyticsCard(table));
  }
}

function renderAnalyticsCard(table) {
  const article = document.createElement("article");
  article.className = "card analytics-card";
  article.innerHTML = `
    <div class="analytics-card-head">
      <div>
        <p class="eyebrow">${escapeHtml(String(table.id || "").toUpperCase())}</p>
        <h3>${escapeHtml(table.title || "Analytics table")}</h3>
      </div>
      <span class="pill subtle">${escapeHtml(table.kind || "chart")}</span>
    </div>
    <p class="muted analytics-description">${escapeHtml(table.description || "")}</p>
  `;

  if (!table.available) {
    article.insertAdjacentHTML("beforeend", `<div class="empty-state">${escapeHtml(table.empty_message || "This table is unavailable for the selected import.")}</div>`);
    return article;
  }

  if (table.kind === "scatter") {
    if (!Array.isArray(table.series) || !table.series.some((series) => Array.isArray(series.points) && series.points.length)) {
      article.insertAdjacentHTML("beforeend", `<div class="empty-state">${escapeHtml(table.empty_message || "No chart data available.")}</div>`);
      return article;
    }
    article.appendChild(renderScatterSection(table));
    article.appendChild(renderDataTable(table));
    return article;
  }

  if (table.kind === "bar") {
    if (!Array.isArray(table.rows) || !table.rows.length) {
      article.insertAdjacentHTML("beforeend", `<div class="empty-state">${escapeHtml(table.empty_message || "No chart data available.")}</div>`);
      return article;
    }
    article.appendChild(renderBarSection(table));
    article.appendChild(renderDataTable(table));
    return article;
  }

  if (table.kind === "heatmap") {
    if (!Array.isArray(table.columns) || !table.columns.length || !Array.isArray(table.matrix_rows) || !table.matrix_rows.length) {
      article.insertAdjacentHTML("beforeend", `<div class="empty-state">${escapeHtml(table.empty_message || "No chart data available.")}</div>`);
      return article;
    }
    article.appendChild(renderHeatmapSection(table));
    article.appendChild(renderHeatmapCellsTable(table));
    return article;
  }

  article.insertAdjacentHTML("beforeend", `<div class="empty-state">Unsupported analytics table kind.</div>`);
  return article;
}

function renderScatterSection(table) {
  const wrapper = document.createElement("section");
  wrapper.className = "analytics-section";
  wrapper.innerHTML = `
    <div class="chart-caption">${escapeHtml(table.x_label || "X")} vs ${escapeHtml(table.y_label || "Y")}</div>
    ${buildScatterSvg(table)}
    <div class="analytics-legend">${buildLegend(table.series || [])}</div>
  `;
  return wrapper;
}

function buildScatterSvg(table) {
  const width = 720;
  const height = 280;
  const margin = { top: 16, right: 18, bottom: 42, left: 50 };
  const plotWidth = width - margin.left - margin.right;
  const plotHeight = height - margin.top - margin.bottom;
  const allDates = uniqueValues((table.series || []).flatMap((series) => (series.points || []).map((point) => point.x)));
  const maxValue = Math.max(1, ...(table.series || []).flatMap((series) => (series.points || []).map((point) => Number(point.y) || 0)));
  const steps = Math.min(4, maxValue);
  const gridValues = uniqueValues([0, ...Array.from({ length: steps }, (_, index) => Math.round(((index + 1) / steps) * maxValue))]).sort((left, right) => left - right);

  const xFor = (value) => {
    if (!allDates.length) {
      return margin.left + plotWidth / 2;
    }
    if (allDates.length === 1) {
      return margin.left + plotWidth / 2;
    }
    const index = allDates.indexOf(value);
    return margin.left + (Math.max(index, 0) / (allDates.length - 1)) * plotWidth;
  };
  const yFor = (value) => margin.top + plotHeight - (Number(value) / maxValue) * plotHeight;

  const grid = gridValues
    .map((value) => {
      const y = yFor(value);
      return `
        <line x1="${margin.left}" y1="${y}" x2="${margin.left + plotWidth}" y2="${y}" class="chart-grid-line"></line>
        <text x="${margin.left - 8}" y="${y + 4}" text-anchor="end" class="chart-axis-text">${escapeHtml(String(value))}</text>
      `;
    })
    .join("");

  const xLabels = allDates
    .map((date, index) => {
      if (allDates.length > 8 && index !== 0 && index !== allDates.length - 1 && index % 2 === 1) {
        return "";
      }
      const x = xFor(date);
      return `<text x="${x}" y="${height - 12}" text-anchor="middle" class="chart-axis-text">${escapeHtml(date)}</text>`;
    })
    .join("");

  const points = (table.series || [])
    .map((series) =>
      (series.points || [])
        .map((point) => {
          const x = xFor(point.x);
          const y = yFor(point.y);
          return `
            <circle cx="${x}" cy="${y}" r="5.5" fill="${escapeHtml(series.color || "#666")}" class="chart-point">
              <title>${escapeHtml(series.label || series.key || "Series")} · ${escapeHtml(String(point.x))}: ${escapeHtml(String(point.y))}</title>
            </circle>
          `;
        })
        .join("")
    )
    .join("");

  return `
    <div class="chart-shell">
      <svg viewBox="0 0 ${width} ${height}" class="chart-svg" role="img" aria-label="${escapeHtml(table.title || "Scatter chart")}">
        <line x1="${margin.left}" y1="${margin.top + plotHeight}" x2="${margin.left + plotWidth}" y2="${margin.top + plotHeight}" class="chart-axis-line"></line>
        <line x1="${margin.left}" y1="${margin.top}" x2="${margin.left}" y2="${margin.top + plotHeight}" class="chart-axis-line"></line>
        ${grid}
        ${xLabels}
        ${points}
      </svg>
    </div>
  `;
}

function buildLegend(series) {
  return series
    .map(
      (item) => `
        <span class="legend-chip">
          <span class="legend-swatch" style="background:${escapeHtml(item.color || "#666")}"></span>
          ${escapeHtml(item.label || item.key || "Series")}
        </span>
      `
    )
    .join("");
}

function renderBarSection(table) {
  const wrapper = document.createElement("section");
  wrapper.className = "analytics-section";
  const maxValue = Math.max(1, ...table.rows.map((row) => Number(row.value) || 0));
  const rowsMarkup = table.rows
    .map((row) => {
      const percent = Math.max(0, Math.min(100, ((Number(row.value) || 0) / maxValue) * 100));
      return `
        <div class="bar-row">
          <div class="bar-label" title="${escapeHtml(row.label)}">${escapeHtml(row.label)}</div>
          <div class="bar-track">
            <div class="bar-fill" style="width:${percent.toFixed(2)}%"></div>
          </div>
          <div class="bar-value">${escapeHtml(formatMetricValue(row.value, table.value_format))}</div>
        </div>
      `;
    })
    .join("");
  wrapper.innerHTML = `
    <div class="chart-caption">${escapeHtml(table.x_label || "X")} vs ${escapeHtml(table.y_label || "Y")}</div>
    <div class="bar-chart">${rowsMarkup}</div>
  `;
  return wrapper;
}

function renderHeatmapSection(table) {
  const wrapper = document.createElement("section");
  wrapper.className = "analytics-section";
  const values = table.matrix_rows.flatMap((row) => row.values.map((cell) => cell.value).filter((value) => value !== null));
  const maxValue = Math.max(1, ...values.map((value) => Number(value) || 0));
  const headCells = table.columns.map((column) => `<th scope="col">${escapeHtml(column.label)}</th>`).join("");
  const bodyRows = table.matrix_rows
    .map((row) => {
      const cells = row.values
        .map((cell) => {
          const value = cell.value;
          const intensity = value === null ? 0 : (Number(value) || 0) / maxValue;
          const style = value === null ? "" : ` style="background:${heatColor(intensity, table.value_format)}"`;
          return `<td${style} title="${escapeHtml(`${row.label} vs ${cell.column}: ${cell.display}`)}">${escapeHtml(cell.display)}</td>`;
        })
        .join("");
      return `
        <tr>
          <th scope="row">${escapeHtml(row.label)}</th>
          ${cells}
          <td class="heatmap-total">${escapeHtml(formatMetricValue(row.total, table.value_format))}</td>
        </tr>
      `;
    })
    .join("");
  wrapper.innerHTML = `
    <div class="chart-caption">${escapeHtml(table.x_label || "X")} vs ${escapeHtml(table.y_label || "Y")}</div>
    <div class="heatmap-scroll">
      <table class="heatmap-table">
        <thead>
          <tr>
            <th scope="col">Winner \\ Loser</th>
            ${headCells}
            <th scope="col">Row total</th>
          </tr>
        </thead>
        <tbody>
          ${bodyRows}
        </tbody>
      </table>
    </div>
  `;
  return wrapper;
}

function renderDataTable(table) {
  const section = document.createElement("section");
  section.className = "analytics-section";
  if (table.kind === "scatter") {
    section.innerHTML = `
      <div class="table-wrap">
        <table class="analytics-table">
          <thead>
            <tr>
              <th scope="col">Date</th>
              <th scope="col">Series</th>
              <th scope="col">Count</th>
            </tr>
          </thead>
          <tbody>
            ${table.rows
              .map(
                (row) => `
                  <tr>
                    <td>${escapeHtml(row.date)}</td>
                    <td>${escapeHtml(row.series_label || row.decision_label || row.decision || row.series_key || "Series")}</td>
                    <td>${escapeHtml(String(row.count))}</td>
                  </tr>
                `
              )
              .join("")}
          </tbody>
        </table>
      </div>
    `;
    return section;
  }

  section.innerHTML = `
    <div class="table-wrap">
      <table class="analytics-table">
        <thead>
          <tr>
            <th scope="col">Checkpoint</th>
            <th scope="col">Value</th>
            <th scope="col">Count</th>
            <th scope="col">Appearances</th>
          </tr>
        </thead>
        <tbody>
          ${table.rows
            .map(
              (row) => `
                <tr>
                  <td>${escapeHtml(row.label)}</td>
                  <td>${escapeHtml(formatMetricValue(row.value, table.value_format))}</td>
                  <td>${escapeHtml(String(row.count))}</td>
                  <td>${escapeHtml(String(row.denominator))}</td>
                </tr>
              `
            )
            .join("")}
        </tbody>
      </table>
    </div>
  `;
  return section;
}

function renderHeatmapCellsTable(table) {
  const section = document.createElement("section");
  section.className = "analytics-section";
  if (!Array.isArray(table.rows) || !table.rows.length) {
    section.innerHTML = `<div class="empty-state">No directed checkpoint pairs contributed non-zero cells.</div>`;
    return section;
  }
  section.innerHTML = `
    <div class="table-wrap">
      <table class="analytics-table">
        <thead>
          <tr>
            <th scope="col">Winner</th>
            <th scope="col">Loser</th>
            <th scope="col">Value</th>
            <th scope="col">Raw count</th>
            <th scope="col">Matchups</th>
          </tr>
        </thead>
        <tbody>
          ${table.rows
            .map(
              (row) => `
                <tr>
                  <td>${escapeHtml(row.row_label)}</td>
                  <td>${escapeHtml(row.column_label)}</td>
                  <td>${escapeHtml(formatMetricValue(row.value, table.value_format))}</td>
                  <td>${escapeHtml(String(row.count))}</td>
                  <td>${escapeHtml(String(row.denominator))}</td>
                </tr>
              `
            )
            .join("")}
        </tbody>
      </table>
    </div>
  `;
  return section;
}

function heatColor(intensity, format) {
  const clamped = Math.max(0, Math.min(1, Number(intensity) || 0));
  if (format === "percentage") {
    return `rgba(47, 95, 79, ${0.08 + clamped * 0.76})`;
  }
  return `rgba(178, 74, 44, ${0.08 + clamped * 0.76})`;
}

function formatMetricValue(value, format) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return "—";
  }
  if (format === "percentage") {
    return `${(Number(value) * 100).toFixed(1)}%`;
  }
  return Number(value).toLocaleString();
}

function uniqueValues(values) {
  return [...new Set(values.filter((value) => value !== undefined && value !== null && String(value) !== ""))];
}

async function loadConfig() {
  const data = await api("/api/v1/config");
  elements.appVersion.textContent = `v${data.app_version}`;
  state.pageSize = Number(data.default_page_size) || 10;
  state.maxPageSize = Number(data.max_page_size) || Math.max(state.pageSize, 50);
  state.effectiveLimit = state.pageSize;
  renderPageSizeOptions();
}

async function refreshImports({ preserveSelection = true } = {}) {
  const data = await api("/api/v1/imports");
  state.imports = data.imports;
  if (!preserveSelection || !state.imports.some((item) => item.import_id === state.selectedImportId)) {
    state.selectedImportId = state.imports[0]?.import_id || null;
  }
  renderImports();
  if (state.selectedImportId) {
    await selectImport(state.selectedImportId, { reloadList: false });
  } else {
    state.selectedImport = null;
    state.rows = [];
    state.analytics = null;
    state.analyticsError = "";
    state.currentCursor = 0;
    state.nextCursor = null;
    state.totalRows = 0;
    renderSelectedImport();
  }
}

async function selectImport(importId, { reloadList = false } = {}) {
  if (!importId) {
    return;
  }
  if (reloadList) {
    await refreshImports();
    return;
  }
  state.selectedImportId = importId;
  renderImports();
  const detail = await api(`/api/v1/imports/${importId}`);
  state.selectedImport = detail.import;
  state.rows = [];
  state.analytics = null;
  state.analyticsError = "";
  state.currentCursor = 0;
  state.nextCursor = null;
  state.totalRows = 0;
  if (state.activeView === "analytics") {
    renderSelectedImport();
    await loadAnalytics();
    return;
  }
  await loadRowsPage(0);
}

async function loadRowsPage(cursor) {
  if (!state.selectedImportId || state.loadingRows) {
    return;
  }
  state.loadingRows = true;
  renderSelectedImport();
  try {
    const data = await api(`/api/v1/imports/${state.selectedImportId}/rows?cursor=${cursor}&limit=${state.pageSize}`);
    state.rows = data.items;
    state.currentCursor = data.cursor;
    state.nextCursor = data.next_cursor;
    state.totalRows = data.total;
    state.effectiveLimit = data.limit || state.pageSize;
    if (state.pageSize !== state.effectiveLimit) {
      state.pageSize = state.effectiveLimit;
      renderPageSizeOptions();
    }
  } finally {
    state.loadingRows = false;
    renderSelectedImport();
  }
}

async function loadAnalytics() {
  if (!state.selectedImportId || state.loadingAnalytics) {
    return;
  }
  state.loadingAnalytics = true;
  state.analyticsError = "";
  renderSelectedImport();
  try {
    state.analytics = await api(`/api/v1/imports/${state.selectedImportId}/analytics`);
  } catch (error) {
    state.analytics = null;
    state.analyticsError = error.message;
  } finally {
    state.loadingAnalytics = false;
    renderSelectedImport();
  }
}

async function switchView(nextView) {
  if (!["rows", "analytics"].includes(nextView) || state.activeView === nextView) {
    return;
  }
  state.activeView = nextView;
  renderSelectedImport();
  if (!state.selectedImportId) {
    return;
  }
  if (nextView === "analytics") {
    if (!state.analytics && !state.loadingAnalytics) {
      await loadAnalytics();
    }
    return;
  }
  if (!state.rows.length && !state.loadingRows) {
    await loadRowsPage(0);
  }
}

async function goToPreviousPage() {
  const previousCursor = Math.max(0, state.currentCursor - Math.max(state.effectiveLimit, 1));
  if (previousCursor === state.currentCursor && state.currentCursor !== 0) {
    return;
  }
  await loadRowsPage(previousCursor);
}

async function goToNextPage() {
  if (state.nextCursor === null) {
    return;
  }
  await loadRowsPage(state.nextCursor);
}

async function jumpToPage(rawValue) {
  if (!state.selectedImportId || state.loadingRows) {
    return;
  }
  const totalPages = state.totalRows ? Math.ceil(state.totalRows / Math.max(state.effectiveLimit, 1)) : 0;
  if (!totalPages) {
    return;
  }
  const requested = Number(rawValue);
  if (!Number.isFinite(requested)) {
    return;
  }
  const pageNumber = Math.min(Math.max(Math.trunc(requested), 1), totalPages);
  const cursor = (pageNumber - 1) * Math.max(state.effectiveLimit, 1);
  await loadRowsPage(cursor);
}

async function changePageSize() {
  const nextSize = Number(elements.pageSize.value);
  if (!Number.isFinite(nextSize) || nextSize < 1 || nextSize === state.pageSize) {
    return;
  }
  state.pageSize = Math.min(nextSize, state.maxPageSize);
  renderPageSizeOptions();
  if (state.selectedImportId && state.activeView === "rows") {
    await loadRowsPage(0);
  }
}

async function uploadCurrentFile(event) {
  event.preventDefault();
  const file = elements.uploadFile.files?.[0];
  if (!file) {
    setUploadMessage("Choose a JSONL file first.", true);
    return;
  }
  elements.uploadButton.disabled = true;
  setUploadMessage("Importing file...");
  try {
    const text = await file.text();
    const data = await api("/api/v1/imports", {
      method: "POST",
      body: JSON.stringify({ filename: file.name, text }),
    });
    setUploadMessage(`Imported ${data.import.filename}.`);
    elements.uploadForm.reset();
    await refreshImports({ preserveSelection: false });
    await selectImport(data.import.import_id, { reloadList: false });
  } catch (error) {
    setUploadMessage(error.message, true);
  } finally {
    elements.uploadButton.disabled = false;
  }
}

async function deleteSelectedImport() {
  if (!state.selectedImportId) {
    return;
  }
  const confirmed = window.confirm(`Delete import ${state.selectedImport?.filename || state.selectedImportId}?`);
  if (!confirmed) {
    return;
  }
  try {
    await api(`/api/v1/imports/${state.selectedImportId}`, { method: "DELETE" });
    setUploadMessage("Import deleted.");
    await refreshImports({ preserveSelection: false });
  } catch (error) {
    setUploadMessage(error.message, true);
  }
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function bindPageJumpInput(input, button) {
  button.addEventListener("click", () => void jumpToPage(input.value));
  input.addEventListener("keydown", (event) => {
    if (event.key !== "Enter") {
      return;
    }
    event.preventDefault();
    void jumpToPage(input.value);
  });
}

async function bootstrap() {
  elements.uploadForm.addEventListener("submit", (event) => void uploadCurrentFile(event));
  elements.refreshImports.addEventListener("click", () => void refreshImports({ preserveSelection: true }));
  elements.pageSize.addEventListener("change", () => void changePageSize());
  elements.viewRows.addEventListener("click", () => void switchView("rows"));
  elements.viewAnalytics.addEventListener("click", () => void switchView("analytics"));
  bindPageJumpInput(elements.pageNumber, elements.jumpPage);
  bindPageJumpInput(elements.pageNumberBottom, elements.jumpPageBottom);
  elements.prevPage.addEventListener("click", () => void goToPreviousPage());
  elements.nextPage.addEventListener("click", () => void goToNextPage());
  elements.prevPageBottom.addEventListener("click", () => void goToPreviousPage());
  elements.nextPageBottom.addEventListener("click", () => void goToNextPage());
  elements.deleteImport.addEventListener("click", () => void deleteSelectedImport());

  try {
    await loadConfig();
    await refreshImports({ preserveSelection: false });
    setUploadMessage("Ready.");
  } catch (error) {
    setUploadMessage(error.message, true);
  }
}

void bootstrap();
