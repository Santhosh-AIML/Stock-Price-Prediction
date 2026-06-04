/**
 * AI Stock Predictor — Frontend Logic
 * Handles theme toggle, stock search, chart rendering, and ML prediction display.
 */

"use strict";

// ─── Chart instances (kept globally for destroy/re-create) ──────────────────
const charts = {};

// ─── State ──────────────────────────────────────────────────────────────────
let currentTicker = "";
let currentPeriod = "6mo";

// ─── DOM refs ────────────────────────────────────────────────────────────────
const $ = (id) => document.getElementById(id);

// ============================================================
//  THEME TOGGLE
// ============================================================

(function initTheme() {
  const html = document.documentElement;
  const toggle = $("themeToggle");
  const icon = $("themeIcon");
  const label = $("themeLabel");

  const saved = localStorage.getItem("theme") || "dark";
  setTheme(saved);

  toggle.addEventListener("click", () => {
    const next = html.getAttribute("data-bs-theme") === "dark" ? "light" : "dark";
    setTheme(next);
    localStorage.setItem("theme", next);
  });

  function setTheme(mode) {
    html.setAttribute("data-bs-theme", mode);
    if (mode === "dark") {
      icon.className = "bi bi-sun-fill";
      label.textContent = "Light";
    } else {
      icon.className = "bi bi-moon-fill";
      label.textContent = "Dark";
    }
    // Re-render charts with new colours
    Object.values(charts).forEach((c) => {
      if (c && typeof c.update === "function") {
        applyChartTheme(c, mode);
        c.update();
      }
    });
  }
})();

// ============================================================
//  SEARCH
// ============================================================

$("searchBtn").addEventListener("click", doSearch);
$("searchInput").addEventListener("keydown", (e) => { if (e.key === "Enter") doSearch(); });

document.querySelectorAll(".ticker-chip").forEach((chip) => {
  chip.addEventListener("click", () => {
    $("searchInput").value = chip.dataset.ticker;
    doSearch();
  });
});

async function doSearch() {
  const ticker = $("searchInput").value.trim().toUpperCase();
  if (!ticker) return;

  currentTicker = ticker;
  hideError();
  showLoading("Fetching stock data…");

  try {
    const res = await post("/api/stock", { ticker, period: currentPeriod });
    if (res.error) { showError(res.error); return; }
    renderStockData(ticker, res);
    $("stockSection").classList.remove("d-none");
  } catch (err) {
    showError("Network error — could not reach the server.");
  } finally {
    hideLoading();
  }
}

// ─── Period selector ─────────────────────────────────────────────────────────
document.querySelectorAll(".period-btn").forEach((btn) => {
  btn.addEventListener("click", async () => {
    if (!currentTicker) return;
    document.querySelectorAll(".period-btn").forEach((b) => b.classList.remove("active"));
    btn.classList.add("active");
    currentPeriod = btn.dataset.period;
    showLoading("Updating chart…");
    try {
      const res = await post("/api/stock", { ticker: currentTicker, period: currentPeriod });
      if (!res.error) renderStockData(currentTicker, res, false);
    } finally {
      hideLoading();
    }
  });
});

// ============================================================
//  RENDER STOCK DATA
// ============================================================

function renderStockData(ticker, data, resetPred = true) {
  const { current, history, signals } = data;

  // Header
  $("companyName").textContent = current.name || ticker;
  $("tickerLabel").textContent = ticker;
  $("sectorBadge").textContent = current.sector || "";

  // Stats
  const fmt = (v, cur = current.currency) =>
    v != null ? `${cur === "INR" ? "₹" : "$"}${v.toLocaleString()}` : "—";

  $("statPrice").textContent = fmt(current.price);
  $("statOpen").textContent = fmt(current.open);
  $("statHigh").textContent = fmt(current.high);
  $("statLow").textContent = fmt(current.low);
  $("statVolume").textContent = current.volume != null ? fmtVolume(current.volume) : "—";
  $("statMarketCap").textContent = current.market_cap != null ? fmtMarketCap(current.market_cap) : "—";

  // Price change vs prev close
  const priceChangeEl = $("statPriceChange");
  if (current.price != null && current.prev_close != null) {
    const change = current.price - current.prev_close;
    const pct = ((change / current.prev_close) * 100).toFixed(2);
    const up = change >= 0;
    priceChangeEl.innerHTML = `<span class="${up ? "price-up" : "price-down"}">
      ${up ? "▲" : "▼"} ${Math.abs(change).toFixed(2)} (${up ? "+" : ""}${pct}%)
    </span>`;
  } else {
    priceChangeEl.textContent = "";
  }

  // Signal badge
  const sig = signals?.signal || "HOLD";
  const signalBadge = $("signalBadge");
  signalBadge.className = `signal-badge signal-${sig}`;
  const iconMap = { BUY: "bi-arrow-up-circle-fill", SELL: "bi-arrow-down-circle-fill", HOLD: "bi-dash-circle-fill" };
  $("signalIcon").className = `bi ${iconMap[sig] || "bi-dash-circle"}`;
  $("signalText").textContent = sig;
  $("signalReason").textContent = signals?.reason || "";

  // Charts
  renderPriceChart(history, ticker);
  renderVolumeChart(history);
  if (signals?.ma_dates) renderSignalChart(signals);

  // Hide old prediction
  if (resetPred) {
    $("predictionResults").classList.add("d-none");
    $("predError").classList.add("d-none");
  }
}

// ============================================================
//  PREDICTION
// ============================================================

$("runPredBtn").addEventListener("click", runPrediction);

async function runPrediction() {
  if (!currentTicker) { showError("Please search for a stock first."); return; }

  const period = $("predPeriod").value;
  const model = $("predModel").value;

  $("predError").classList.add("d-none");
  $("predictionResults").classList.add("d-none");
  showLoading(model === "lstm" ? "Training LSTM model… (may take ~30s)" : "Running prediction…");

  try {
    const res = await post("/api/predict", { ticker: currentTicker, period, model });
    if (res.error) {
      $("predError").textContent = res.error;
      $("predError").classList.remove("d-none");
      return;
    }
    renderPrediction(res);
  } catch (err) {
    $("predError").textContent = "Network error — could not complete prediction.";
    $("predError").classList.remove("d-none");
  } finally {
    hideLoading();
  }
}

function renderPrediction(data) {
  const { lr, lstm } = data;
  const cur = $("statPrice").textContent;

  // Next-day price cards
  const cardsEl = $("nextDayCards");
  cardsEl.innerHTML = "";
  const hasLR = lr && !lr.error;
  const hasLSTM = lstm && !lstm.error;

  if (hasLR) cardsEl.appendChild(makeNextDayCard("Linear Regression", lr.next_day_price, "bi-graph-up", "primary"));
  if (hasLSTM) cardsEl.appendChild(makeNextDayCard("LSTM Neural Network", lstm.next_day_price, "bi-cpu", "success"));

  // Tab visibility
  const tabLR = document.querySelector('[data-tab="lr"]');
  const tabLSTM = document.querySelector('[data-tab="lstm"]');

  tabLR.style.display = hasLR ? "" : "none";
  tabLSTM.style.display = hasLSTM ? "" : "none";

  if (hasLR) {
    renderModelPane("lr", lr);
    activateTab("lr");
  } else if (hasLSTM) {
    renderModelPane("lstm", lstm);
    activateTab("lstm");
  }

  $("predictionResults").classList.remove("d-none");
}

function makeNextDayCard(title, price, icon, color) {
  const div = document.createElement("div");
  div.className = "col-sm-6 col-md-4";
  div.innerHTML = `
    <div class="panel-card p-3 d-flex align-items-center gap-3">
      <div class="rounded-circle d-flex align-items-center justify-content-center flex-shrink-0"
           style="width:3rem;height:3rem;background:rgba(var(--bs-${color}-rgb),.15);">
        <i class="bi ${icon} text-${color} fs-5"></i>
      </div>
      <div>
        <p class="small text-muted mb-0">${title}</p>
        <p class="fw-bold mb-0 fs-5">Next Day: <span class="text-${color}">${price != null ? "$" + price.toFixed(2) : "—"}</span></p>
      </div>
    </div>`;
  return div;
}

function renderModelPane(key, modelData) {
  // Metrics
  const metricsEl = $(key + "Metrics");
  metricsEl.innerHTML = "";
  if (modelData.metrics) {
    const { rmse, mae, r2 } = modelData.metrics;
    metricsEl.innerHTML = `
      <div class="col-4"><div class="metric-box">
        <div class="metric-label">RMSE</div>
        <div class="metric-value text-warning">${rmse}</div>
      </div></div>
      <div class="col-4"><div class="metric-box">
        <div class="metric-label">MAE</div>
        <div class="metric-value text-info">${mae}</div>
      </div></div>
      <div class="col-4"><div class="metric-box">
        <div class="metric-label">R²</div>
        <div class="metric-value ${r2 >= 0.8 ? "text-success" : r2 >= 0.5 ? "text-warning" : "text-danger"}">${r2}</div>
      </div></div>`;
  }

  // Actual vs Predicted chart
  const canvasId = key + "Chart";
  destroyChart(canvasId);
  const theme = document.documentElement.getAttribute("data-bs-theme");
  const gridColor = theme === "dark" ? "rgba(255,255,255,.08)" : "rgba(0,0,0,.06)";
  const textColor = theme === "dark" ? "#adb5bd" : "#495057";

  charts[canvasId] = new Chart($(canvasId), {
    type: "line",
    data: {
      labels: modelData.dates,
      datasets: [
        {
          label: "Actual",
          data: modelData.actual,
          borderColor: "#4f8ef7",
          backgroundColor: "rgba(79,142,247,.08)",
          borderWidth: 2,
          pointRadius: 0,
          tension: 0.3,
          fill: true,
        },
        {
          label: "Predicted",
          data: modelData.predicted,
          borderColor: "#f59e0b",
          backgroundColor: "transparent",
          borderWidth: 2,
          borderDash: [5, 3],
          pointRadius: 0,
          tension: 0.3,
        },
      ],
    },
    options: chartOptions({ gridColor, textColor, title: "Actual vs Predicted Close Price" }),
  });
}

// Tabs
document.querySelectorAll("#predTabs .nav-link").forEach((btn) => {
  btn.addEventListener("click", () => activateTab(btn.dataset.tab));
});

function activateTab(tab) {
  document.querySelectorAll("#predTabs .nav-link").forEach((b) => {
    b.classList.toggle("active", b.dataset.tab === tab);
  });
  ["lr", "lstm"].forEach((k) => {
    $("pane-" + k).classList.toggle("d-none", k !== tab);
  });
}

// ============================================================
//  CHARTS
// ============================================================

function renderPriceChart(history, ticker) {
  destroyChart("priceChart");
  const theme = document.documentElement.getAttribute("data-bs-theme");
  const gridColor = theme === "dark" ? "rgba(255,255,255,.06)" : "rgba(0,0,0,.05)";
  const textColor = theme === "dark" ? "#adb5bd" : "#495057";

  const gradient = (ctx) => {
    const g = ctx.createLinearGradient(0, 0, 0, 280);
    g.addColorStop(0, "rgba(79,142,247,.25)");
    g.addColorStop(1, "rgba(79,142,247,0)");
    return g;
  };

  charts["priceChart"] = new Chart($("priceChart"), {
    type: "line",
    data: {
      labels: history.dates,
      datasets: [{
        label: `${ticker} Close`,
        data: history.close,
        borderColor: "#4f8ef7",
        backgroundColor: (ctx) => gradient(ctx.chart.ctx),
        borderWidth: 2,
        pointRadius: 0,
        tension: 0.3,
        fill: true,
      }],
    },
    options: chartOptions({ gridColor, textColor, title: "Closing Price" }),
  });
}

function renderVolumeChart(history) {
  destroyChart("volumeChart");
  const theme = document.documentElement.getAttribute("data-bs-theme");
  const gridColor = theme === "dark" ? "rgba(255,255,255,.06)" : "rgba(0,0,0,.05)";
  const textColor = theme === "dark" ? "#adb5bd" : "#495057";

  charts["volumeChart"] = new Chart($("volumeChart"), {
    type: "bar",
    data: {
      labels: history.dates,
      datasets: [{
        label: "Volume",
        data: history.volume,
        backgroundColor: "rgba(79,142,247,.45)",
        borderColor: "#4f8ef7",
        borderWidth: 0,
        borderRadius: 2,
      }],
    },
    options: chartOptions({ gridColor, textColor, title: "Daily Volume" }),
  });
}

function renderSignalChart(signals) {
  destroyChart("signalChart");
  const theme = document.documentElement.getAttribute("data-bs-theme");
  const gridColor = theme === "dark" ? "rgba(255,255,255,.06)" : "rgba(0,0,0,.05)";
  const textColor = theme === "dark" ? "#adb5bd" : "#495057";

  charts["signalChart"] = new Chart($("signalChart"), {
    type: "line",
    data: {
      labels: signals.ma_dates,
      datasets: [
        {
          label: "Close",
          data: signals.close,
          borderColor: "#4f8ef7",
          borderWidth: 1.5,
          pointRadius: 0,
          tension: 0.3,
          fill: false,
        },
        {
          label: "MA20",
          data: signals.ma20,
          borderColor: "#22c55e",
          borderWidth: 2,
          pointRadius: 0,
          tension: 0.3,
          fill: false,
          borderDash: [],
        },
        {
          label: "MA50",
          data: signals.ma50,
          borderColor: "#ef4444",
          borderWidth: 2,
          pointRadius: 0,
          tension: 0.3,
          fill: false,
          borderDash: [6, 3],
        },
      ],
    },
    options: chartOptions({ gridColor, textColor, title: "Moving Averages" }),
  });
}

// ─── Shared chart options factory ─────────────────────────────────────────────
function chartOptions({ gridColor, textColor, title }) {
  return {
    responsive: true,
    maintainAspectRatio: false,
    animation: { duration: 400 },
    interaction: { mode: "index", intersect: false },
    plugins: {
      legend: {
        labels: { color: textColor, font: { size: 12 }, boxWidth: 16, padding: 12 },
      },
      tooltip: {
        backgroundColor: "rgba(15,17,28,.92)",
        titleColor: "#e2e8f0",
        bodyColor: "#94a3b8",
        padding: 10,
        cornerRadius: 8,
      },
    },
    scales: {
      x: {
        ticks: { color: textColor, maxTicksLimit: 8, font: { size: 11 } },
        grid: { color: gridColor },
      },
      y: {
        ticks: { color: textColor, font: { size: 11 } },
        grid: { color: gridColor },
      },
    },
  };
}

function applyChartTheme(chart, mode) {
  const gridColor = mode === "dark" ? "rgba(255,255,255,.06)" : "rgba(0,0,0,.05)";
  const textColor = mode === "dark" ? "#adb5bd" : "#495057";
  const sc = chart.options.scales;
  if (sc?.x) { sc.x.ticks.color = textColor; sc.x.grid.color = gridColor; }
  if (sc?.y) { sc.y.ticks.color = textColor; sc.y.grid.color = gridColor; }
  if (chart.options.plugins?.legend?.labels) chart.options.plugins.legend.labels.color = textColor;
}

function destroyChart(id) {
  if (charts[id]) { charts[id].destroy(); delete charts[id]; }
}

// ============================================================
//  HELPERS
// ============================================================

async function post(url, body) {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  return res.json();
}

function showLoading(text = "Loading…") {
  $("loadingText").textContent = text;
  $("loadingOverlay").classList.add("active");
}

function hideLoading() {
  $("loadingOverlay").classList.remove("active");
}

function showError(msg) {
  $("searchErrorMsg").textContent = msg;
  $("searchError").classList.remove("d-none");
}

function hideError() {
  $("searchError").classList.add("d-none");
}

function fmtVolume(v) {
  if (v >= 1e9) return (v / 1e9).toFixed(1) + "B";
  if (v >= 1e6) return (v / 1e6).toFixed(1) + "M";
  if (v >= 1e3) return (v / 1e3).toFixed(1) + "K";
  return v.toLocaleString();
}

function fmtMarketCap(v) {
  if (v >= 1e12) return "$" + (v / 1e12).toFixed(2) + "T";
  if (v >= 1e9)  return "$" + (v / 1e9).toFixed(1)  + "B";
  if (v >= 1e6)  return "$" + (v / 1e6).toFixed(1)  + "M";
  return "$" + v.toLocaleString();
}
