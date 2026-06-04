/**
 * main.js
 * --------
 * Client-side logic for the AI Stock Price Predictor dashboard.
 * Handles form submission, API calls, Plotly chart rendering,
 * metric display, and the 30-day forecast table.
 */

'use strict';

/* ─── DOM refs ─────────────────────────────────────────────────────────── */
const form          = document.getElementById('predictForm');
const tickerInput   = document.getElementById('tickerInput');
const periodSelect  = document.getElementById('periodSelect');
const submitBtn     = document.getElementById('submitBtn');
const btnLabel      = document.getElementById('btnLabel');
const btnSpinner    = document.getElementById('btnSpinner');

const errorContainer = document.getElementById('errorContainer');
const errorMessage   = document.getElementById('errorMessage');
const loadingState   = document.getElementById('loadingState');
const loadingTicker  = document.getElementById('loadingTicker');
const resultsSection = document.getElementById('resultsSection');

/* ─── Plotly shared config ─────────────────────────────────────────────── */
const PLOT_BG   = '#141625';
const GRID_CLR  = 'rgba(255,255,255,0.05)';
const FONT_CLR  = '#94a3b8';

const LAYOUT_DEFAULTS = {
  paper_bgcolor: PLOT_BG,
  plot_bgcolor:  PLOT_BG,
  font: { family: 'Inter, sans-serif', color: FONT_CLR, size: 12 },
  xaxis: {
    gridcolor: GRID_CLR, zeroline: false, showline: false,
    tickfont: { size: 11 }, rangeslider: { visible: false },
  },
  yaxis: {
    gridcolor: GRID_CLR, zeroline: false, showline: false,
    tickfont: { size: 11 },
  },
  legend: { orientation: 'h', y: -0.18, x: 0.5, xanchor: 'center',
            font: { size: 11 } },
  margin: { l: 54, r: 18, t: 24, b: 40 },
  hoverlabel: {
    bgcolor: '#1a1d30', bordercolor: '#6366f1',
    font: { family: 'JetBrains Mono, monospace', size: 12, color: '#e2e8f0' },
  },
};

const CONFIG = { displayModeBar: false, responsive: true };

/* ─── Quick-pick chips ─────────────────────────────────────────────────── */
document.querySelectorAll('.quick-chip').forEach(chip => {
  chip.addEventListener('click', () => {
    tickerInput.value = chip.dataset.ticker;
    tickerInput.focus();
  });
});

/* ─── Form submit ──────────────────────────────────────────────────────── */
form.addEventListener('submit', async (e) => {
  e.preventDefault();

  const ticker = tickerInput.value.trim().toUpperCase();
  const period = periodSelect.value;

  if (!ticker) return;

  setLoading(true, ticker);
  hideError();
  hideResults();

  try {
    const res = await fetch('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ ticker, period }),
    });

    const data = await res.json();

    if (!res.ok) {
      showError(data.error || 'An unexpected error occurred. Please try again.');
      return;
    }

    renderResults(data, ticker);

  } catch (err) {
    showError('Network error — could not reach the prediction server. Please try again.');
  } finally {
    setLoading(false);
  }
});

/* ─── UI state helpers ─────────────────────────────────────────────────── */
function setLoading(active, ticker = '') {
  if (active) {
    submitBtn.disabled = true;
    btnLabel.classList.add('d-none');
    btnSpinner.classList.remove('d-none');
    loadingState.style.display = 'block';
    loadingTicker.textContent = `Training LSTM model for ${ticker}…`;
  } else {
    submitBtn.disabled = false;
    btnLabel.classList.remove('d-none');
    btnSpinner.classList.add('d-none');
    loadingState.style.display = 'none';
  }
}

function showError(msg) {
  errorMessage.textContent = msg;
  errorContainer.style.display = 'block';
  errorContainer.scrollIntoView({ behavior: 'smooth', block: 'center' });
}

function hideError() {
  errorContainer.style.display = 'none';
}

function hideResults() {
  resultsSection.style.display = 'none';
}

/* ─── Main render ──────────────────────────────────────────────────────── */
function renderResults(data, ticker) {
  const { stock_info, historical, train_predictions,
          test_predictions, future_predictions, metrics, trend } = data;

  // Stock header
  document.getElementById('headerTicker').textContent = ticker;
  document.getElementById('headerName').textContent   = stock_info.name || ticker;
  document.getElementById('headerMeta').textContent   =
    [stock_info.sector, stock_info.exchange, stock_info.currency]
      .filter(Boolean).join(' · ');

  const trendBadge = document.getElementById('trendBadge');
  const isBullish  = trend.trend === 'Bullish';
  trendBadge.textContent = isBullish ? '▲ Bullish' : '▼ Bearish';
  trendBadge.className   = `trend-badge ${isBullish ? 'bullish' : 'bearish'}`;

  // Metric cards
  document.getElementById('metCurrentPrice').textContent =
    formatPrice(trend.current_price, stock_info.currency);
  document.getElementById('metAccuracy').textContent =
    `${metrics.accuracy_pct}%`;
  document.getElementById('metRmse').textContent =
    formatPrice(metrics.rmse, stock_info.currency);
  document.getElementById('metMape').textContent =
    `${metrics.mape}%`;

  // Charts
  renderMainChart(historical, train_predictions, test_predictions,
                  future_predictions, stock_info.currency);
  renderMAChart(historical, stock_info.currency);

  // Trend analysis grid
  renderTrendGrid(trend, stock_info.currency);

  // Forecast table
  renderForecastTable(future_predictions, trend.current_price, stock_info.currency);

  // Show results
  resultsSection.style.display = 'block';
  resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

/* ─── Main Chart ──────────────────────────────────────────────────────── */
function renderMainChart(historical, trainPred, testPred, futurePred, currency) {
  const traces = [
    {
      x: historical.dates,
      y: historical.prices,
      name: 'Actual Price',
      type: 'scatter',
      mode: 'lines',
      line: { color: '#6366f1', width: 1.5 },
      hovertemplate: `%{x}<br>${currency} %{y:.2f}<extra>Actual</extra>`,
    },
    {
      x: trainPred.dates,
      y: trainPred.prices,
      name: 'Train Fit',
      type: 'scatter',
      mode: 'lines',
      line: { color: '#10b981', width: 1.5, dash: 'dot' },
      hovertemplate: `%{x}<br>${currency} %{y:.2f}<extra>Train Fit</extra>`,
    },
    {
      x: testPred.dates,
      y: testPred.prices,
      name: 'Test Prediction',
      type: 'scatter',
      mode: 'lines',
      line: { color: '#f59e0b', width: 2 },
      hovertemplate: `%{x}<br>${currency} %{y:.2f}<extra>Test Pred</extra>`,
    },
    {
      x: futurePred.dates,
      y: futurePred.prices,
      name: '30-Day Forecast',
      type: 'scatter',
      mode: 'lines+markers',
      line: { color: '#f43f5e', width: 2, dash: 'dash' },
      marker: { size: 4, color: '#f43f5e' },
      hovertemplate: `%{x}<br>${currency} %{y:.2f}<extra>Forecast</extra>`,
    },
  ];

  // Shaded forecast region
  const shapes = [];
  if (futurePred.dates.length > 1) {
    shapes.push({
      type: 'rect',
      x0: futurePred.dates[0],
      x1: futurePred.dates[futurePred.dates.length - 1],
      y0: 0, y1: 1,
      yref: 'paper',
      fillcolor: 'rgba(244,63,94,0.04)',
      line: { width: 0 },
    });
  }

  const layout = {
    ...LAYOUT_DEFAULTS,
    shapes,
    xaxis: { ...LAYOUT_DEFAULTS.xaxis, type: 'date' },
    yaxis: {
      ...LAYOUT_DEFAULTS.yaxis,
      tickprefix: currency === 'INR' ? '₹' : '$',
    },
  };

  Plotly.newPlot('mainChart', traces, layout, CONFIG);
}

/* ─── MA Chart ─────────────────────────────────────────────────────────── */
function renderMAChart(historical, currency) {
  const prices = historical.prices;
  const dates  = historical.dates;

  const ma20 = movingAverage(prices, 20);
  const ma50 = movingAverage(prices, 50);

  const traces = [
    {
      x: dates, y: prices,
      name: 'Close', type: 'scatter', mode: 'lines',
      line: { color: '#6366f1', width: 1 },
      hovertemplate: `%{x}<br>${currency} %{y:.2f}<extra>Close</extra>`,
    },
    {
      x: dates, y: ma20,
      name: 'MA-20', type: 'scatter', mode: 'lines',
      line: { color: '#f59e0b', width: 1.5, dash: 'dot' },
      hovertemplate: `%{x}<br>MA20: %{y:.2f}<extra></extra>`,
    },
    {
      x: dates, y: ma50,
      name: 'MA-50', type: 'scatter', mode: 'lines',
      line: { color: '#10b981', width: 1.5, dash: 'dot' },
      hovertemplate: `%{x}<br>MA50: %{y:.2f}<extra></extra>`,
    },
  ];

  const layout = {
    ...LAYOUT_DEFAULTS,
    xaxis: { ...LAYOUT_DEFAULTS.xaxis, type: 'date' },
    yaxis: { ...LAYOUT_DEFAULTS.yaxis },
    margin: { l: 54, r: 18, t: 16, b: 40 },
  };

  Plotly.newPlot('maChart', traces, layout, CONFIG);
}

/* ─── Trend Grid ──────────────────────────────────────────────────────── */
function renderTrendGrid(trend, currency) {
  const sym = currency === 'INR' ? '₹' : '$';
  const items = [
    { label: 'Current Price',   value: `${sym}${fmt(trend.current_price)}`, cls: 'val-neutral' },
    { label: 'MA 20',           value: trend.ma_20 ? `${sym}${fmt(trend.ma_20)}` : 'N/A', cls: 'val-neutral' },
    { label: 'MA 50',           value: trend.ma_50 ? `${sym}${fmt(trend.ma_50)}` : 'N/A', cls: 'val-neutral' },
    { label: 'RSI (14)',        value: trend.rsi ? trend.rsi.toFixed(1) : 'N/A',
      cls: trend.rsi > 70 ? 'val-negative' : trend.rsi < 30 ? 'val-positive' : 'val-neutral' },
    { label: '30d Change',      value: `${trend.price_change_30d > 0 ? '+' : ''}${fmt(trend.price_change_30d)}%`,
      cls: trend.price_change_30d >= 0 ? 'val-positive' : 'val-negative' },
    { label: 'Volatility (ann.)', value: `${fmt(trend.volatility_pct)}%`, cls: 'val-neutral' },
  ];

  document.getElementById('trendGrid').innerHTML = items.map(i => `
    <div class="trend-item">
      <div class="trend-item-label">${i.label}</div>
      <div class="trend-item-value ${i.cls}">${i.value}</div>
    </div>
  `).join('');
}

/* ─── Forecast Table ──────────────────────────────────────────────────── */
function renderForecastTable(futurePred, currentPrice, currency) {
  const sym = currency === 'INR' ? '₹' : '$';
  const tbody = document.getElementById('forecastTableBody');

  tbody.innerHTML = futurePred.dates.map((date, i) => {
    const price  = futurePred.prices[i];
    const change = ((price - currentPrice) / currentPrice * 100).toFixed(2);
    const isUp   = price >= currentPrice;

    return `
      <tr>
        <td>${i + 1}</td>
        <td>${date}</td>
        <td>${sym}${fmt(price)}</td>
        <td class="${isUp ? 'val-positive' : 'val-negative'}">
          ${isUp ? '+' : ''}${change}%
        </td>
        <td>
          <span class="trend-pill ${isUp ? 'up' : 'down'}">
            ${isUp ? '▲ Up' : '▼ Down'}
          </span>
        </td>
      </tr>
    `;
  }).join('');
}

/* ─── Helpers ─────────────────────────────────────────────────────────── */

/** Simple moving average (returns array same length as data, null for warm-up) */
function movingAverage(data, window) {
  return data.map((_, i) => {
    if (i < window - 1) return null;
    const slice = data.slice(i - window + 1, i + 1);
    return slice.reduce((a, b) => a + b, 0) / window;
  });
}

/** Format a number with two decimal places and thousands separator */
function fmt(n) {
  if (n == null || isNaN(n)) return 'N/A';
  return Number(n).toLocaleString('en-US', {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  });
}

/** Format a price with currency symbol */
function formatPrice(value, currency) {
  const sym = currency === 'INR' ? '₹' : '$';
  return `${sym}${fmt(value)}`;
}
