"""
AI-Based Stock Price Prediction - ML Models
Implements Linear Regression and LSTM for next-day price prediction,
plus moving-average-based Buy/Hold/Sell signal generation.
"""

import os
import warnings
import math
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings("ignore")

# Suppress TensorFlow info/warning logs
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

SAVED_MODELS_DIR = os.path.join(os.path.dirname(__file__), "saved_models")
os.makedirs(SAVED_MODELS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Data Fetching
# ---------------------------------------------------------------------------

VALID_PERIODS = {"1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"}


def fetch_stock_data(ticker: str, period: str = "1y"):
    """
    Download OHLCV data for *ticker* over *period* via yfinance.

    Returns
    -------
    df   : pd.DataFrame  indexed by date
    info : dict          company metadata
    """
    if period not in VALID_PERIODS:
        period = "1y"

    stock = yf.Ticker(ticker)

    # Validate ticker by attempting to fetch info
    try:
        info = stock.info
    except Exception:
        info = {}

    # An invalid ticker returns an info dict with very few keys
    if not info or info.get("trailingPegRatio") is None and info.get("symbol") is None and len(info) < 5:
        # Try fetching history anyway — some valid tickers skip info
        pass

    df = stock.history(period=period, auto_adjust=True)

    if df is None or df.empty:
        raise ValueError(
            f"No data found for '{ticker}'. "
            "Check the ticker symbol (e.g. AAPL, TSLA, TCS.NS, RELIANCE.NS)."
        )

    df.index = pd.to_datetime(df.index).tz_localize(None)
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    return df, info


# ---------------------------------------------------------------------------
# Feature Engineering
# ---------------------------------------------------------------------------

def _add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add technical indicator columns used by the LR model."""
    d = df.copy()
    d["MA5"] = d["Close"].rolling(5).mean()
    d["MA20"] = d["Close"].rolling(20).mean()
    d["MA50"] = d["Close"].rolling(50).mean()
    d["Price_Change"] = d["Close"].pct_change()
    d["Volatility"] = d["Close"].rolling(10).std()
    d["Day"] = np.arange(len(d))
    return d.dropna()


# ---------------------------------------------------------------------------
# Linear Regression
# ---------------------------------------------------------------------------

def linear_regression_predict(df: pd.DataFrame) -> dict:
    """
    Train a Linear Regression model on rolling technical features and
    predict the next-day closing price.

    Returns a dict with:
      - dates, actual, predicted  (for Chart.js)
      - next_day_price
      - metrics: rmse, mae, r2
    """
    d = _add_features(df)

    feature_cols = ["Day", "Open", "High", "Low", "Volume", "MA5", "MA20", "Price_Change", "Volatility"]
    X = d[feature_cols].values
    y = d["Close"].values

    # 80/20 train-test split (no shuffle — preserve time order)
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    scaler_X = MinMaxScaler()
    X_train_s = scaler_X.fit_transform(X_train)
    X_test_s = scaler_X.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_s, y_train)

    y_pred = model.predict(X_test_s)

    # Next-day prediction: extrapolate Day by 1 and use last known OHLCV
    last = d.iloc[-1]
    next_features = np.array([[
        last["Day"] + 1,
        last["Open"],
        last["High"],
        last["Low"],
        last["Volume"],
        last["MA5"],
        last["MA20"],
        last["Price_Change"],
        last["Volatility"],
    ]])
    next_features_s = scaler_X.transform(next_features)
    next_day_price = float(model.predict(next_features_s)[0])

    metrics = calculate_metrics(y_test, y_pred)

    return {
        "dates": d.index[split:].strftime("%Y-%m-%d").tolist(),
        "actual": _round_list(y_test.tolist()),
        "predicted": _round_list(y_pred.tolist()),
        "next_day_price": round(next_day_price, 2),
        "metrics": metrics,
        "model_name": "Linear Regression",
    }


# ---------------------------------------------------------------------------
# LSTM
# ---------------------------------------------------------------------------

LOOK_BACK = 60  # days of history the LSTM looks at


def lstm_predict(df: pd.DataFrame) -> dict:
    """
    Train a two-layer LSTM model on closing prices and predict the
    next-day closing price.

    Returns the same dict shape as `linear_regression_predict`.
    """
    # Lazy import to avoid startup cost when only LR is needed
    try:
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout
        from tensorflow.keras.callbacks import EarlyStopping
        tf.get_logger().setLevel("ERROR")
    except ImportError as exc:
        raise RuntimeError(f"TensorFlow is not installed: {exc}") from exc

    prices = df["Close"].values.reshape(-1, 1)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(prices)

    # Build sequences
    X_seq, y_seq = [], []
    for i in range(LOOK_BACK, len(scaled)):
        X_seq.append(scaled[i - LOOK_BACK:i, 0])
        y_seq.append(scaled[i, 0])

    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)

    split = int(len(X_seq) * 0.8)
    X_train, X_test = X_seq[:split], X_seq[split:]
    y_train, y_test = y_seq[:split], y_seq[split:]

    # Reshape for LSTM: (samples, timesteps, features)
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    # Build LSTM model
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(LOOK_BACK, 1)),
        Dropout(0.2),
        LSTM(64, return_sequences=False),
        Dropout(0.2),
        Dense(32),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mean_squared_error")

    early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    model.fit(
        X_train, y_train,
        epochs=30,
        batch_size=32,
        validation_split=0.1,
        callbacks=[early_stop],
        verbose=0,
    )

    # Predictions on test set
    y_pred_scaled = model.predict(X_test, verbose=0)
    y_pred = scaler.inverse_transform(y_pred_scaled).flatten()
    y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

    # Next-day prediction
    last_sequence = scaled[-LOOK_BACK:].reshape(1, LOOK_BACK, 1)
    next_scaled = model.predict(last_sequence, verbose=0)
    next_day_price = float(scaler.inverse_transform(next_scaled)[0][0])

    metrics = calculate_metrics(y_test_actual, y_pred)

    # Dates corresponding to the test set
    test_dates = df.index[LOOK_BACK + split:LOOK_BACK + split + len(y_test_actual)]

    return {
        "dates": test_dates.strftime("%Y-%m-%d").tolist(),
        "actual": _round_list(y_test_actual.tolist()),
        "predicted": _round_list(y_pred.tolist()),
        "next_day_price": round(next_day_price, 2),
        "metrics": metrics,
        "model_name": "LSTM Neural Network",
    }


# ---------------------------------------------------------------------------
# Trading Signals (Moving Average Crossover)
# ---------------------------------------------------------------------------

def get_trading_signals(df: pd.DataFrame) -> dict:
    """
    Generate Buy / Hold / Sell signals using MA20 × MA50 crossover.
    Also returns the last 90 days of MA data for Chart.js overlay.
    """
    d = df.copy()
    d["MA20"] = d["Close"].rolling(20).mean()
    d["MA50"] = d["Close"].rolling(50).mean()
    d = d.dropna()

    if d.empty:
        return {"signal": "HOLD", "reason": "Not enough data for signal calculation."}

    last = d.iloc[-1]
    prev = d.iloc[-2] if len(d) > 1 else last

    current_signal = "HOLD"
    reason = "MA20 and MA50 are close — no clear trend."

    if last["MA20"] > last["MA50"] and prev["MA20"] <= prev["MA50"]:
        current_signal = "BUY"
        reason = "Golden Cross: MA20 crossed above MA50 — bullish signal."
    elif last["MA20"] < last["MA50"] and prev["MA20"] >= prev["MA50"]:
        current_signal = "SELL"
        reason = "Death Cross: MA20 crossed below MA50 — bearish signal."
    elif last["MA20"] > last["MA50"]:
        current_signal = "BUY"
        reason = "MA20 is above MA50 — uptrend in progress."
    elif last["MA20"] < last["MA50"]:
        current_signal = "SELL"
        reason = "MA20 is below MA50 — downtrend in progress."

    # Return last 90 days for overlay chart
    tail = d.tail(90)
    return {
        "signal": current_signal,
        "reason": reason,
        "ma_dates": tail.index.strftime("%Y-%m-%d").tolist(),
        "ma20": _round_list(tail["MA20"].tolist()),
        "ma50": _round_list(tail["MA50"].tolist()),
        "close": _round_list(tail["Close"].tolist()),
    }


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def calculate_metrics(y_true, y_pred) -> dict:
    """Return RMSE, MAE, and R² score."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {
        "rmse": round(rmse, 4),
        "mae": round(mae, 4),
        "r2": round(r2, 4),
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _round_list(values, ndigits=2):
    return [round(float(v), ndigits) if v is not None and not math.isnan(v) else None for v in values]
