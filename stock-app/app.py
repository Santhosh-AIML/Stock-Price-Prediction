"""
AI-Based Stock Price Prediction - Flask Application
Main entry point: handles routes and API endpoints.
"""

import os
import json
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from model import (
    fetch_stock_data,
    linear_regression_predict,
    lstm_predict,
    get_trading_signals,
    calculate_metrics,
)

app = Flask(__name__)
CORS(app)

# ---------------------------------------------------------------------------
# HTML Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    """Render the main dashboard page."""
    return render_template("index.html")


# ---------------------------------------------------------------------------
# API Routes
# ---------------------------------------------------------------------------

@app.route("/api/stock", methods=["POST"])
def get_stock():
    """
    Fetch real-time and historical stock data.
    Body: { ticker: str, period: str }
    Returns: current price, OHLCV, and historical data for Chart.js.
    """
    body = request.get_json(force=True) or {}
    ticker = body.get("ticker", "").strip().upper()
    period = body.get("period", "6mo")

    if not ticker:
        return jsonify({"error": "Ticker symbol is required."}), 400

    try:
        df, info = fetch_stock_data(ticker, period)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 404
    except Exception as exc:
        return jsonify({"error": f"Failed to fetch data: {exc}"}), 500

    # Build historical series for Chart.js
    hist = {
        "dates": df.index.strftime("%Y-%m-%d").tolist(),
        "close": round_list(df["Close"].tolist()),
        "open": round_list(df["Open"].tolist()),
        "high": round_list(df["High"].tolist()),
        "low": round_list(df["Low"].tolist()),
        "volume": [int(v) for v in df["Volume"].tolist()],
    }

    current = {
        "price": safe_round(info.get("currentPrice") or info.get("regularMarketPrice")),
        "open": safe_round(info.get("regularMarketOpen") or info.get("open")),
        "high": safe_round(info.get("regularMarketDayHigh") or info.get("dayHigh")),
        "low": safe_round(info.get("regularMarketDayLow") or info.get("dayLow")),
        "volume": info.get("regularMarketVolume") or info.get("volume"),
        "prev_close": safe_round(info.get("regularMarketPreviousClose") or info.get("previousClose")),
        "name": info.get("longName") or info.get("shortName") or ticker,
        "currency": info.get("currency", "USD"),
        "sector": info.get("sector", "—"),
        "market_cap": info.get("marketCap"),
    }

    signals = get_trading_signals(df)

    return jsonify({"current": current, "history": hist, "signals": signals})


@app.route("/api/predict", methods=["POST"])
def predict():
    """
    Run Linear Regression and LSTM predictions on stock data.
    Body: { ticker: str, period: str, model: "lr" | "lstm" | "both" }
    Returns: predictions, metrics, and chart data.
    """
    body = request.get_json(force=True) or {}
    ticker = body.get("ticker", "").strip().upper()
    period = body.get("period", "1y")
    model_type = body.get("model", "both")

    if not ticker:
        return jsonify({"error": "Ticker symbol is required."}), 400

    try:
        df, _ = fetch_stock_data(ticker, period)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 404
    except Exception as exc:
        return jsonify({"error": f"Failed to fetch data: {exc}"}), 500

    if len(df) < 60:
        return jsonify({"error": "Not enough historical data for prediction (need ≥ 60 days)."}), 422

    result = {"ticker": ticker, "period": period}

    if model_type in ("lr", "both"):
        try:
            lr_result = linear_regression_predict(df)
            result["lr"] = lr_result
        except Exception as exc:
            result["lr"] = {"error": str(exc)}

    if model_type in ("lstm", "both"):
        try:
            lstm_result = lstm_predict(df)
            result["lstm"] = lstm_result
        except Exception as exc:
            result["lstm"] = {"error": str(exc)}

    return jsonify(result)


@app.route("/api/health")
def health():
    """Health check endpoint."""
    return jsonify({"status": "ok"})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def safe_round(value, ndigits=2):
    try:
        return round(float(value), ndigits) if value is not None else None
    except (TypeError, ValueError):
        return None


def round_list(values, ndigits=2):
    return [safe_round(v, ndigits) for v in values]


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    debug = os.environ.get("FLASK_ENV", "production") != "production"
    app.run(host="0.0.0.0", port=port, debug=debug)
