"""
app.py
------
Main Flask application for the AI Stock Price Prediction System.

Routes:
  GET  /           - Serve the dashboard UI
  POST /api/predict - Run LSTM prediction for a given ticker
  GET  /api/health  - Health check
"""

import os
import json
import logging
from datetime import datetime, timedelta

from flask import Flask, render_template, request, jsonify

from stock_data import (
    fetch_stock_data,
    get_stock_info,
    preprocess_for_lstm,
    compute_trend_analysis,
)
from model import build_lstm_model, train_model, evaluate_model, predict_future

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------
app = Flask(__name__)
app.config["JSON_SORT_KEYS"] = False

SEQUENCE_LENGTH = 60   # Days of history used as model input
FUTURE_DAYS = 30       # Days ahead to forecast
EPOCHS = 25            # Training epochs (EarlyStopping may cut this short)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    """Serve the main dashboard page."""
    return render_template("index.html")


@app.route("/api/health")
def health():
    """Simple health-check endpoint."""
    return jsonify({"status": "ok", "timestamp": datetime.utcnow().isoformat()})


@app.route("/api/predict", methods=["POST"])
def predict():
    """
    Run stock prediction pipeline for a given ticker.

    Request JSON body:
        { "ticker": "AAPL", "period": "2y" }

    Response JSON:
        {
          "ticker": str,
          "stock_info": {...},
          "historical": { "dates": [...], "prices": [...] },
          "train_predictions": { "dates": [...], "prices": [...] },
          "test_predictions": { "dates": [...], "prices": [...] },
          "future_predictions": { "dates": [...], "prices": [...] },
          "metrics": { "rmse": float, "mape": float, "accuracy_pct": float },
          "trend": {...}
        }
    """
    data = request.get_json(silent=True) or {}
    ticker = data.get("ticker", "").strip().upper()
    period = data.get("period", "2y")

    # --- Validation ---
    if not ticker:
        return jsonify({"error": "Ticker symbol is required."}), 400

    if period not in ("1y", "2y", "5y", "max"):
        period = "2y"

    logger.info(f"Prediction request: ticker={ticker}, period={period}")

    try:
        # 1. Fetch historical data
        df = fetch_stock_data(ticker, period)

        if len(df) < SEQUENCE_LENGTH + 10:
            return jsonify({
                "error": (f"Not enough data for '{ticker}'. "
                          f"Need at least {SEQUENCE_LENGTH + 10} trading days. "
                          f"Try a longer period.")
            }), 422

        # 2. Stock metadata + trend
        stock_info = get_stock_info(ticker)
        trend = compute_trend_analysis(df)

        # 3. Preprocess for LSTM
        (X_train, X_test, y_train, y_test,
         scaler, scaled_data, train_size) = preprocess_for_lstm(df, SEQUENCE_LENGTH)

        # 4. Build + train model
        model = build_lstm_model(SEQUENCE_LENGTH)
        train_model(model, X_train, y_train, epochs=EPOCHS)

        # 5. Evaluate on test set
        eval_results = evaluate_model(model, X_test, y_test, scaler)

        # 6. Predict future prices
        last_seq = scaled_data[-SEQUENCE_LENGTH:]
        future_prices = predict_future(model, last_seq, scaler, FUTURE_DAYS)

        # 7. Build date arrays
        all_dates = df.index.strftime("%Y-%m-%d").tolist()

        # Train prediction dates (shifted by sequence_length)
        train_pred_dates = all_dates[SEQUENCE_LENGTH:train_size]
        test_pred_dates = all_dates[train_size:]

        # Future dates (business days after last historical date)
        last_date = df.index[-1]
        future_dates = []
        current = last_date
        while len(future_dates) < FUTURE_DAYS:
            current += timedelta(days=1)
            if current.weekday() < 5:  # Mon-Fri
                future_dates.append(current.strftime("%Y-%m-%d"))

        # 8. Train predictions (we predict only test; generate train preds too)
        import numpy as np
        train_preds_scaled = model.predict(X_train, verbose=0)
        train_preds = scaler.inverse_transform(train_preds_scaled).flatten().tolist()
        train_preds = [round(p, 4) for p in train_preds]

        # Ensure arrays align
        min_train = min(len(train_pred_dates), len(train_preds))
        min_test = min(len(test_pred_dates), len(eval_results["predictions"]))

        response = {
            "ticker": ticker,
            "stock_info": stock_info,
            "historical": {
                "dates": all_dates,
                "prices": [round(p, 4) for p in df["Close"].tolist()],
            },
            "train_predictions": {
                "dates": train_pred_dates[:min_train],
                "prices": train_preds[:min_train],
            },
            "test_predictions": {
                "dates": test_pred_dates[:min_test],
                "prices": eval_results["predictions"][:min_test],
            },
            "future_predictions": {
                "dates": future_dates,
                "prices": future_prices,
            },
            "metrics": {
                "rmse": eval_results["rmse"],
                "mape": eval_results["mape"],
                "accuracy_pct": eval_results["accuracy_pct"],
            },
            "trend": trend,
        }

        logger.info(f"Prediction complete for {ticker}.")
        return jsonify(response)

    except ValueError as e:
        logger.warning(f"Validation error for {ticker}: {e}")
        return jsonify({"error": str(e)}), 422
    except Exception as e:
        logger.exception(f"Unexpected error for {ticker}: {e}")
        return jsonify({
            "error": "An internal error occurred during prediction. "
                     "Please try again with a different ticker or period."
        }), 500


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    logger.info(f"Starting Stock Prediction Server on port {port}")
    app.run(host="0.0.0.0", port=port, debug=debug)
