# AI Stock Price Prediction System

A full-stack, AI-powered stock price forecasting web application built with **Python**, **Flask**, **LSTM (TensorFlow/Keras)**, and **Plotly**.

## Features

- **LSTM Neural Network** — Stacked LSTM model for time-series forecasting
- **Real-time Data** — Pulls live stock data from Yahoo Finance via `yfinance`
- **Interactive Charts** — Plotly-powered charts for history, predictions, and moving averages
- **30-Day Forecast** — Recursive future price prediction with confidence table
- **Trend Analysis** — RSI, MA-20, MA-50, annualised volatility, and 30-day price change
- **Model Metrics** — RMSE, MAPE, and directional accuracy displayed prominently
- **Broad Ticker Support** — US stocks (AAPL, TSLA), Indian stocks (TCS.NS, INFY.NS), and more
- **Error Handling** — Graceful errors for invalid symbols or insufficient data
- **Responsive UI** — Bootstrap 5 + custom dark-mode design

## Project Structure

```
stock-prediction/
├── app.py           # Flask routes and prediction pipeline
├── model.py         # LSTM model build, train, evaluate, and forecast
├── stock_data.py    # yfinance data fetching, preprocessing, trend analysis
├── templates/
│   └── index.html   # Main dashboard (Bootstrap 5 + Plotly)
├── static/
│   ├── css/style.css
│   └── js/main.js
├── requirements.txt
├── README.md
└── .gitignore
```

## Stack

| Layer      | Technology                           |
|------------|--------------------------------------|
| Backend    | Python 3.11, Flask 3                 |
| ML/DL      | TensorFlow 2 / Keras (LSTM)          |
| Data       | yfinance, Pandas, NumPy              |
| ML Utils   | Scikit-learn (MinMaxScaler, MSE)     |
| Charts     | Plotly.js (frontend)                 |
| UI         | Bootstrap 5, custom CSS, JavaScript  |

## Local Setup

```bash
# Clone the repo
git clone https://github.com/<you>/stock-ai-predictor
cd stock-ai-predictor/artifacts/stock-prediction

# Install dependencies
pip install -r requirements.txt

# Run the app
python app.py
```

Open `http://localhost:5000` in your browser.

## API Endpoints

| Method | Path          | Description                     |
|--------|---------------|---------------------------------|
| GET    | `/`           | Dashboard UI                    |
| POST   | `/api/predict`| Run LSTM prediction (JSON body) |
| GET    | `/api/health` | Health check                    |

### POST `/api/predict`

**Request body:**
```json
{ "ticker": "AAPL", "period": "2y" }
```

**Response:**
```json
{
  "ticker": "AAPL",
  "stock_info": { "name": "Apple Inc.", "sector": "Technology", ... },
  "historical":        { "dates": [...], "prices": [...] },
  "train_predictions": { "dates": [...], "prices": [...] },
  "test_predictions":  { "dates": [...], "prices": [...] },
  "future_predictions":{ "dates": [...], "prices": [...] },
  "metrics": { "rmse": 3.14, "mape": 1.23, "accuracy_pct": 72.5 },
  "trend": { "current_price": 182.5, "ma_20": 179.2, "rsi": 56.1, ... }
}
```

## Deployment

### Replit
The app reads `PORT` from the environment. It's pre-configured to run on Replit — just hit the **Run** button.

### GitHub / Other
Push the `artifacts/stock-prediction/` folder as a standalone repo and deploy to any Python hosting service (Render, Railway, Fly.io, etc.).

## Disclaimer

> This tool is for **educational purposes only** and does not constitute financial advice.
> Past performance and model predictions are not indicative of future results.
