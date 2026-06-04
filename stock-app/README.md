# AI-Based Stock Price Prediction Using Machine Learning

A full-stack web application that fetches real-time stock data, trains Linear Regression and LSTM models, and predicts future stock prices with an interactive dashboard.

## Features

- **Real-time data** via Yahoo Finance API (yfinance)
- **Search** any stock by ticker symbol (AAPL, TSLA, TCS.NS, RELIANCE.NS, etc.)
- **Dashboard** showing current price, OHLCV stats, and interactive price charts
- **Linear Regression** model for price prediction
- **LSTM Neural Network** for sequence-based price prediction
- **Metrics**: RMSE, MAE, R² score
- **Buy/Hold/Sell signals** from moving average crossover (MA20 × MA50)
- **Dark / Light mode** toggle
- **Responsive** Bootstrap 5 design

## Tech Stack

| Layer     | Technology                               |
|-----------|------------------------------------------|
| Backend   | Python 3.11, Flask, Flask-CORS           |
| Data      | yfinance, Pandas, NumPy                  |
| ML        | Scikit-learn (Linear Regression), TensorFlow/Keras (LSTM) |
| Frontend  | Bootstrap 5, Chart.js, Vanilla JS        |

## Project Structure

```
stock-app/
├── app.py              # Flask routes & entry point
├── model.py            # ML models, data fetching, signals
├── requirements.txt
├── README.md
├── templates/
│   ├── base.html       # Base layout with navbar & theme toggle
│   └── index.html      # Main dashboard page
├── static/
│   ├── css/style.css   # Custom styles (dark/light aware)
│   └── js/main.js      # All frontend logic
└── saved_models/       # Auto-created directory for model weights
```

## Setup Instructions

### Replit
The app runs automatically via the configured workflow. Click **Run** and open the preview pane.

### Local (VS Code)

```bash
# 1. Clone / open the project
cd stock-app

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
python app.py
# → Open http://localhost:8000
```

## API Endpoints

| Method | Path            | Description                              |
|--------|-----------------|------------------------------------------|
| GET    | `/`             | Main dashboard HTML                      |
| POST   | `/api/stock`    | Fetch OHLCV + signals for a ticker       |
| POST   | `/api/predict`  | Run ML prediction (LR / LSTM / both)     |
| GET    | `/api/health`   | Health check                             |

### `/api/stock` body
```json
{ "ticker": "AAPL", "period": "6mo" }
```

### `/api/predict` body
```json
{ "ticker": "AAPL", "period": "1y", "model": "both" }
```
`model` can be `"lr"`, `"lstm"`, or `"both"`.

## Disclaimer
This application is for **educational purposes only** and does not constitute financial advice. Always consult a qualified financial advisor before making investment decisions.
