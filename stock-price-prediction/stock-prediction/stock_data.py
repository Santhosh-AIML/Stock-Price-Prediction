"""
stock_data.py
-------------
Handles fetching and preprocessing stock market data using yfinance.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import logging

logger = logging.getLogger(__name__)


def fetch_stock_data(ticker: str, period: str = "2y") -> pd.DataFrame:
    """
    Fetch historical stock data for a given ticker symbol.

    Args:
        ticker: Stock ticker symbol (e.g., 'AAPL', 'TSLA', 'TCS.NS')
        period: Data period ('1y', '2y', '5y', 'max')

    Returns:
        DataFrame with OHLCV data

    Raises:
        ValueError: If the ticker is invalid or data is unavailable
    """
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period)

        if df.empty:
            raise ValueError(f"No data found for ticker '{ticker}'. "
                             "Please check the symbol and try again.")

        # Keep only relevant columns
        df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
        df.index = pd.to_datetime(df.index)
        df.index = df.index.tz_localize(None)  # Remove timezone

        logger.info(f"Fetched {len(df)} rows for {ticker}")
        return df

    except Exception as e:
        if "No data found" in str(e) or "ValueError" in type(e).__name__:
            raise
        raise ValueError(f"Failed to fetch data for '{ticker}': {str(e)}")


def get_stock_info(ticker: str) -> dict:
    """
    Get basic info about a stock.

    Args:
        ticker: Stock ticker symbol

    Returns:
        Dictionary with stock metadata
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        return {
            "name": info.get("longName", ticker),
            "sector": info.get("sector", "N/A"),
            "currency": info.get("currency", "USD"),
            "exchange": info.get("exchange", "N/A"),
            "market_cap": info.get("marketCap", None),
            "pe_ratio": info.get("trailingPE", None),
            "52w_high": info.get("fiftyTwoWeekHigh", None),
            "52w_low": info.get("fiftyTwoWeekLow", None),
        }
    except Exception:
        return {"name": ticker, "sector": "N/A", "currency": "USD",
                "exchange": "N/A", "market_cap": None, "pe_ratio": None,
                "52w_high": None, "52w_low": None}


def preprocess_for_lstm(df: pd.DataFrame, sequence_length: int = 60):
    """
    Preprocess stock data for LSTM model training.

    Args:
        df: Raw stock DataFrame
        sequence_length: Number of past time steps to use for prediction

    Returns:
        Tuple of (X_train, X_test, y_train, y_test, scaler, scaled_data)
    """
    # Use closing prices
    close_prices = df["Close"].values.reshape(-1, 1)

    # Scale to [0, 1]
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(close_prices)

    # Split: 80% train, 20% test
    train_size = int(len(scaled_data) * 0.80)
    train_data = scaled_data[:train_size]

    # Build sequences
    X_train, y_train = _build_sequences(train_data, sequence_length)
    X_test, y_test = _build_sequences(scaled_data[train_size - sequence_length:],
                                      sequence_length)

    # Reshape for LSTM: (samples, timesteps, features)
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    return X_train, X_test, y_train, y_test, scaler, scaled_data, train_size


def _build_sequences(data: np.ndarray, seq_len: int):
    """
    Build input/output sequences for time-series modeling.

    Args:
        data: 1D or 2D numpy array of scaled values
        seq_len: Look-back window size

    Returns:
        X (inputs), y (targets) arrays
    """
    X, y = [], []
    for i in range(seq_len, len(data)):
        X.append(data[i - seq_len:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)


def compute_trend_analysis(df: pd.DataFrame) -> dict:
    """
    Compute simple trend indicators (MA, RSI, volatility).

    Args:
        df: Stock DataFrame with 'Close' column

    Returns:
        Dictionary of trend metrics
    """
    close = df["Close"]

    ma_20 = close.rolling(window=20).mean().iloc[-1]
    ma_50 = close.rolling(window=50).mean().iloc[-1]
    current_price = close.iloc[-1]

    # RSI (14-period)
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss
    rsi = (100 - 100 / (1 + rs)).iloc[-1]

    # 20-day volatility (annualised)
    daily_returns = close.pct_change().dropna()
    volatility = daily_returns.std() * np.sqrt(252) * 100

    # 30-day price change %
    price_change_30d = ((current_price - close.iloc[-30]) / close.iloc[-30] * 100
                        if len(close) >= 30 else 0)

    trend_label = "Bullish" if current_price > ma_50 else "Bearish"

    return {
        "current_price": round(float(current_price), 2),
        "ma_20": round(float(ma_20), 2) if not np.isnan(ma_20) else None,
        "ma_50": round(float(ma_50), 2) if not np.isnan(ma_50) else None,
        "rsi": round(float(rsi), 2) if not np.isnan(rsi) else None,
        "volatility_pct": round(float(volatility), 2),
        "price_change_30d": round(float(price_change_30d), 2),
        "trend": trend_label,
    }
