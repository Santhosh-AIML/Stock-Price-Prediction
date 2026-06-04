"""
model.py
--------
Defines and trains the LSTM-based stock price prediction model
using TensorFlow / Keras.
"""

import numpy as np
from sklearn.metrics import mean_squared_error
import math
import logging
import os

# Suppress TensorFlow info/warning messages
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

logger = logging.getLogger(__name__)


def build_lstm_model(sequence_length: int = 60):
    """
    Build a stacked LSTM model for time-series regression.

    Architecture:
        LSTM(128) -> Dropout(0.2)
        LSTM(64)  -> Dropout(0.2)
        Dense(32) -> Dense(1)

    Args:
        sequence_length: Number of past time steps used as input

    Returns:
        Compiled Keras model
    """
    # Lazy import to speed up module loading
    from tensorflow import keras
    from tensorflow.keras import layers

    model = keras.Sequential([
        # First LSTM layer — return sequences for stacking
        layers.LSTM(128, return_sequences=True,
                    input_shape=(sequence_length, 1)),
        layers.Dropout(0.2),

        # Second LSTM layer
        layers.LSTM(64, return_sequences=False),
        layers.Dropout(0.2),

        # Fully-connected head
        layers.Dense(32, activation="relu"),
        layers.Dense(1),
    ])

    model.compile(optimizer="adam", loss="mean_squared_error")
    logger.info("LSTM model built successfully.")
    return model


def train_model(model, X_train, y_train,
                epochs: int = 20, batch_size: int = 32,
                validation_split: float = 0.1):
    """
    Train the LSTM model.

    Args:
        model: Compiled Keras model
        X_train: Training inputs  (samples, timesteps, 1)
        y_train: Training targets (samples,)
        epochs: Number of training epochs
        batch_size: Mini-batch size
        validation_split: Fraction of training data used for validation

    Returns:
        Keras History object
    """
    from tensorflow import keras

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
        ),
    ]

    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        callbacks=callbacks,
        verbose=0,
    )

    logger.info(f"Training complete. Epochs run: {len(history.history['loss'])}")
    return history


def evaluate_model(model, X_test, y_test, scaler):
    """
    Generate predictions on the test set and compute error metrics.

    Args:
        model: Trained Keras model
        X_test: Test inputs
        y_test: True test targets (scaled)
        scaler: Fitted MinMaxScaler (to inverse-transform predictions)

    Returns:
        dict with 'predictions', 'actual', 'rmse', 'mape', 'accuracy_pct'
    """
    # Raw scaled predictions
    predictions_scaled = model.predict(X_test, verbose=0)

    # Inverse transform back to original price scale
    predictions = scaler.inverse_transform(predictions_scaled)
    actual = scaler.inverse_transform(y_test.reshape(-1, 1))

    # RMSE
    rmse = math.sqrt(mean_squared_error(actual, predictions))

    # MAPE (Mean Absolute Percentage Error)
    mape = np.mean(np.abs((actual - predictions) / actual)) * 100

    # Directional accuracy (did we predict the right direction?)
    actual_dir = np.diff(actual.flatten())
    pred_dir = np.diff(predictions.flatten())
    directional_accuracy = np.mean(
        np.sign(actual_dir) == np.sign(pred_dir)
    ) * 100

    logger.info(f"RMSE={rmse:.4f}, MAPE={mape:.2f}%, "
                f"Directional Accuracy={directional_accuracy:.1f}%")

    return {
        "predictions": predictions.flatten().tolist(),
        "actual": actual.flatten().tolist(),
        "rmse": round(rmse, 4),
        "mape": round(float(mape), 2),
        "accuracy_pct": round(float(directional_accuracy), 1),
    }


def predict_future(model, last_sequence: np.ndarray,
                   scaler, n_days: int = 30) -> list:
    """
    Predict n_days into the future using recursive forecasting.

    Args:
        model: Trained Keras model
        last_sequence: Last known sequence of scaled prices, shape (seq_len, 1)
        scaler: Fitted MinMaxScaler
        n_days: Number of future days to predict

    Returns:
        List of predicted prices (original scale)
    """
    current_seq = last_sequence.copy()  # (seq_len, 1)
    future_predictions = []

    for _ in range(n_days):
        # Reshape for model: (1, seq_len, 1)
        x_input = current_seq.reshape(1, current_seq.shape[0], 1)
        pred_scaled = model.predict(x_input, verbose=0)[0, 0]

        # Store the prediction
        future_predictions.append(pred_scaled)

        # Roll the window forward
        current_seq = np.roll(current_seq, -1, axis=0)
        current_seq[-1, 0] = pred_scaled

    # Inverse transform to original price scale
    future_prices = scaler.inverse_transform(
        np.array(future_predictions).reshape(-1, 1)
    ).flatten().tolist()

    return [round(p, 4) for p in future_prices]
