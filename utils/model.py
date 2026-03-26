"""
utils/model.py
Model training, evaluation, and forward-looking forecast generation.
Supports: Random Forest, XGBoost, Linear Regression.
"""

import numpy as np
import pandas as pd
from datetime import timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    from xgboost import XGBRegressor
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False

FEATURE_COLS = ["MA_10", "MA_50", "Return", "Volatility", "Lag_1", "Lag_5", "Lag_10"]


def _build_model(model_choice: str):
    if model_choice == "XGBoost" and _HAS_XGB:
        return XGBRegressor(n_estimators=200, learning_rate=0.05,
                            max_depth=5, random_state=42,
                            verbosity=0, n_jobs=-1)
    elif model_choice == "Linear Regression":
        return LinearRegression()
    else:
        return RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1)


def train_model(df: pd.DataFrame, model_choice: str, test_split_pct: int):
    X = df[FEATURE_COLS].values
    y = df["Close"].values

    split = int(len(X) * (1 - test_split_pct / 100))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    scaler = MinMaxScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    model = _build_model(model_choice)
    model.fit(X_train_s, y_train)
    y_pred = model.predict(X_test_s)

    metrics = {
        "MAE":  mean_absolute_error(y_test, y_pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "R²":   r2_score(y_test, y_pred),
        "MAPE": float(np.mean(np.abs((y_test - y_pred) / y_test)) * 100),
    }

    return model, scaler, y_test, y_pred, split, metrics


def forecast_future(df: pd.DataFrame, model, scaler, forecast_days: int):
    last_rows   = df[FEATURE_COLS].tail(forecast_days).values
    last_scaled = scaler.transform(last_rows)
    preds = model.predict(last_scaled)

    last_date    = df.index[-1]
    future_dates = pd.bdate_range(start=last_date + timedelta(days=1), periods=forecast_days)
    return future_dates, preds