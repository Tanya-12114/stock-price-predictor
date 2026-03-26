"""
utils/data.py
Market data retrieval, feature engineering, and technical indicators.
"""

import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st


@st.cache_data(ttl=300, show_spinner=False)
def fetch_data(ticker: str, start, end) -> pd.DataFrame:
    """Download OHLCV data from Yahoo Finance."""
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


@st.cache_data(ttl=300, show_spinner=False)
def fetch_peers(tickers: list[str], start, end) -> dict[str, pd.DataFrame]:
    """Fetch closing prices for a list of peer tickers."""
    result = {}
    for t in tickers:
        try:
            df = yf.download(t, start=start, end=end, progress=False, auto_adjust=True)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if not df.empty:
                result[t] = df
        except Exception:
            pass
    return result


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive technical indicators used as model features."""
    df = df.copy()
    close = df["Close"]

    df["MA_10"]      = close.rolling(10).mean()
    df["MA_50"]      = close.rolling(50).mean()
    df["Return"]     = close.pct_change()
    df["Volatility"] = df["Return"].rolling(10).std()
    df["Lag_1"]      = close.shift(1)
    df["Lag_5"]      = close.shift(5)
    df["Lag_10"]     = close.shift(10)

    df.dropna(inplace=True)
    return df


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def compute_macd(series: pd.Series, fast=12, slow=26, signal=9):
    ema_fast   = series.ewm(span=fast, adjust=False).mean()
    ema_slow   = series.ewm(span=slow, adjust=False).mean()
    macd_line  = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram  = macd_line - signal_line
    return macd_line, signal_line, histogram

def compute_bollinger(series: pd.Series, period: int = 20, std: float = 2.0):
    """Returns (upper_band, middle_band, lower_band)."""
    mid   = series.rolling(period).mean()
    sigma = series.rolling(period).std()
    return mid + std * sigma, mid, mid - std * sigma