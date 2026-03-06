"""
utils/data.py
Handles market data retrieval and feature engineering.
"""

import pandas as pd
import yfinance as yf
import streamlit as st


@st.cache_data(ttl=300, show_spinner=False)
def fetch_data(ticker: str, start, end) -> pd.DataFrame:
    """Download OHLCV data from Yahoo Finance and flatten any MultiIndex columns."""
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive technical indicators used as model features:
      - Rolling moving averages (10-day, 50-day)
      - Daily return and 10-day realised volatility
      - Price lags (1, 5, 10 trading days)
    """
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