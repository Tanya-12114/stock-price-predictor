"""
components/sidebar.py
Full-featured sidebar with model selection, forecast config, chart toggles, and peer comparison.
"""

import streamlit as st
from datetime import date, timedelta


def render_sidebar() -> dict:
    st.sidebar.title("📈 Stock Predictor")
    st.sidebar.markdown("---")

    # ── Ticker ────────────────────────────────────────────────────────────────
    st.sidebar.markdown("**Stock Symbol**")
    ticker = st.sidebar.text_input("", "AAPL", label_visibility="collapsed").upper().strip()

    # ── Date Range ────────────────────────────────────────────────────────────
    st.sidebar.markdown("**Date Range**")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_date = st.date_input("From", value=date.today() - timedelta(days=730))
    with col2:
        end_date = st.date_input("To", value=date.today())

    st.sidebar.markdown("---")

    # ── Model Settings ────────────────────────────────────────────────────────
    st.sidebar.markdown("**Model Settings**")
    model_choice = st.sidebar.selectbox(
        "Algorithm",
        ["Random Forest", "XGBoost", "Linear Regression"],
        index=0,
    )
    forecast_days = st.sidebar.slider("Forecast Days", min_value=7, max_value=90, value=30, step=7)
    test_split    = st.sidebar.slider("Test Split %", min_value=10, max_value=40, value=20, step=5)

    st.sidebar.markdown("---")

    # ── Chart Toggles ─────────────────────────────────────────────────────────
    st.sidebar.markdown("**Chart Options**")
    show_ma     = st.sidebar.toggle("Moving Averages (20 / 50d)", value=True)
    show_volume = st.sidebar.toggle("Volume Bars", value=True)
    show_rsi    = st.sidebar.toggle("RSI Indicator", value=True)
    show_macd   = st.sidebar.toggle("MACD Indicator", value=False)
    show_bb     = st.sidebar.toggle("Bollinger Bands", value=False)

    st.sidebar.markdown("---")

    # ── Peer Comparison ───────────────────────────────────────────────────────
    st.sidebar.markdown("**Peer Comparison**")
    peers_raw = st.sidebar.text_input(
        "Compare with (comma-separated)",
        placeholder="e.g. MSFT, GOOGL, AMZN",
        value="",
    )
    peers = [p.strip().upper() for p in peers_raw.split(",") if p.strip()] if peers_raw else []

    return {
        "ticker":        ticker,
        "start_date":    start_date,
        "end_date":      end_date,
        "model_choice":  model_choice,
        "forecast_days": forecast_days,
        "test_split":    test_split,
        "show_ma":       show_ma,
        "show_volume":   show_volume,
        "show_rsi":      show_rsi,
        "show_macd":     show_macd,
        "show_bb":       show_bb,
        "peers":         peers,
    }