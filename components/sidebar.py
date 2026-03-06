import streamlit as st
from datetime import date

def render_sidebar():
    st.sidebar.title("Stock Predictor")

    ticker = st.sidebar.text_input("Ticker Symbol", "AAPL")

    col1, col2 = st.sidebar.columns(2)

    with col1:
        start_date = st.date_input(
            "From",
            value=date(2024, 2, 25)
        )

    with col2:
        end_date = st.date_input(
            "To",
            value=date.today()
        )

    # ── Hidden Defaults (Still Used Internally) ───────────────────────────
    cfg = {
        "ticker": ticker,
        "start_date": start_date,
        "end_date": end_date,
        "model_choice": "Random Forest",
        "forecast_days": 30,
        "test_split": 0.2,
        "show_ma": False,
        "show_volume": False,
    }

    return cfg
