"""
Stock Price Prediction Dashboard — Entry Point
"""

import streamlit as st
from utils.data import fetch_data, add_features
from utils.model import train_model, forecast_future
from components.sidebar import render_sidebar
from components.metrics import render_metrics
from components.charts import render_historical, render_prediction, render_forecast

st.set_page_config(
    page_title="Stock Price Predictor",
    page_icon="https://img.freepik.com/premium-vector/market-forecast-icon-3d-illustration-from-finance-management-collection-creative-market-forecast-3d-icon-web-design-templates-infographics-more_676904-812.jpg",
    layout="wide",
    initial_sidebar_state="expanded",
)

with open("assets/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
cfg = render_sidebar()

# ── Data ──────────────────────────────────────────────────────────────────────
with st.spinner(f"Loading {cfg['ticker']}…"):
    df_raw = fetch_data(cfg["ticker"], cfg["start_date"], cfg["end_date"])

if df_raw.empty:
    st.error(f"No data found for **{cfg['ticker']}**. Please check the ticker symbol.")
    st.stop()

# ── Page Header ───────────────────────────────────────────────────────────────
current = float(df_raw["Close"].iloc[-1])
prev    = float(df_raw["Close"].iloc[-2])
delta   = current - prev
pct     = (delta / prev) * 100
clr     = "#16a34a" if delta >= 0 else "#dc2626"
arrow   = "▲" if delta >= 0 else "▼"

left_col, right_col = st.columns([3, 1])
with left_col:
    st.markdown(f"## {cfg['ticker']}")
    st.caption("Stock Price Analysis & ML Forecast")
with right_col:
    st.markdown(
        f"""<div style="text-align:right; padding-top:8px;">
            <div style="font-size:1.6rem; font-weight:700; color:#111827;">${current:,.2f}</div>
            <div style="font-size:0.85rem; font-weight:600; color:{clr};">{arrow} ${abs(delta):.2f} ({pct:+.2f}%)</div>
        </div>""",
        unsafe_allow_html=True,
    )

st.divider()

# ── KPI Metrics ───────────────────────────────────────────────────────────────
render_metrics(df_raw)
st.divider()

# ── Model ─────────────────────────────────────────────────────────────────────
df_feat = add_features(df_raw)

with st.spinner("Training model…"):
    model, scaler, y_test, y_pred, split, metrics = train_model(
        df_feat,
        cfg["model_choice"],
        cfg["test_split"]
    )

future_dates, future_preds = forecast_future(df_feat, model, scaler, cfg["forecast_days"])

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["  Historical Analysis  ", "  Model Performance  ", "  Price Forecast  "])

with tab1:
    render_historical(df_raw, cfg["ticker"], cfg["show_ma"], cfg["show_volume"])

with tab2:
    render_prediction(df_feat, y_test, y_pred, split, metrics, cfg["model_choice"], cfg["ticker"])

with tab3:
    render_forecast(df_feat, future_dates, future_preds, metrics["RMSE"], cfg["ticker"], cfg["forecast_days"])

st.markdown(
    "<p style='text-align:center; color:#9ca3af; font-size:0.75rem; padding-top:1rem;'>"
    "</p>",
    unsafe_allow_html=True,
)