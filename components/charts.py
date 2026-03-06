"""
components/charts.py
All Plotly chart builders for the three dashboard tabs.
Bug fix: add_vline x must be a numeric timestamp, not a string.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# ── Palette ───────────────────────────────────────────────────────────────────
C_UP     = "#16a34a"
C_DOWN   = "#dc2626"
C_BLUE   = "#2563eb"
C_PURPLE = "#7c3aed"
C_AMBER  = "#d97706"
C_GRAY   = "#9ca3af"
C_LGRAY  = "#e5e7eb"

LAYOUT = dict(
    template="plotly_white",
    hovermode="x unified",
    legend=dict(
        orientation="h", yanchor="bottom", y=1.02,
        xanchor="right", x=1, font=dict(size=11),
    ),
    margin=dict(l=4, r=4, t=48, b=4),
    font=dict(family="Inter, ui-sans-serif, sans-serif", size=12, color="#374151"),
    paper_bgcolor="white",
    plot_bgcolor="#fafafa",
    xaxis=dict(showgrid=False, linecolor=C_LGRAY, tickcolor=C_LGRAY),
    yaxis=dict(gridcolor=C_LGRAY, linecolor=C_LGRAY, tickcolor=C_LGRAY),
)

_CHART_CFG = dict(config={"displayModeBar": False})


def _ts(date) -> int:
    """Convert a pandas Timestamp or date to milliseconds (Plotly epoch)."""
    return int(pd.Timestamp(date).timestamp() * 1000)


# ── Tab 1 – Historical Analysis ───────────────────────────────────────────────

def render_historical(df: pd.DataFrame, ticker: str, show_ma: bool, show_volume: bool) -> None:
    rows    = 2 if show_volume else 1
    heights = [0.72, 0.28] if show_volume else [1.0]

    fig = make_subplots(
        rows=rows, cols=1, shared_xaxes=True,
        row_heights=heights, vertical_spacing=0.04,
    )

    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["Open"], high=df["High"],
        low=df["Low"],   close=df["Close"],
        name="OHLC",
        increasing=dict(line=dict(color=C_UP),   fillcolor=C_UP),
        decreasing=dict(line=dict(color=C_DOWN),  fillcolor=C_DOWN),
    ), row=1, col=1)

    if show_ma:
        for w, c, d in [(20, C_BLUE, "solid"), (50, C_AMBER, "dot")]:
            fig.add_trace(go.Scatter(
                x=df.index, y=df["Close"].rolling(w).mean(),
                name=f"{w}-day MA",
                line=dict(color=c, width=1.6, dash=d),
                opacity=0.85,
            ), row=1, col=1)

    if show_volume:
        colors = [C_UP if c >= o else C_DOWN for c, o in zip(df["Close"], df["Open"])]
        fig.add_trace(go.Bar(
            x=df.index, y=df["Volume"],
            name="Volume",
            marker_color=colors, opacity=0.5, showlegend=False,
        ), row=2, col=1)
        fig.update_yaxes(title_text="Volume", tickformat=".2s", row=2, col=1,
                         gridcolor=C_LGRAY, linecolor=C_LGRAY)

    fig.update_layout(**LAYOUT, title=f"{ticker} — Price History", height=560)
    fig.update_xaxes(rangeslider_visible=False)
    fig.update_yaxes(title_text="Price (USD)", row=1, col=1)

    st.plotly_chart(fig, use_container_width=True, **_CHART_CFG)


# ── Tab 2 – Model Performance ─────────────────────────────────────────────────

def render_prediction(
    df: pd.DataFrame,
    y_test: np.ndarray,
    y_pred: np.ndarray,
    split: int,
    metrics: dict,
    model_name: str,
    ticker: str,
) -> None:
    # KPI row
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("MAE",      f"${metrics['MAE']:.2f}",   help="Mean Absolute Error")
    m2.metric("RMSE",     f"${metrics['RMSE']:.2f}",  help="Root Mean Squared Error")
    m3.metric("R² Score", f"{metrics['R²']:.4f}",     help="Coefficient of determination")
    m4.metric("MAPE",     f"{metrics['MAPE']:.2f}%",  help="Mean Absolute Percentage Error")

    st.markdown("&nbsp;", unsafe_allow_html=True)

    train_idx = df.index[:split]
    test_idx  = df.index[split:]

    # Actual vs Predicted
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=train_idx, y=df["Close"].values[:split],
        name="Training", line=dict(color=C_GRAY, width=1.5), opacity=0.6,
    ))
    fig.add_trace(go.Scatter(
        x=test_idx, y=y_test,
        name="Actual", line=dict(color=C_UP, width=2),
    ))
    fig.add_trace(go.Scatter(
        x=test_idx, y=y_pred,
        name="Predicted", line=dict(color=C_DOWN, width=2, dash="dash"),
    ))

    # Vertical divider between train/test — use numeric timestamp to avoid type error
    split_ts = _ts(train_idx[-1])
    fig.add_shape(type="line", x0=split_ts, x1=split_ts, y0=0, y1=1,
                  xref="x", yref="paper",
                  line=dict(color=C_GRAY, width=1, dash="dot"))
    fig.add_annotation(x=split_ts, y=1, xref="x", yref="paper",
                       text="Train / Test", showarrow=False,
                       font=dict(size=10, color=C_GRAY), yanchor="bottom", xanchor="left")

    fig.update_layout(
        **LAYOUT,
        title=f"{model_name} — Actual vs. Predicted  ({ticker})",
        height=420, yaxis_title="Price (USD)",
    )
    st.plotly_chart(fig, use_container_width=True, **_CHART_CFG)

    # Residuals
    residuals = y_test - y_pred
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=test_idx, y=residuals, mode="lines",
        line=dict(color=C_BLUE, width=1.5),
        fill="tozeroy", fillcolor="rgba(37,99,235,0.08)",
        name="Residual",
    ))
    fig2.add_hline(y=0, line_dash="dot", line_color=C_GRAY, line_width=1)
    fig2.update_layout(
        **LAYOUT,
        title="Prediction Residuals  (Actual − Predicted)",
        height=260, yaxis_title="Residual (USD)", showlegend=False,
    )
    st.plotly_chart(fig2, use_container_width=True, **_CHART_CFG)


# ── Tab 3 – Price Forecast ────────────────────────────────────────────────────

def render_forecast(
    df: pd.DataFrame,
    future_dates: pd.DatetimeIndex,
    future_preds: np.ndarray,
    rmse: float,
    ticker: str,
    forecast_days: int,
) -> None:
    lookback = df["Close"].iloc[-90:]
    upper    = future_preds + rmse
    lower    = future_preds - rmse

    fig = go.Figure()

    # Historical tail
    fig.add_trace(go.Scatter(
        x=lookback.index, y=lookback.values,
        name="Historical", line=dict(color=C_GRAY, width=2),
    ))

    # Confidence band
    fig.add_trace(go.Scatter(
        x=list(future_dates) + list(future_dates[::-1]),
        y=list(upper) + list(lower[::-1]),
        fill="toself",
        fillcolor="rgba(37,99,235,0.08)",
        line=dict(color="rgba(0,0,0,0)"),
        name=f"±1 RMSE  (${rmse:.2f})",
        hoverinfo="skip",
    ))

    # Forecast line
    fig.add_trace(go.Scatter(
        x=future_dates, y=future_preds,
        name="Forecast",
        mode="lines+markers",
        line=dict(color=C_BLUE, width=2.5),
        marker=dict(size=4, color=C_BLUE),
    ))

    # Today divider — use numeric timestamp
    today_ts = _ts(df.index[-1])
    fig.add_shape(type="line", x0=today_ts, x1=today_ts, y0=0, y1=1,
                  xref="x", yref="paper",
                  line=dict(color=C_GRAY, width=1, dash="dot"))
    fig.add_annotation(x=today_ts, y=0.98, xref="x", yref="paper",
                       text="Today", showarrow=False,
                       font=dict(size=10, color=C_GRAY), yanchor="top", xanchor="right")

    fig.update_layout(
        **LAYOUT,
        title=f"{ticker} — {forecast_days}-Day Forward Forecast",
        height=460, yaxis_title="Price (USD)",
    )
    st.plotly_chart(fig, use_container_width=True, **_CHART_CFG)

    # Forecast table
    st.markdown("##### Forecast Table")
    forecast_df = pd.DataFrame({
        "Date":             future_dates.strftime("%Y-%m-%d"),
        "Predicted ($)":    np.round(future_preds, 2),
        "Lower Bound ($)":  np.round(lower, 2),
        "Upper Bound ($)":  np.round(upper, 2),
        "Range Width ($)":  np.round(upper - lower, 2),
    })
    st.dataframe(
        forecast_df.style.format({
            "Predicted ($)":   "${:,.2f}",
            "Lower Bound ($)": "${:,.2f}",
            "Upper Bound ($)": "${:,.2f}",
            "Range Width ($)": "${:,.2f}",
        }),
        use_container_width=True,
        hide_index=True,
    )