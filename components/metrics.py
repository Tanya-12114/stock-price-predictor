"""
components/metrics.py
KPI strip using native Streamlit metric components.
"""

import pandas as pd
import streamlit as st


def render_metrics(df: pd.DataFrame) -> None:
    current   = float(df["Close"].iloc[-1])
    previous  = float(df["Close"].iloc[-2])
    delta     = current - previous
    delta_pct = (delta / previous) * 100

    period_high  = float(df["High"].max())
    period_low   = float(df["Low"].min())
    avg_volume   = df["Volume"].mean()
    total_return = ((current / float(df["Close"].iloc[0])) - 1) * 100
    volatility   = float(df["Close"].pct_change().std() * (252 ** 0.5) * 100)

    c1, c2, c3, c4, c5, c6, c7 = st.columns(7)

    c1.metric(
        "Last Price",
        f"${current:,.2f}",
        f"{delta:+.2f}  ({delta_pct:+.2f}%)",
    )
    c2.metric("Period High",     f"${period_high:,.2f}")
    c3.metric("Period Low",      f"${period_low:,.2f}")
    c4.metric("Total Return",    f"{total_return:+.1f}%")
    c5.metric("Ann. Volatility", f"{volatility:.1f}%")
    c6.metric("Avg Volume",      f"{avg_volume / 1_000_000:.2f}M")
    c7.metric("Trading Days",    f"{len(df):,}")