"""
components/charts.py
All Plotly chart builders for the dashboard tabs.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from utils.data import compute_rsi, compute_macd, compute_bollinger

# ── Palette ───────────────────────────────────────────────────────────────────
C_UP     = "#16a34a"
C_DOWN   = "#dc2626"
C_BLUE   = "#2563eb"
C_PURPLE = "#7c3aed"
C_AMBER  = "#d97706"
C_GRAY   = "#9ca3af"
C_LGRAY  = "#e5e7eb"

PEER_COLORS = ["#7c3aed", "#d97706", "#0891b2", "#db2777", "#059669"]

LAYOUT = dict(
    template="plotly_white",
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=11)),
    margin=dict(l=4, r=4, t=48, b=4),
    font=dict(family="Inter, ui-sans-serif, sans-serif", size=12, color="#374151"),
    paper_bgcolor="white",
    plot_bgcolor="#fafafa",
    xaxis=dict(showgrid=False, linecolor=C_LGRAY, tickcolor=C_LGRAY),
    yaxis=dict(gridcolor=C_LGRAY, linecolor=C_LGRAY, tickcolor=C_LGRAY),
)

_CHART_CFG = dict(config={"displayModeBar": False})


def _ts(date) -> int:
    return int(pd.Timestamp(date).timestamp() * 1000)


# ── Tab 1 – Historical Analysis ───────────────────────────────────────────────

def render_historical(
    df: pd.DataFrame,
    ticker: str,
    show_ma: bool,
    show_volume: bool,
    show_rsi: bool,
    show_macd: bool,
    show_bb: bool = False,
) -> None:

    # Decide subplot layout
    extra_rows = sum([show_volume, show_rsi, show_macd])
    total_rows = 1 + extra_rows
    row_heights = [0.55] if total_rows > 1 else [1.0]
    if show_volume: row_heights.append(0.15)
    if show_rsi:    row_heights.append(0.15)
    if show_macd:   row_heights.append(0.15)
    # normalise
    s = sum(row_heights)
    row_heights = [r / s for r in row_heights]

    fig = make_subplots(
        rows=total_rows, cols=1, shared_xaxes=True,
        row_heights=row_heights, vertical_spacing=0.03,
    )

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
        name="OHLC",
        increasing=dict(line=dict(color=C_UP),  fillcolor=C_UP),
        decreasing=dict(line=dict(color=C_DOWN), fillcolor=C_DOWN),
    ), row=1, col=1)

    if show_ma:
        for w, c, d in [(20, C_BLUE, "solid"), (50, C_AMBER, "dot")]:
            fig.add_trace(go.Scatter(
                x=df.index, y=df["Close"].rolling(w).mean(),
                name=f"{w}-day MA", line=dict(color=c, width=1.6, dash=d), opacity=0.85,
            ), row=1, col=1)

    if show_bb:
        bb_upper, bb_mid, bb_lower = compute_bollinger(df["Close"])
        fig.add_trace(go.Scatter(
            x=df.index, y=bb_upper, name="BB Upper",
            line=dict(color="#0891b2", width=1, dash="dot"), opacity=0.7,
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=df.index, y=bb_lower, name="BB Lower",
            line=dict(color="#0891b2", width=1, dash="dot"), opacity=0.7,
            fill="tonexty", fillcolor="rgba(8,145,178,0.05)",
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=df.index, y=bb_mid, name="BB Mid",
            line=dict(color="#0891b2", width=1), opacity=0.5,
        ), row=1, col=1)

    current_row = 2

    if show_volume:
        colors = [C_UP if c >= o else C_DOWN for c, o in zip(df["Close"], df["Open"])]
        fig.add_trace(go.Bar(
            x=df.index, y=df["Volume"], name="Volume",
            marker_color=colors, opacity=0.5, showlegend=False,
        ), row=current_row, col=1)
        fig.update_yaxes(title_text="Volume", tickformat=".2s", row=current_row, col=1,
                         gridcolor=C_LGRAY, linecolor=C_LGRAY)
        current_row += 1

    if show_rsi:
        rsi = compute_rsi(df["Close"])
        fig.add_trace(go.Scatter(
            x=df.index, y=rsi, name="RSI(14)",
            line=dict(color=C_PURPLE, width=1.5), showlegend=True,
        ), row=current_row, col=1)
        fig.add_hline(y=70, line_dash="dot", line_color=C_DOWN,   line_width=1, row=current_row, col=1)
        fig.add_hline(y=30, line_dash="dot", line_color=C_UP,     line_width=1, row=current_row, col=1)
        fig.update_yaxes(title_text="RSI", range=[0, 100], row=current_row, col=1,
                         gridcolor=C_LGRAY, linecolor=C_LGRAY)
        current_row += 1

    if show_macd:
        macd_line, signal_line, histogram = compute_macd(df["Close"])
        colors_hist = [C_UP if v >= 0 else C_DOWN for v in histogram]
        fig.add_trace(go.Bar(
            x=df.index, y=histogram, name="MACD Hist",
            marker_color=colors_hist, opacity=0.6, showlegend=False,
        ), row=current_row, col=1)
        fig.add_trace(go.Scatter(
            x=df.index, y=macd_line,   name="MACD",   line=dict(color=C_BLUE,  width=1.4),
        ), row=current_row, col=1)
        fig.add_trace(go.Scatter(
            x=df.index, y=signal_line, name="Signal", line=dict(color=C_AMBER, width=1.4, dash="dot"),
        ), row=current_row, col=1)
        fig.update_yaxes(title_text="MACD", row=current_row, col=1,
                         gridcolor=C_LGRAY, linecolor=C_LGRAY)

    chart_height = 420 + extra_rows * 120
    fig.update_layout(**LAYOUT, title=f"{ticker} — Price History", height=chart_height)
    fig.update_xaxes(rangeslider_visible=False)
    fig.update_yaxes(title_text="Price (USD)", row=1, col=1)
    st.plotly_chart(fig, use_container_width=True, **_CHART_CFG)

    # ── Returns Heatmap ───────────────────────────────────────────────────────
    st.markdown("##### Monthly Returns Heatmap")
    monthly = df["Close"].resample("ME").last().pct_change().dropna() * 100
    monthly_df = monthly.to_frame("Return")
    monthly_df["Year"]  = monthly_df.index.year
    monthly_df["Month"] = monthly_df.index.strftime("%b")

    years  = sorted(monthly_df["Year"].unique())
    months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

    z_data  = []
    text_data = []
    for yr in years:
        row_z    = []
        row_text = []
        for mo in months:
            val = monthly_df[(monthly_df["Year"] == yr) & (monthly_df["Month"] == mo)]["Return"]
            if len(val):
                v = float(val.iloc[0])
                row_z.append(v)
                row_text.append(f"{v:+.1f}%")
            else:
                row_z.append(None)
                row_text.append("")
        z_data.append(row_z)
        text_data.append(row_text)

    hmap = go.Figure(go.Heatmap(
        z=z_data, x=months, y=[str(y) for y in years],
        text=text_data, texttemplate="%{text}",
        colorscale=[[0, "#dc2626"],[0.5, "#f9fafb"],[1, "#16a34a"]],
        zmid=0, showscale=True,
        colorbar=dict(title="Return %", thickness=14, len=0.8),
        hovertemplate="<b>%{y} %{x}</b><br>Return: %{text}<extra></extra>",
    ))
    hmap.update_layout(
        **{k: v for k, v in LAYOUT.items() if k not in ("xaxis", "yaxis", "margin")},
        title=f"{ticker} — Monthly Returns (%)",
        height=max(200, len(years) * 38 + 80),
        margin=dict(l=4, r=4, t=48, b=4),
        xaxis=dict(side="top"),
    )
    st.plotly_chart(hmap, use_container_width=True, **_CHART_CFG)

    # ── Price Cards ───────────────────────────────────────────────────────────
    latest       = float(df["Close"].iloc[-1])
    earliest     = float(df["Close"].iloc[0])
    period_high  = float(df["High"].max())
    period_low   = float(df["Low"].min())
    total_return = ((latest / earliest) - 1) * 100

    arrow   = "▲" if total_return >= 0 else "▼"
    t_color = C_UP if total_return >= 0 else C_DOWN
    t_bg    = "#f0fdf4" if total_return >= 0 else "#fef2f2"
    t_label = "OVERALL GAIN" if total_return >= 0 else "OVERALL LOSS"

    st.markdown(f"""
    <div class="price-grid">
      <div class="price-card" style="background:{t_bg}; border-color:{t_color}30;">
        <div class="pc-label">{t_label}</div>
        <div class="pc-value" style="color:{t_color};">{arrow} {abs(total_return):.2f}%</div>
        <div class="pc-sub" style="color:{t_color};">${earliest:,.2f} → ${latest:,.2f}</div>
      </div>
      <div class="price-card">
        <div class="pc-label">Current Price</div>
        <div class="pc-value">${latest:,.2f}</div>
        <div class="pc-sub" style="color:#64748b;">Latest Close</div>
      </div>
      <div class="price-card" style="background:#f0fdf4; border-color:#16a34a30;">
        <div class="pc-label">Period High</div>
        <div class="pc-value" style="color:#16a34a;">↑ ${period_high:,.2f}</div>
        <div class="pc-sub" style="color:#16a34a;">+{((period_high-earliest)/earliest*100):.1f}% from start</div>
      </div>
      <div class="price-card" style="background:#fef2f2; border-color:#dc262630;">
        <div class="pc-label">Period Low</div>
        <div class="pc-value" style="color:#dc2626;">↓ ${period_low:,.2f}</div>
        <div class="pc-sub" style="color:#dc2626;">{((period_low-earliest)/earliest*100):.1f}% from start</div>
      </div>
    </div>
    """, unsafe_allow_html=True)


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
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("MAE",      f"${metrics['MAE']:.2f}",  help="Mean Absolute Error")
    m2.metric("RMSE",     f"${metrics['RMSE']:.2f}", help="Root Mean Squared Error")
    m3.metric("R² Score", f"{metrics['R²']:.4f}",    help="Coefficient of determination")
    m4.metric("MAPE",     f"{metrics['MAPE']:.2f}%", help="Mean Absolute Percentage Error")

    st.markdown("&nbsp;", unsafe_allow_html=True)

    train_idx = df.index[:split]
    test_idx  = df.index[split:]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=train_idx, y=df["Close"].values[:split],
        name="Training", line=dict(color=C_GRAY, width=1.5), opacity=0.6))
    fig.add_trace(go.Scatter(x=test_idx, y=y_test,
        name="Actual",    line=dict(color=C_UP,   width=2)))
    fig.add_trace(go.Scatter(x=test_idx, y=y_pred,
        name="Predicted", line=dict(color=C_DOWN, width=2, dash="dash")))

    split_ts = _ts(train_idx[-1])
    fig.add_shape(type="line", x0=split_ts, x1=split_ts, y0=0, y1=1,
                  xref="x", yref="paper", line=dict(color=C_GRAY, width=1, dash="dot"))
    fig.add_annotation(x=split_ts, y=1, xref="x", yref="paper",
                       text="Train / Test", showarrow=False,
                       font=dict(size=10, color=C_GRAY), yanchor="bottom", xanchor="left")
    fig.update_layout(**LAYOUT, title=f"{model_name} — Actual vs. Predicted  ({ticker})",
                      height=420, yaxis_title="Price (USD)")
    st.plotly_chart(fig, use_container_width=True, **_CHART_CFG)

    residuals = y_test - y_pred
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=test_idx, y=residuals, mode="lines",
        line=dict(color=C_BLUE, width=1.5),
        fill="tozeroy", fillcolor="rgba(37,99,235,0.08)", name="Residual"))
    fig2.add_hline(y=0, line_dash="dot", line_color=C_GRAY, line_width=1)
    fig2.update_layout(**LAYOUT, title="Prediction Residuals  (Actual − Predicted)",
                       height=260, yaxis_title="Residual (USD)", showlegend=False)
    st.plotly_chart(fig2, use_container_width=True, **_CHART_CFG)

    # Price cards
    last_actual = float(y_test[-1])
    last_pred_v = float(y_pred[-1])
    diff        = last_pred_v - last_actual
    diff_pct    = (diff / last_actual) * 100
    max_actual  = float(y_test.max())
    min_actual  = float(y_test.min())

    d_color = C_UP if diff >= 0 else C_DOWN
    d_bg    = "#f0fdf4" if diff >= 0 else "#fef2f2"
    d_arrow = "▲" if diff >= 0 else "▼"
    d_label = "MODEL ABOVE ACTUAL" if diff >= 0 else "MODEL BELOW ACTUAL"

    st.markdown(f"""
    <div class="price-grid">
      <div class="price-card" style="background:{d_bg}; border-color:{d_color}30;">
        <div class="pc-label">{d_label}</div>
        <div class="pc-value" style="color:{d_color};">{d_arrow} ${abs(diff):.2f}</div>
        <div class="pc-sub" style="color:{d_color};">{diff_pct:+.2f}% difference</div>
      </div>
      <div class="price-card">
        <div class="pc-label">Last Actual Price</div>
        <div class="pc-value">${last_actual:,.2f}</div>
        <div class="pc-sub" style="color:#64748b;">Test set end</div>
      </div>
      <div class="price-card">
        <div class="pc-label">Last Predicted Price</div>
        <div class="pc-value">${last_pred_v:,.2f}</div>
        <div class="pc-sub" style="color:#64748b;">Model output</div>
      </div>
      <div class="price-card" style="background:#fffbeb; border-color:#d9770630;">
        <div class="pc-label">Test Range</div>
        <div class="pc-value" style="color:#d97706;">${min_actual:,.0f}–${max_actual:,.0f}</div>
        <div class="pc-sub" style="color:#d97706;">Low → High in test period</div>
      </div>
    </div>
    """, unsafe_allow_html=True)


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
    fig.add_trace(go.Scatter(x=lookback.index, y=lookback.values,
        name="Historical", line=dict(color=C_GRAY, width=2)))
    fig.add_trace(go.Scatter(
        x=list(future_dates) + list(future_dates[::-1]),
        y=list(upper) + list(lower[::-1]),
        fill="toself", fillcolor="rgba(37,99,235,0.08)",
        line=dict(color="rgba(0,0,0,0)"),
        name=f"±1 RMSE  (${rmse:.2f})", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=future_dates, y=future_preds,
        name="Forecast", mode="lines+markers",
        line=dict(color=C_BLUE, width=2.5), marker=dict(size=4, color=C_BLUE)))

    today_ts = _ts(df.index[-1])
    fig.add_shape(type="line", x0=today_ts, x1=today_ts, y0=0, y1=1,
                  xref="x", yref="paper", line=dict(color=C_GRAY, width=1, dash="dot"))
    fig.add_annotation(x=today_ts, y=0.98, xref="x", yref="paper",
                       text="Today", showarrow=False,
                       font=dict(size=10, color=C_GRAY), yanchor="top", xanchor="right")
    fig.update_layout(**LAYOUT, title=f"{ticker} — {forecast_days}-Day Forward Forecast",
                      height=460, yaxis_title="Price (USD)")
    st.plotly_chart(fig, use_container_width=True, **_CHART_CFG)

    # Price cards
    last_price  = float(df["Close"].iloc[-1])
    first_pred  = float(future_preds[0])
    last_pred   = float(future_preds[-1])
    pred_change = ((last_pred - last_price) / last_price) * 100
    next_change = ((first_pred - last_price) / last_price) * 100

    f_color = C_UP if pred_change >= 0 else C_DOWN
    f_bg    = "#f0fdf4" if pred_change >= 0 else "#fef2f2"
    f_arrow = "▲" if pred_change >= 0 else "▼"
    n_color = C_UP if next_change >= 0 else C_DOWN
    n_bg    = "#f0fdf4" if next_change >= 0 else "#fef2f2"

    st.markdown(f"""
    <div class="price-grid">
      <div class="price-card" style="background:{f_bg}; border-color:{f_color}30;">
        <div class="pc-label">{forecast_days}-DAY FORECAST TREND</div>
        <div class="pc-value" style="color:{f_color};">{f_arrow} {abs(pred_change):.2f}%</div>
        <div class="pc-sub" style="color:{f_color};">${last_price:,.2f} → ${last_pred:,.2f}</div>
      </div>
      <div class="price-card">
        <div class="pc-label">Today's Price</div>
        <div class="pc-value">${last_price:,.2f}</div>
        <div class="pc-sub" style="color:#64748b;">Current close</div>
      </div>
      <div class="price-card" style="background:{n_bg}; border-color:{n_color}30;">
        <div class="pc-label">Next Session</div>
        <div class="pc-value" style="color:{n_color};">${first_pred:,.2f}</div>
        <div class="pc-sub" style="color:{n_color};">{next_change:+.2f}% from today</div>
      </div>
      <div class="price-card">
        <div class="pc-label">End of Forecast</div>
        <div class="pc-value">${last_pred:,.2f}</div>
        <div class="pc-sub" style="color:#64748b;">Day {forecast_days} target</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Forecast table + CSV export
    st.markdown("##### Forecast Table")
    forecast_df = pd.DataFrame({
        "Date":            future_dates.strftime("%Y-%m-%d"),
        "Predicted ($)":   np.round(future_preds, 2),
        "Lower Bound ($)": np.round(lower, 2),
        "Upper Bound ($)": np.round(upper, 2),
        "Range Width ($)": np.round(upper - lower, 2),
    })
    st.dataframe(
        forecast_df.style.format({
            "Predicted ($)":   "${:,.2f}",
            "Lower Bound ($)": "${:,.2f}",
            "Upper Bound ($)": "${:,.2f}",
            "Range Width ($)": "${:,.2f}",
        }),
        use_container_width=True, hide_index=True,
    )

    csv_bytes = forecast_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇ Download Forecast CSV",
        data=csv_bytes,
        file_name=f"{ticker}_forecast_{forecast_days}d.csv",
        mime="text/csv",
    )


# ── Tab 4 – Peer Comparison ───────────────────────────────────────────────────

def render_peers(
    df_main: pd.DataFrame,
    ticker: str,
    peers_data: dict,
) -> None:
    if not peers_data:
        st.info("Enter ticker symbols in the sidebar under **Peer Comparison** to compare stocks.")
        return

    all_colors = [C_BLUE] + PEER_COLORS[:len(peers_data)]

    # ── Normalised Price (rebased to 100) ─────────────────────────────────────
    fig = go.Figure()
    base = float(df_main["Close"].iloc[0])
    fig.add_trace(go.Scatter(
        x=df_main.index,
        y=(df_main["Close"] / base) * 100,
        name=ticker, line=dict(color=C_BLUE, width=2.5),
    ))
    for (t, df_p), col in zip(peers_data.items(), PEER_COLORS):
        base_p = float(df_p["Close"].iloc[0])
        fig.add_trace(go.Scatter(
            x=df_p.index,
            y=(df_p["Close"] / base_p) * 100,
            name=t, line=dict(color=col, width=2),
        ))
    fig.update_layout(**LAYOUT,
                      title="Normalised Price Performance (Base = 100)",
                      height=420, yaxis_title="Indexed Price")
    st.plotly_chart(fig, use_container_width=True, **_CHART_CFG)

    # ── Rolling 30-Day Correlation ────────────────────────────────────────────
    st.markdown("##### Rolling 30-Day Return Correlation")
    ret_main = df_main["Close"].pct_change().rename(ticker)
    fig2 = go.Figure()
    for (t, df_p), col in zip(peers_data.items(), PEER_COLORS):
        ret_peer = df_p["Close"].pct_change().rename(t)
        combined = pd.concat([ret_main, ret_peer], axis=1).dropna()
        rolling_corr = combined[ticker].rolling(30).corr(combined[t])
        fig2.add_trace(go.Scatter(
            x=rolling_corr.index, y=rolling_corr,
            name=f"{ticker} vs {t}", line=dict(color=col, width=1.8),
        ))
    fig2.add_hline(y=0, line_dash="dot", line_color=C_GRAY, line_width=1)
    fig2.update_layout(**LAYOUT, title=f"Rolling 30-Day Correlation with {ticker}",
                       height=300, yaxis_title="Correlation", yaxis_range=[-1, 1])
    st.plotly_chart(fig2, use_container_width=True, **_CHART_CFG)

    # ── Summary comparison table ──────────────────────────────────────────────
    st.markdown("##### Performance Summary")
    rows = []
    for t, clr in zip([ticker] + list(peers_data.keys()), all_colors):
        df_t = df_main if t == ticker else peers_data[t]
        first = float(df_t["Close"].iloc[0])
        last  = float(df_t["Close"].iloc[-1])
        ret   = (last / first - 1) * 100
        hi    = float(df_t["High"].max())
        lo    = float(df_t["Low"].min())
        vol   = float(df_t["Close"].pct_change().std() * (252**0.5) * 100)
        rows.append({
            "Ticker": t,
            "Current ($)": f"${last:,.2f}",
            "Return":      f"{ret:+.1f}%",
            "Period High": f"${hi:,.2f}",
            "Period Low":  f"${lo:,.2f}",
            "Ann. Vol.":   f"{vol:.1f}%",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)