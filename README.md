# Stock Price Prediction Dashboard

A modular Streamlit application for market data analysis and ML-based price forecasting.

## Project Structure

```
stock_predictor/
├── app.py                   # Entry point
├── requirements.txt
├── assets/
│   └── style.css            # Global styles
├── utils/
│   ├── data.py              # Data fetching & feature engineering
│   └── model.py             # Model training & forecasting
└── components/
    ├── sidebar.py           # Configuration panel
    ├── metrics.py           # KPI strip
    └── charts.py            # All Plotly chart builders
```

## Setup & Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Launch the app
streamlit run app.py
```

## Features

- Live OHLCV data via Yahoo Finance
- Two ML models: Linear Regression and Random Forest
- Three analysis tabs: Historical, Model Performance, Forecast
- Configurable test split, forecast horizon, and chart overlays
- Confidence band on forecasts (±1 RMSE)

## Disclaimer

For educational and research purposes only. Not financial advice.
