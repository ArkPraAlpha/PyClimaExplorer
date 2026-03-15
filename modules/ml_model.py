"""
ml_model.py — ML Forecasting Backend for PyClimaExplorer
"""

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge


def generate_forecast(df: pd.DataFrame, target_index: str, forecast_horizon: int = 12):
    """
    Fit a Polynomial Ridge Regression on historical climate index data
    and forecast future values.

    Parameters
    ----------
    df              : DataFrame with columns ['Year', target_index]
    target_index    : name of the climate index column e.g. 'AMO', 'ENSO'
    forecast_horizon: number of years to forecast ahead (default 12 → up to 2035)

    Returns
    -------
    hist_years      : np.ndarray  — historical year values
    hist_trend      : np.ndarray  — model's fitted values on historical data
    future_years    : np.ndarray  — forecasted year values
    future_forecast : np.ndarray  — forecasted index values
    """

    # ── Validate inputs ───────────────────────────────────────────────────────
    if target_index not in df.columns:
        raise ValueError(f"Column '{target_index}' not found in DataFrame. "
                         f"Available: {list(df.columns)}")

    df_clean = df[["Year", target_index]].dropna()
    if len(df_clean) < 10:
        raise ValueError(f"Not enough data points to train model "
                         f"(need ≥ 10, got {len(df_clean)}).")

    # ── Prepare arrays ────────────────────────────────────────────────────────
    X_hist = df_clean["Year"].values.reshape(-1, 1).astype(float)
    y_hist = df_clean[target_index].values.astype(float)

    # ── Build pipeline: PolynomialFeatures → Ridge ────────────────────────────
    pipeline = Pipeline([
        ("poly",  PolynomialFeatures(degree=3, include_bias=False)),
        ("ridge", Ridge(alpha=1.0)),
    ])

    pipeline.fit(X_hist, y_hist)

    # ── Historical trendline (model fit on training data) ─────────────────────
    hist_trend = pipeline.predict(X_hist)

    # ── Future years ──────────────────────────────────────────────────────────
    last_year    = int(df_clean["Year"].max())
    future_years = np.arange(last_year + 1,
                             last_year + forecast_horizon + 1,
                             dtype=float).reshape(-1, 1)

    future_forecast = pipeline.predict(future_years)

    return (
        df_clean["Year"].values,   # hist_years
        hist_trend,                # hist_trend
        future_years.flatten(),    # future_years
        future_forecast,           # future_forecast
    )