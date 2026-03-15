"""
data_processing.py
==================
Statistical and analytical processing for the PyClimaExplorer dashboard.

Works exclusively with data returned by modules.data_loader and has
no UI, plotting, or Streamlit dependencies.

Compatible with: Python 3.10+
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from typing import Any

# ── Internal import (loader must be initialised before calling these) ─────────
from modules.data_loader import get_index_timeseries  # noqa: E402


# ═══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _validate_df(df: pd.DataFrame, func_name: str) -> None:
    """Raise ValueError if *df* does not have the expected Year/Value columns."""
    required = {"Year", "Value"}
    if not required.issubset(df.columns):
        raise ValueError(
            f"{func_name}: DataFrame must contain columns {required}. "
            f"Got: {set(df.columns)}"
        )
    if df.empty:
        raise ValueError(f"{func_name}: Input DataFrame is empty.")


# ═══════════════════════════════════════════════════════════════════════════════
#  1. ANOMALY
# ═══════════════════════════════════════════════════════════════════════════════

def compute_anomaly(
    df: pd.DataFrame,
    baseline_start: int,
    baseline_end: int,
) -> pd.DataFrame:
    """
    Compute the anomaly of a climate index relative to a chosen baseline period.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns ``Year`` (int) and ``Value`` (float).
    baseline_start : int
        First year of the baseline period (inclusive).
    baseline_end : int
        Last year of the baseline period (inclusive).

    Returns
    -------
    pd.DataFrame
        Columns: ``Year``, ``Anomaly``.

    Raises
    ------
    ValueError
        If the baseline period yields no data rows.
    """
    _validate_df(df, "compute_anomaly")
    df = df.sort_values("Year")
    baseline_mask = (df["Year"] >= baseline_start) & (df["Year"] <= baseline_end)
    baseline_df = df.loc[baseline_mask]

    if baseline_df.empty:
        raise ValueError(
            f"compute_anomaly: No data found between {baseline_start} "
            f"and {baseline_end}."
        )

    baseline_mean: float = baseline_df["Value"].mean()

    result = df[["Year"]].copy()
    result["Anomaly"] = df["Value"] - baseline_mean
    return result.reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  2. TREND
# ═══════════════════════════════════════════════════════════════════════════════

def compute_trend(df: pd.DataFrame) -> dict[str, float]:
    """
    Compute the linear trend of a time series using ordinary least-squares
    regression via ``scipy.stats.linregress``.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns ``Year`` and ``Value``.

    Returns
    -------
    dict
        Keys:
        - ``slope_per_year``   : regression slope (units/year)
        - ``slope_per_decade`` : slope scaled to per-decade
        - ``intercept``        : regression intercept
        - ``r_squared``        : coefficient of determination (R²)
        - ``p_value``          : two-tailed p-value for the slope

    Raises
    ------
    ValueError
        If the series has fewer than 3 data points.
    """
    _validate_df(df, "compute_trend")

    clean = df.dropna(subset=["Year", "Value"])
    clean = clean.sort_values("Year")
    if len(clean) < 3:
        raise ValueError(
            "compute_trend: Need at least 3 non-NaN rows to compute a trend."
        )

    x: np.ndarray = clean["Year"].to_numpy(dtype=float)
    y: np.ndarray = clean["Value"].to_numpy(dtype=float)

    slope, intercept, r_value, p_value, _ = stats.linregress(x, y)

    return {
        "slope_per_year":   float(slope),
        "slope_per_decade": float(slope * 10),
        "intercept":        float(intercept),
        "r_squared":        float(r_value ** 2),
        "p_value":          float(p_value),
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  3. DECADAL AVERAGE
# ═══════════════════════════════════════════════════════════════════════════════

def compute_decadal_average(df: pd.DataFrame) -> pd.DataFrame:
    """
    Group a time series by decade and return the mean value per decade.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns ``Year`` and ``Value``.

    Returns
    -------
    pd.DataFrame
        Columns: ``Decade`` (int), ``Value`` (float).
        Sorted ascending by decade.

    Example
    -------
    >>> compute_decadal_average(df)
       Decade  Value
    0    1900   0.24
    1    1910   0.28
    """
    _validate_df(df, "compute_decadal_average")

    work = df[["Year", "Value"]].copy()
    work["Decade"] = (work["Year"] // 10) * 10

    result = (
        work.groupby("Decade", as_index=False)["Value"]
        .mean()
        .sort_values("Decade")
        .reset_index(drop=True)
    )
    return result


# ═══════════════════════════════════════════════════════════════════════════════
#  4. ROLLING MEAN
# ═══════════════════════════════════════════════════════════════════════════════

def compute_rolling_mean(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """
    Compute a centred rolling mean over the time series.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns ``Year`` and ``Value``.
    window : int
        Number of years for the rolling window (must be ≥ 1).

    Returns
    -------
    pd.DataFrame
        Columns: ``Year``, ``RollingMean``.
        Edge rows within ``window // 2`` of each end will be NaN.

    Raises
    ------
    ValueError
        If *window* is less than 1.
    """
    _validate_df(df, "compute_rolling_mean")

    if window < 1:
        raise ValueError(
            f"compute_rolling_mean: window must be ≥ 1, got {window}."
        )

    work = df[["Year", "Value"]].sort_values("Year").copy()
    work = work.sort_values("Year")
    work["Value"] = work["Value"].astype(float)
    work["RollingMean"] = (
        work["Value"].rolling(window=window, center=True, min_periods=1).mean()
    )

    return work[["Year", "RollingMean"]].reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  5. CORRELATION MATRIX
# ═══════════════════════════════════════════════════════════════════════════════

def compute_correlation_matrix(index_list: list[str]) -> pd.DataFrame:
    """
    Build a Pearson correlation matrix across multiple climate indices.

    For each index name the function calls ``get_index_timeseries()`` from the
    active data loader, then merges all series on the ``Year`` column before
    computing pairwise correlations.

    Parameters
    ----------
    index_list : list[str]
        Names of climate indices (must match available indices in the loaded
        dataset, e.g. ``["nino34", "amo", "pdo"]``).

    Returns
    -------
    pd.DataFrame
        Square correlation matrix indexed and columned by index names.

    Raises
    ------
    ValueError
        If ``index_list`` is empty or if no common years exist after merging.
    """
    if not index_list:
        raise ValueError("compute_correlation_matrix: index_list must not be empty.")
    
    if len(index_list) < 2:
        raise ValueError("compute_correlation_matrix: Need at least two indices.")

    merged: pd.DataFrame | None = None

    for name in index_list:
        ts = get_index_timeseries(name)

        # Normalise column names
        ts = ts.rename(columns={c: c.strip() for c in ts.columns})
        if "Value" not in ts.columns:
            raise ValueError(
                f"compute_correlation_matrix: index '{name}' returned a "
                f"DataFrame without a 'Value' column. Got: {list(ts.columns)}"
            )

        ts = ts[["Year", "Value"]].rename(columns={"Value": name})

        if merged is None:
            merged = ts
        else:
            merged = pd.merge(merged, ts, on="Year", how="inner")

    if merged is None or merged.empty:
        raise ValueError(
            "compute_correlation_matrix: No overlapping years found after "
            "merging all indices."
        )

    corr_matrix = merged.drop(columns=["Year"]).corr(method="pearson")
    return corr_matrix


# ═══════════════════════════════════════════════════════════════════════════════
#  6. Z-SCORE NORMALISATION
# ═══════════════════════════════════════════════════════════════════════════════

def zscore_normalize(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalise a time series to zero mean and unit variance (z-score).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns ``Year`` and ``Value``.

    Returns
    -------
    pd.DataFrame
        Columns: ``Year``, ``ZScore``.

    Raises
    ------
    ValueError
        If the standard deviation of ``Value`` is zero (constant series).
    """
    _validate_df(df, "zscore_normalize")

    std: float = df["Value"].std(ddof=1)
    if std == 0.0:
        raise ValueError(
            "zscore_normalize: Standard deviation is zero — "
            "cannot normalise a constant series."
        )

    mean: float = df["Value"].mean()
    result = df[["Year"]].copy()
    result["ZScore"] = (df["Value"] - mean) / std
    return result.reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  7. CLIMATE PHASE DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

def detect_climate_phase(df: pd.DataFrame) -> str:
    """
    Determine the current ENSO phase from the most recent value in the series.

    Thresholds
    ----------
    - Value > +0.5  → ``"El Niño"``
    - Value < -0.5  → ``"La Niña"``
    - Otherwise     → ``"Neutral"``

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns ``Year`` and ``Value``.

    Returns
    -------
    str
        One of: ``"El Niño"``, ``"La Niña"``, ``"Neutral"``.
    """
    _validate_df(df, "detect_climate_phase")

    latest_value: float = (
        df.sort_values("Year").dropna(subset=["Value"])["Value"].iloc[-1]
    )

    if latest_value > 0.5:
        return "El Niño"
    if latest_value < -0.5:
        return "La Niña"
    return "Neutral"


# ═══════════════════════════════════════════════════════════════════════════════
#  8. SPATIAL TREND STATISTICS
# ═══════════════════════════════════════════════════════════════════════════════

def compute_spatial_trend_stats(
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    grid: np.ndarray,
) -> dict[str, float]:
    """
    Compute summary statistics over a 2-D spatial climate grid.

    Parameters
    ----------
    latitudes : np.ndarray
        1-D array of latitude values (degrees).
    longitudes : np.ndarray
        1-D array of longitude values (degrees).
    grid : np.ndarray
        2-D array of shape ``(len(latitudes), len(longitudes))``
        containing the gridded climate variable (e.g. SST anomaly,
        EOF loading factor).

    Returns
    -------
    dict
        Keys: ``"mean"``, ``"min"``, ``"max"``, ``"std"``.
        All values are Python ``float``.

    Raises
    ------
    ValueError
        If *grid* dimensions are inconsistent with *latitudes* /
        *longitudes*, or if *grid* contains only NaN values.
    """
    if grid.ndim != 2:
        raise ValueError(
            f"compute_spatial_trend_stats: grid must be 2-D, got {grid.ndim}-D."
        )
    if grid.shape != (len(latitudes), len(longitudes)):
        raise ValueError(
            f"compute_spatial_trend_stats: grid shape {grid.shape} does not "
            f"match (len(latitudes)={len(latitudes)}, "
            f"len(longitudes)={len(longitudes)})."
        )

    valid = grid[~np.isnan(grid)]
    if valid.size == 0:
        raise ValueError(
            "compute_spatial_trend_stats: grid contains only NaN values."
        )

    return {
        "mean": float(np.nanmean(grid)),
        "min":  float(np.nanmin(grid)),
        "max":  float(np.nanmax(grid)),
        "std":  float(np.nanstd(grid, ddof=1)),
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  PUBLIC API
# ═══════════════════════════════════════════════════════════════════════════════

__all__ = [
    "compute_anomaly",
    "compute_trend",
    "compute_decadal_average",
    "compute_rolling_mean",
    "compute_correlation_matrix",
    "zscore_normalize",
    "detect_climate_phase",
    "compute_spatial_trend_stats",
]
