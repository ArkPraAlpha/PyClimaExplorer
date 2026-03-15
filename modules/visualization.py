"""
visualization.py
================
Interactive Plotly visualizations for the PyClimaExplorer climate dashboard.

All functions:
  - Accept pre-processed data from modules.data_loader / modules.data_processing
  - Return plotly.graph_objects.Figure objects (never display them)
  - Are compatible with Streamlit st.plotly_chart()
  - Use template="plotly_white" for a consistent climate-science aesthetic
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ─────────────────────────────────────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

_TEMPLATE = "plotly_white"

# Consistent colour palette for named climate indices
_INDEX_COLORS: dict[str, str] = {
    "AMO":  "#1f77b4",
    "ENSO": "#ff7f0e",
    "PDO":  "#2ca02c",
    "NAO":  "#d62728",
    "SOI":  "#9467bd",
    "IOD":  "#8c564b",
    "AO":   "#e377c2",
}

_CLIMATE_DIVERGING = "RdBu_r"   # canonical diverging scale for anomalies
_CLIMATE_SEQUENTIAL = "Viridis"  # sequential scale for magnitudes


# ─────────────────────────────────────────────────────────────────────────────
#  PRIVATE HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _base_layout(**overrides) -> dict:
    """Return a base Plotly layout dict with shared styling."""
    layout = dict(
        template=_TEMPLATE,
        font=dict(family="Inter, Arial, sans-serif", size=12, color="#2c3e50"),
        plot_bgcolor="white",
        paper_bgcolor="white",
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="#dce3ea",
            borderwidth=1,
        ),
        margin=dict(l=60, r=30, t=70, b=55),
        xaxis=dict(showgrid=True, gridcolor="#ebebeb", zeroline=False),
        yaxis=dict(showgrid=True, gridcolor="#ebebeb", zeroline=True,
                   zerolinecolor="#cccccc"),
    )
    layout.update(overrides)
    return layout


def _index_color(name: str) -> str:
    """Return a consistent colour for a known index, or fall back to a hash."""
    if name in _INDEX_COLORS:
        return _INDEX_COLORS[name]
    # Deterministic fallback using a Plotly qualitative palette
    palette = px.colors.qualitative.Plotly
    return palette[hash(name) % len(palette)]


def _add_trend_line(fig: go.Figure, x: np.ndarray, y: np.ndarray,
                    color: str = "#c0392b", name: str = "Linear Trend") -> go.Figure:
    """Fit and add a linear regression line to an existing figure."""
    mask = ~np.isnan(y)
    if mask.sum() < 2:
        return fig
    coeffs = np.polyfit(x[mask].astype(float), y[mask], 1)
    trend = np.poly1d(coeffs)(x.astype(float))
    fig.add_trace(go.Scatter(
        x=x, y=trend,
        name=name,
        mode="lines",
        line=dict(color=color, width=2, dash="dash"),
        hovertemplate=f"<b>{name}</b>: %{{y:.3f}}<extra></extra>",
    ))
    return fig


# ─────────────────────────────────────────────────────────────────────────────
#  1. plot_index_timeseries
# ─────────────────────────────────────────────────────────────────────────────

def plot_index_timeseries(
    df: pd.DataFrame,
    index_name: str = "Index",
    color: str | None = None,
    show_markers: bool = True,
) -> go.Figure:
    """
    Visualize a single climate index over time.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns ``Year`` (numeric) and ``Value`` (float).
    index_name : str
        Label used in the legend and y-axis title.
    color : str | None
        Line colour.  Defaults to the canonical colour for ``index_name``.
    show_markers : bool
        Whether to overlay circle markers on the line.

    Returns
    -------
    go.Figure
    """
    df = df.sort_values("Year")
    _validate_columns(df, ["Year", "Value"], "plot_index_timeseries")

    line_color = color or _index_color(index_name)
    mode = "lines+markers" if show_markers else "lines"

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["Year"],
        y=df["Value"],
        name=index_name,
        mode=mode,
        line=dict(color=line_color, width=2),
        marker=dict(size=4, color=line_color),
        hovertemplate=(
            "<b>" + index_name + "</b><br>"
            "Year: %{x}<br>"
            "Value: %{y:.4f}<extra></extra>"
        ),
    ))

    fig.update_layout(
        **_base_layout(
            title=dict(text="Climate Index Time Series", font=dict(size=16)),
            xaxis_title="Year",
            yaxis_title=f"{index_name} Value",
        )
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
#  2. plot_multi_index_comparison
# ─────────────────────────────────────────────────────────────────────────────

def plot_multi_index_comparison(df: pd.DataFrame) -> go.Figure:
    """
    Compare multiple climate indices on the same chart.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain column ``Year`` plus one column per index
        (e.g. ``ENSO``, ``PDO``, ``AMO``).

    Returns
    -------
    go.Figure
    """
    if "Year" not in df.columns:
        raise ValueError("DataFrame must contain a 'Year' column.")

    index_cols = [c for c in df.columns if c != "Year"]
    if not index_cols:
        raise ValueError("DataFrame must contain at least one index column.")

    fig = go.Figure()

    for col in index_cols:
        fig.add_trace(go.Scatter(
            x=df["Year"],
            y=df[col],
            name=col,
            mode="lines",
            line=dict(color=_index_color(col), width=1.8),
            hovertemplate=(
                f"<b>{col}</b><br>Year: %{{x}}<br>Value: %{{y:.4f}}<extra></extra>"
            ),
        ))

    fig.update_layout(
        **_base_layout(
            title=dict(text="Climate Index Comparison", font=dict(size=16)),
            xaxis_title="Year",
            yaxis_title="Index Value",
        )
    )
    return fig

def plot_normalized_index_comparison(df: pd.DataFrame) -> go.Figure:
    """
    Compare multiple climate indices normalized using Z-score.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain column 'Year' and one column per normalized index.

    Returns
    -------
    go.Figure
    """

    if "Year" not in df.columns:
        raise ValueError("DataFrame must contain 'Year' column")

    df = df.sort_values("Year")

    index_cols = [c for c in df.columns if c != "Year"]

    fig = go.Figure()

    for col in index_cols:
        fig.add_trace(go.Scatter(
            x=df["Year"],
            y=df[col],
            name=f"{col} (zscore)",
            mode="lines",
            line=dict(color=_index_color(col), width=2),
            hovertemplate=f"<b>{col}</b><br>Year: %{{x}}<br>Z-score: %{{y:.3f}}<extra></extra>"
        ))

    fig.update_layout(
        **_base_layout(
            title=dict(text="Normalized Climate Index Comparison", font=dict(size=16)),
            xaxis_title="Year",
            yaxis_title="Z-score (standard deviations)"
        )
    )

    return fig
# ─────────────────────────────────────────────────────────────────────────────
#  3. plot_decadal_bar_chart
# ─────────────────────────────────────────────────────────────────────────────

def plot_decadal_bar_chart(
    df: pd.DataFrame,
    index_name: str = "Index",
    color_positive: str = "#e74c3c",
    color_negative: str = "#3498db",
) -> go.Figure:
    """
    Visualize decadal averages of a climate index as a colour-coded bar chart.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns ``Decade`` (str/int) and ``Value`` (float).
    index_name : str
        Label used in titles and hover text.
    color_positive : str
        Bar colour for positive anomaly decades.
    color_negative : str
        Bar colour for negative anomaly decades.

    Returns
    -------
    go.Figure
    """
    _validate_columns(df, ["Decade", "Value"], "plot_decadal_bar_chart")

    colors = [
        color_positive if v >= 0 else color_negative
        for v in df["Value"]
    ]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df["Decade"].astype(str),
        y=df["Value"],
        marker_color=colors,
        name=index_name,
        hovertemplate=(
            "<b>Decade: %{x}</b><br>"
            f"{index_name}: %{{y:.4f}}<extra></extra>"
        ),
    ))

    fig.add_hline(y=0, line_dash="dot", line_color="#888888", line_width=1)

    fig.update_layout(
        **_base_layout(
            title=dict(text="Decadal Climate Index Average", font=dict(size=16)),
            xaxis_title="Decade",
            yaxis_title=f"Average {index_name} Value",
            hovermode="x",
            legend=dict(y=1.06),
        )
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
#  4. plot_rolling_trend
# ─────────────────────────────────────────────────────────────────────────────

def plot_rolling_trend(
    df: pd.DataFrame,
    raw_col: str | None = None,
    index_name: str = "Index",
    color: str | None = None,
) -> go.Figure:
    """
    Visualize a smoothed rolling-mean climate trend alongside the raw series.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``Year`` and ``RollingMean``.
        Optionally contains a raw-values column (provide its name via
        ``raw_col`` to overlay it at low opacity).
    raw_col : str | None
        Column name of the unsmoothed series.  Pass ``None`` to omit.
    index_name : str
        Label used in titles.
    color : str | None
        Line colour for the smoothed series.

    Returns
    -------
    go.Figure
    """
    df = df.sort_values("Year")
    _validate_columns(df, ["Year", "RollingMean"], "plot_rolling_trend")

    line_color = color or _index_color(index_name)
    fig = go.Figure()

    # Optional raw series
    if raw_col and raw_col in df.columns:
        fig.add_trace(go.Scatter(
            x=df["Year"],
            y=df[raw_col],
            name=f"{index_name} (raw)",
            mode="lines",
            line=dict(color=line_color, width=1),
            opacity=0.30,
            hovertemplate=f"<b>Raw</b>: %{{y:.4f}}<extra></extra>",
        ))

    fig.add_trace(go.Scatter(
        x=df["Year"],
        y=df["RollingMean"],
        name="Rolling Mean",
        mode="lines",
        line=dict(color=line_color, width=2.8),
        hovertemplate="<b>Rolling Mean</b>: %{y:.4f}<extra></extra>",
    ))

    fig.update_layout(
        **_base_layout(
            title=dict(text="Rolling Climate Trend", font=dict(size=16)),
            xaxis_title="Year",
            yaxis_title=f"{index_name} Value",
        )
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
#  5. plot_global_sst_trend
# ─────────────────────────────────────────────────────────────────────────────

def plot_global_sst_trend(
    df: pd.DataFrame,
    add_trend_line: bool = True,
) -> go.Figure:
    """
    Show global sea surface temperature (SST) trend with an optional
    linear regression overlay.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns ``Year`` (numeric) and ``Value`` (float, °C).
    add_trend_line : bool
        If True, fit and display a linear regression line.

    Returns
    -------
    go.Figure
    """
    df = df.sort_values("Year")
    _validate_columns(df, ["Year", "Value"], "plot_global_sst_trend")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["Year"],
        y=df["Value"],
        name="Global SST",
        mode="lines",
        line=dict(color="#e74c3c", width=2),
        fill="tozeroy",
        fillcolor="rgba(231, 76, 60, 0.08)",
        hovertemplate="<b>SST</b>: %{y:.3f} °C<br>Year: %{x}<extra></extra>",
    ))

    if add_trend_line:
        fig = _add_trend_line(
            fig,
            x=df["Year"].values,
            y=df["Value"].values,
            color="#c0392b",
            name="Linear Trend",
        )

    fig.update_layout(
        **_base_layout(
            title=dict(
                text="Global Sea Surface Temperature Trend",
                font=dict(size=16),
            ),
            xaxis_title="Year",
            yaxis_title="SST (°C)",
        )
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
#  6. plot_correlation_heatmap
# ─────────────────────────────────────────────────────────────────────────────

def plot_correlation_heatmap(corr: pd.DataFrame) -> go.Figure:
    """
    Visualize a symmetric correlation matrix as an annotated heatmap.

    Parameters
    ----------
    corr : pd.DataFrame
        Square correlation matrix (e.g. from ``compute_correlation_matrix()``).

    Returns
    -------
    go.Figure
    """
    if corr.shape[0] != corr.shape[1]:
        raise ValueError("Correlation matrix must be square.")

    labels = corr.columns.tolist()
    z = corr.values.round(3)

    fig = go.Figure(go.Heatmap(
        z=z,
        x=labels,
        y=labels,
        colorscale=_CLIMATE_DIVERGING,
        zmid=0,
        zmin=-1,
        zmax=1,
        text=z,
        texttemplate="%{text:.2f}",
        textfont=dict(size=11),
        colorbar=dict(
            title="Pearson r",
            thickness=14,
            len=0.85,
            tickfont=dict(size=10),
        ),
        hovertemplate=(
            "<b>%{y} vs %{x}</b><br>r = %{z:.3f}<extra></extra>"
        ),
    ))

    fig.update_layout(
        **_base_layout(
            title=dict(
                text="Climate Index Correlation Matrix",
                font=dict(size=16),
            ),
            xaxis=dict(title="", showgrid=False, tickangle=-30),
            yaxis=dict(title="", showgrid=False, autorange="reversed"),
            height=420 + max(0, (len(labels) - 4) * 25),
            margin=dict(l=90, r=30, t=70, b=90),
        )
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
#  7. plot_spatial_pattern_map
# ─────────────────────────────────────────────────────────────────────────────

def plot_spatial_pattern_map(
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    grid: np.ndarray,
    title: str = "Climate Spatial Pattern",
    colorbar_title: str = "Loading Factor",
    zmin: float | None = None,
    zmax: float | None = None,
) -> go.Figure:
    """
    Display a 2-D spatial climate pattern (e.g. an EOF loading map).

    Parameters
    ----------
    latitudes : np.ndarray
        1-D array of latitude values (degrees North).
    longitudes : np.ndarray
        1-D array of longitude values (degrees East).
    grid : np.ndarray
        2-D array with shape ``(len(latitudes), len(longitudes))``.
    title : str
        Chart title.
    colorbar_title : str
        Label for the colour bar.
    zmin, zmax : float | None
        Colour-scale limits.  Defaults to ±max(|grid|).

    Returns
    -------
    go.Figure
    """
    _validate_grid(latitudes, longitudes, grid, "plot_spatial_pattern_map")
    if np.isnan(grid).all():
        raise ValueError("Grid contains only NaN values")
    abs_max = float(np.nanmax(np.abs(grid)))
    zmin = zmin if zmin is not None else -abs_max
    zmax = zmax if zmax is not None else abs_max

    fig = go.Figure(go.Heatmap(
        z=grid,
        x=longitudes,
        y=latitudes,
        colorscale=_CLIMATE_DIVERGING,
        zmid=0,
        zmin=zmin,
        zmax=zmax,
        colorbar=dict(
            title=colorbar_title,
            thickness=14,
            tickfont=dict(size=10),
        ),
        hovertemplate=(
            "Lat: %{y:.1f}°<br>Lon: %{x:.1f}°<br>"
            "Value: %{z:.3f}<extra></extra>"
        ),
    ))

    fig.update_layout(
        **_base_layout(
            title=dict(text=title, font=dict(size=16)),
            xaxis_title="Longitude",
            yaxis_title="Latitude",
            xaxis=dict(showgrid=False, zeroline=False),
            yaxis=dict(showgrid=False, zeroline=False),
            margin=dict(l=60, r=30, t=70, b=55),
        )
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
#  8. plot_spatial_trend_map
# ─────────────────────────────────────────────────────────────────────────────

def plot_spatial_trend_map(
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    grid: np.ndarray,
    units: str = "°C / decade",
) -> go.Figure:
    """
    Display a spatial map of per-gridpoint linear trend magnitudes.

    Parameters
    ----------
    latitudes : np.ndarray
        1-D array of latitude values.
    longitudes : np.ndarray
        1-D array of longitude values.
    grid : np.ndarray
        2-D trend array, shape ``(len(latitudes), len(longitudes))``.
    units : str
        Units appended to the colour-bar label.

    Returns
    -------
    go.Figure
    """
    _validate_grid(latitudes, longitudes, grid, "plot_spatial_trend_map")
    if np.isnan(grid).all():
        raise ValueError("Grid contains only NaN values")
    abs_max = float(np.nanmax(np.abs(grid)))

    fig = go.Figure(go.Heatmap(
        z=grid,
        x=longitudes,
        y=latitudes,
        colorscale=_CLIMATE_DIVERGING,
        zmid=0,
        zmin=-abs_max,
        zmax=abs_max,
        colorbar=dict(
            title=f"Trend<br>({units})",
            thickness=14,
            tickfont=dict(size=10),
        ),
        hovertemplate=(
            "Lat: %{y:.1f}°<br>Lon: %{x:.1f}°<br>"
            f"Trend: %{{z:.4f}} {units}<extra></extra>"
        ),
    ))

    fig.update_layout(
        **_base_layout(
            title=dict(text="Spatial Climate Trend", font=dict(size=16)),
            xaxis_title="Longitude",
            yaxis_title="Latitude",
            xaxis=dict(showgrid=False, zeroline=False),
            yaxis=dict(showgrid=False, zeroline=False),
            margin=dict(l=60, r=30, t=70, b=55),
        )
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
#  9. plot_anomaly_chart
# ─────────────────────────────────────────────────────────────────────────────

def plot_anomaly_chart(
    df: pd.DataFrame,
    index_name: str = "Index",
    baseline_label: str = "1991–2020 baseline",
) -> go.Figure:
    """
    Show climate anomalies relative to a baseline period.

    Positive anomalies are coloured red; negative anomalies blue.
    A dashed zero reference line marks the baseline.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns ``Year`` (numeric) and ``Anomaly`` (float).
    index_name : str
        Label used in titles and hover text.
    baseline_label : str
        Baseline period description shown in the zero-line annotation.

    Returns
    -------
    go.Figure
    """
    df = df.sort_values("Year")
    _validate_columns(df, ["Year", "Anomaly"], "plot_anomaly_chart")

    pos_mask = df["Anomaly"] >= 0
    neg_mask = ~pos_mask

    fig = go.Figure()

    # Positive bars
    fig.add_trace(go.Bar(
        x=df.loc[pos_mask, "Year"],
        y=df.loc[pos_mask, "Anomaly"],
        name="Positive",
        marker_color="#e74c3c",
        hovertemplate="<b>Year:</b> %{x}<br><b>Anomaly:</b> +%{y:.3f}<extra></extra>",
    ))

    # Negative bars
    fig.add_trace(go.Bar(
        x=df.loc[neg_mask, "Year"],
        y=df.loc[neg_mask, "Anomaly"],
        name="Negative",
        marker_color="#3498db",
        hovertemplate="<b>Year:</b> %{x}<br><b>Anomaly:</b> %{y:.3f}<extra></extra>",
    ))

    # Zero reference line
    fig.add_hline(
        y=0,
        line_dash="dash",
        line_color="#555555",
        line_width=1.2,
        annotation_text=baseline_label,
        annotation_position="top right",
        annotation_font=dict(size=10, color="#555555"),
    )

    fig.update_layout(
        **_base_layout(
            title=dict(
                text=f"Climate Anomaly Relative to Baseline — {index_name}",
                font=dict(size=16),
            ),
            xaxis_title="Year",
            yaxis_title=f"{index_name} Anomaly",
            barmode="relative",
            hovermode="x unified",
            legend=dict(y=1.06),
        )
    )
    return fig

def plot_climate_phase_timeline(df: pd.DataFrame) -> go.Figure:
    """
    Visualize ENSO climate phases across time.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns 'Year' and 'Phase'

    Returns
    -------
    go.Figure
    """

    _validate_columns(df, ["Year", "Phase"], "plot_climate_phase_timeline")

    df = df.sort_values("Year")

    phase_colors = {
        "El Niño": "#e74c3c",
        "La Niña": "#3498db",
        "Neutral": "#7f8c8d"
    }

    colors = [phase_colors.get(p, "#95a5a6") for p in df["Phase"]]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=df["Year"],
        y=[1]*len(df),
        marker_color=colors,
        hovertext=df["Phase"],
        hovertemplate="<b>Year:</b> %{x}<br><b>Phase:</b> %{hovertext}<extra></extra>",
        showlegend=False
    ))

    fig.update_layout(
        **_base_layout(
            title=dict(text="ENSO Climate Phase Timeline", font=dict(size=16)),
            xaxis_title="Year",
            yaxis=dict(showticklabels=False, showgrid=False),
        )
    )

    return fig
# ─────────────────────────────────────────────────────────────────────────────
#  VALIDATION HELPERS  (private)
# ─────────────────────────────────────────────────────────────────────────────

def _validate_columns(df: pd.DataFrame, required: list[str], fn: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"{fn}(): DataFrame is missing required column(s): {missing}"
        )


def _validate_grid(lat, lon, grid, fn: str) -> None:
    if grid.ndim != 2:
        raise ValueError(f"{fn}(): grid must be 2-D, got shape {grid.shape}.")
    if grid.shape != (len(lat), len(lon)):
        raise ValueError(
            f"{fn}(): grid shape {grid.shape} does not match "
            f"(len(lat)={len(lat)}, len(lon)={len(lon)})."
        )