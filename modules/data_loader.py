"""
data_loader.py
--------------
Loads and exposes climate data from ERSST_v5.cvdp_data.1900-2018.nc.
No UI code. Designed for downstream use with Streamlit / Plotly / pandas.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path

# ── Variable registries ──────────────────────────────────────────────────────

INDEX_VARIABLES: list[str] = [
    "amo_timeseries_mon",
    "pdo_timeseries_mon",
    "nino34",
    "nino3",
    "nino4",
    "atlantic_meridional_mode",
    "indian_ocean_dipole",
    "north_tropical_atlantic",
    "south_tropical_atlantic",
    "tropical_indian_ocean",
    "southern_ocean",
]

SPATIAL_VARIABLES: list[str] = [
    "amo_pattern_mon",
    "pdo_pattern_mon",
    "sst_spatialmean_ann",
    "sst_spatialstddev_ann",
    "sst_trends_ann",
    "sst_trends_mon",
]

GLOBAL_SST_VARIABLES: list[str] = [
    "sst_global_avg_ann",
    "sst_global_avg_mon",
    "sst_global_avg_runtrend_8yr",
    "sst_global_avg_runtrend_10yr",
    "sst_global_avg_runtrend_12yr",
    "sst_global_avg_runtrend_14yr",
    "sst_global_avg_runtrend_16yr",
]

# ── Module-level cache ────────────────────────────────────────────────────────

_dataset: xr.Dataset | None = None


# ═════════════════════════════════════════════════════════════════════════════
#  Dataset Loader
# ═════════════════════════════════════════════════════════════════════════════

def load_dataset(file_path: str) -> xr.Dataset:
    """
    Load the NetCDF dataset using xarray with lazy loading.
    Caches the result in memory; subsequent calls return the cached object.

    Parameters
    ----------
    file_path : str
        Path to ERSST_v5.cvdp_data.1900-2018.nc

    Returns
    -------
    xr.Dataset
    """
    global _dataset
    _dataset = None
    try:
        _dataset = xr.open_dataset(
        file_path,
        engine="netcdf4",
        decode_times=False,
        chunks="auto"
    )
    except FileNotFoundError:
        raise FileNotFoundError(f"Dataset not found: {file_path}")
    except Exception as exc:
        raise RuntimeError(f"Failed to open dataset: {exc}") from exc

    return _dataset


def _require_dataset() -> xr.Dataset:
    """Raise if load_dataset() has not been called yet."""
    if _dataset is None:
        raise RuntimeError(
            "Dataset not loaded. Call load_dataset(file_path) first."
        )
    return _dataset


# ═════════════════════════════════════════════════════════════════════════════
#  Metadata
# ═════════════════════════════════════════════════════════════════════════════

def get_available_variables() -> dict[str, list[str]]:
    """
    Return the supported variable names grouped by category.
    Only lists variables that actually exist in the loaded dataset.

    Returns
    -------
    dict with keys "indices", "spatial_patterns", "global_sst"
    """
    ds = _require_dataset()
    present = set(ds.data_vars)

    return {
        "indices":          [v for v in INDEX_VARIABLES       if v in present],
        "spatial_patterns": [v for v in SPATIAL_VARIABLES     if v in present],
        "global_sst":       [v for v in GLOBAL_SST_VARIABLES  if v in present],
    }

def get_dataset_info() -> dict:
    """
    Return general metadata about the loaded dataset.
    Useful for displaying dataset information in the UI.
    """
    ds = _require_dataset()

    return {
        "variables": list(ds.data_vars),
        "dimensions": dict(ds.dims),
        "coordinates": list(ds.coords),
        "size_mb": round(ds.nbytes / 1e6, 2)
    }


def list_indices() -> list[str]:
    """
    Return available climate indices for dropdown menus.
    """
    return get_available_variables()["indices"]


def list_spatial_variables() -> list[str]:
    """
    Return available spatial climate variables.
    """
    return get_available_variables()["spatial_patterns"]


# ═════════════════════════════════════════════════════════════════════════════
#  Climate Index Time Series
# ═════════════════════════════════════════════════════════════════════════════

def get_index_timeseries(index_name: str) -> pd.DataFrame:

    ds = _require_dataset()

    # ── Bug 1 fix: convert dask-backed DataVariables to plain list ────────────
    actual_vars         = list(ds.data_vars)
    actual_vars_stripped = [v.strip() for v in actual_vars]

    if index_name.strip() not in actual_vars_stripped:
        raise KeyError(
            f"{index_name} not found in dataset. "
            f"Variables actually in file: {actual_vars}"
        )

    resolved_name = actual_vars[actual_vars_stripped.index(index_name.strip())]
    da = ds[resolved_name]

    # ── Bug 3 fix: case-insensitive dim search ────────────────────────────────
    time_dim = None
    for _d in da.dims:
        if _d.lower() == "time":
            time_dim = _d
            break

    if time_dim is None:
        raise ValueError(
            f"{index_name} must contain a 'time' dimension. "
            f"Found dims: {da.dims}"
        )

    if "TIME" not in ds.coords:
        raise ValueError("Dataset missing TIME coordinate")

    # ── Bug 2 fix: .compute() materialises dask array before numpy ops ────────
    values = np.array(da.values).reshape(-1)
    values = np.where(np.isfinite(values), values, np.nan)

    years            = np.asarray(ds["TIME"].values).astype(int)
    months_per_year  = 12
    expected_len     = len(years) * months_per_year

    if len(values) < expected_len:
        raise ValueError(
            f"{index_name}: expected {expected_len} months but got {len(values)}"
        )

    values  = values[:expected_len].reshape(len(years), months_per_year)
    annual  = np.nanmean(values, axis=1)

    df = pd.DataFrame({"Year": years, "Value": annual.astype(float)})
    return df.dropna().reset_index(drop=True)

# ═════════════════════════════════════════════════════════════════════════════
#  Spatial Pattern Variables
# ═════════════════════════════════════════════════════════════════════════════

def get_spatial_pattern(
    variable_name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract a 2-D spatial climate field.

    Parameters
    ----------
    variable_name : str
        One of SPATIAL_VARIABLES, e.g. "amo_pattern_mon"

    Returns
    -------
    tuple of (latitudes, longitudes, data_grid)
        latitudes  : 1-D np.ndarray  shape (nlat,)
        longitudes : 1-D np.ndarray  shape (nlon,)
        data_grid  : 2-D np.ndarray  shape (nlat, nlon)
    """
    ds = _require_dataset()

    if variable_name not in ds.data_vars:
        raise KeyError(
            f"'{variable_name}' not found in dataset. "
            f"Available spatial variables: {SPATIAL_VARIABLES}"
        )

    da: xr.DataArray = ds[variable_name]

    # Collapse any non-spatial dimensions (e.g. time, frequency) by mean
    spatial_dims = {"lat", "lon"}
    extra_dims = [d for d in da.dims if d not in spatial_dims]
    if extra_dims:
        da = da.mean(dim=extra_dims)

    latitudes  = ds["lat"].values.astype(float)
    longitudes = ds["lon"].values.astype(float)
    grid       = da.values.astype(float)

    # Ensure shape is (nlat, nlon)
    if grid.shape != (len(latitudes), len(longitudes)):
        grid = grid.reshape(len(latitudes), len(longitudes))

    return latitudes, longitudes, grid


# ═════════════════════════════════════════════════════════════════════════════
#  Global SST Variables
# ═════════════════════════════════════════════════════════════════════════════

def get_global_sst(variable_name: str) -> pd.DataFrame:
    """
    Extract a global SST scalar time series.

    Parameters
    ----------
    variable_name : str
        One of GLOBAL_SST_VARIABLES, e.g. "sst_global_avg_ann"

    Returns
    -------
    pd.DataFrame
        Columns: Year (int), Value (float)
    """
    ds = _require_dataset()

    if variable_name not in ds.data_vars:
        raise KeyError(
            f"'{variable_name}' not found in dataset. "
            f"Available global SST variables: {GLOBAL_SST_VARIABLES}"
        )

    da: xr.DataArray = ds[variable_name]

    # Annual variables may use 'TIME' (yearly); monthly use 'time'
    if "TIME" in da.dims:
        time_values = ds["TIME"].values
        values      = da.values.squeeze()
        years       = _extract_years(time_values)
        df = pd.DataFrame({"Year": years, "Value": values.astype(float)})

    elif "time" in da.dims:
        values = da.values.squeeze()
        values = np.where(np.isfinite(values), values, np.nan)

        years = ds["TIME"].values
        months_per_year = 12
        expected_len = len(years) * months_per_year

        # trim to expected length
        values = values[:expected_len]

        # reshape months → years
        values = values.reshape(len(years), months_per_year)

        annual = np.nanmean(values, axis=1)

        df = pd.DataFrame({
            "Year": years.astype(int),
            "Value": annual.astype(float)
        })
    else:
        raise ValueError(
            f"Cannot identify a time dimension in variable '{variable_name}'. "
            f"Dims found: {da.dims}"
        )

    df = df.dropna(subset=["Value"]).reset_index(drop=True)
    return df


# ═════════════════════════════════════════════════════════════════════════════
#  Filtering Utilities
# ═════════════════════════════════════════════════════════════════════════════

def filter_time_range(
    start_year: int,
    end_year: int,
    ds: xr.Dataset | None = None,
) -> xr.Dataset:
    """
    Subset the dataset to a year range using the 'time' coordinate.

    Parameters
    ----------
    start_year : int
    end_year   : int
    ds         : xr.Dataset, optional
        If None, uses the cached dataset.

    Returns
    -------
    xr.Dataset  (view, not a copy — xarray lazy selection)
    """
    if ds is None:
        ds = _require_dataset()

    start = f"{start_year}-01-01"
    end   = f"{end_year}-12-31"

    if "time" in ds.coords:
        return ds.sel(time=slice(start, end))
    if "TIME" in ds.coords:
        return ds.sel(TIME=slice(start_year, end_year))

    raise ValueError("Dataset has neither 'time' nor 'TIME' coordinate.")


def filter_region(
    min_lat: float,
    max_lat: float,
    min_lon: float,
    max_lon: float,
    ds: xr.Dataset | None = None,
) -> xr.Dataset:
    """
    Subset the dataset to a lat/lon bounding box.

    Parameters
    ----------
    min_lat, max_lat : float
    min_lon, max_lon : float
    ds               : xr.Dataset, optional
        If None, uses the cached dataset.

    Returns
    -------
    xr.Dataset  (lazy selection)
    """
    if ds is None:
        ds = _require_dataset()

    if "lat" not in ds.coords or "lon" not in ds.coords:
        raise ValueError("Dataset does not have 'lat'/'lon' coordinates.")

    return ds.sel(
        lat=slice(min_lat, max_lat),
        lon=slice(min_lon, max_lon),
    )


# ═════════════════════════════════════════════════════════════════════════════
#  Internal helpers
# ═════════════════════════════════════════════════════════════════════════════

def _detect_time_dim(da: xr.DataArray, preferred: str = "time") -> str:
    """Return the first time-like dimension found in *da*."""
    candidates = [preferred, "TIME", "time"]
    for c in candidates:
        if c in da.dims:
            return c
    raise ValueError(
        f"No recognised time dimension in variable '{da.name}'. "
        f"Dims: {da.dims}"
    )


def _extract_years(time_values: np.ndarray) -> np.ndarray:
    """
    Convert a numpy array of time values to integer years.
    Handles cftime objects, numpy datetime64, and plain integers/floats.
    """
    try:
        # cftime or datetime-like
        return np.array([int(str(t)[:4]) for t in time_values])
    except Exception:
        pass
    try:
        return pd.to_datetime(time_values).year.values.astype(int)
    except Exception:
        pass
    # Plain numeric (e.g. year stored as float 1900.0)
    return time_values.astype(int)
