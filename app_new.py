"""
PyClimaExplorer — Interactive Climate Data Dashboard
TECHNEX '26 | Hack It Out Hackathon | IIT (BHU) Varanasi
"""

import streamlit as st
import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy import stats
import warnings
from modules.data_loader import (
    load_dataset,
    get_dataset_info,
    get_index_timeseries,
    get_spatial_pattern,
    get_global_sst
)
# ADD after line 29 (from modules.visualization import *):
from modules.ml_model import generate_forecast

from modules.data_processing import (
    compute_rolling_mean,
    compute_decadal_average,
    compute_correlation_matrix
)

from modules.visualization import *
warnings.filterwarnings("ignore")

# ── Try importing xarray for real NetCDF support ─────────────────────────────
try:
    import xarray as xr
    XARRAY_AVAILABLE = True
except ImportError:
    XARRAY_AVAILABLE = False

# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ═══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="PyClimaExplorer – Climate Data",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  [data-testid="stSidebar"] { background: #0e1621; }
  [data-testid="stSidebar"] * { color: #e8edf3 !important; }
  [data-testid="stSidebar"] .stSelectbox label,
  [data-testid="stSidebar"] .stMultiSelect label,
  [data-testid="stSidebar"] .stSlider label { color: #90b4d4 !important; font-size: 0.82rem; }
  [data-testid="stSidebar"] .stDivider { border-color: #2a3a50; }
  .main-title { font-size: 3.5rem; font-weight: 700; color: #1a3a5c; letter-spacing: -0.5px; }
  .section-label { font-size: 0.78rem; font-weight: 600; color: #5a7a9a;
                   text-transform: uppercase; letter-spacing: 0.8px; margin-bottom: 2px; }
  .stat-block { background: #152232; border-left: 3px solid #4a9eda;
                border-radius: 6px; padding: 8px 12px; margin: 6px 0;
                box-shadow: 0 1px 4px rgba(0,0,0,0.35); }
  .stat-val   { font-size: 1rem; font-weight: 700; color: #e8f4ff; }
  .stat-lbl   { font-size: 0.72rem; color: #7aaecc; }
            
  div[data-testid="stTabs"] button[aria-selected="true"] {
    background: #1a3a5c; color: white; border-radius: 6px; }
  div[data-testid="stTabs"] button {
    border-radius: 6px; padding: 4px 16px; }
  .enso-tag { display:inline-block; background:#e67e22; color:white;
              border-radius:4px; padding:2px 8px; font-size:0.78rem; font-weight:600; }
  /* Carousel card hover effect */
  .story-card { transition: box-shadow 0.2s; border-radius: 10px; }
  .story-card:hover { box-shadow: 0 4px 20px rgba(26,58,92,0.13); }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════
REGION_BOUNDS = {
    "Global":         (-90, 90, -180, 180),
    "North Atlantic": (0,  80,  -80,   20),
    "Pacific Ocean":  (-60, 60, 120,  -70),
    "Europe":         (35,  75,  -10,   40),
    "India":          (8,   37,   68,   98),
    "Arctic":         (60,  90, -180,  180),
}

INDEX_META = {
    "AMO":  {"color": "#1f77b4", "full": "Atlantic Multidecadal Oscillation"},
    "ENSO": {"color": "#ff7f0e", "full": "El Niño–Southern Oscillation"},
    "PDO":  {"color": "#2ca02c", "full": "Pacific Decadal Oscillation"},
    "NAO":  {"color": "#d62728", "full": "North Atlantic Oscillation"},
    "SOI":  {"color": "#9467bd", "full": "Southern Oscillation Index"},

}
INDEX_MAP = {
    "AMO": "amo_timeseries_mon",
    "PDO": "pdo_timeseries_mon",
    "ENSO": "nino34",
    "NAO": None,
    "SOI": None
}

DATASET_INFO = {
    "ERA5 Reanalysis": {
        "Provider": "ECMWF", "Coverage": "1940 – Present",
        "Resolution": "0.25° × 0.25°", "Variables": "240+",
        "Format": "NetCDF (.nc)", "URL": "https://cds.climate.copernicus.eu",
    },
    "CMIP6 Climate Model": {
        "Provider": "WCRP / PCMDI", "Coverage": "1850 – 2100",
        "Resolution": "~1° × 1°", "Variables": "Temperature, Precip, Sea Level",
        "Format": "NetCDF (.nc)", "URL": "https://esgf-node.llnl.gov",
    },
    "ERSST Sea Surface Temperature": {
        "Provider": "NOAA", "Coverage": "1854 – Present",
        "Resolution": "2° × 2°", "Variables": "Sea Surface Temperature",
        "Format": "NetCDF (.nc)", "URL": "https://www.ncei.noaa.gov",
    },
}

# ═══════════════════════════════════════════════════════════════════════════════
#  STORY MODE DATA
# ═══════════════════════════════════════════════════════════════════════════════
story_mode_data = {
    "1930s Dust Bowl Drought": {
        "year": 1934,
        "default_indices": ["PDO", "AMO", "NAO"],
        "description": (
            "A decade-long drought devastated the North American Great Plains. "
            "Driven by a negative PDO phase and anomalous high pressure, the region "
            "lost topsoil at an unprecedented scale — displacing 3.5 million people "
            "in the worst ecological and humanitarian catastrophe of the 20th century."
        ),
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/0/0e/Dust-storm-Texas-1935.png/640px-Dust-storm-Texas-1935.png",
        "local_images": [
            ("images/108802891-56a9e2075f9b58b7d0ffaa81.jpg", "Dust storm approaching — a haboob engulfs the plains, 1930s"),
            
        ],
        "accent": "#BA7517",
        "badge_bg": "#FAEEDA",
        "badge_text": "#633806",
    },
    "1972–73 Severe El Niño": {
        "year": 1972,
        "default_indices": ["ENSO", "SOI", "PDO"],
        "description": (
            "One of the strongest El Niño events of the 20th century. It collapsed "
            "Peru's anchovy fishery and triggered catastrophic monsoon failures across "
            "the Sahel and India. Global food prices spiked, demonstrating for the "
            "first time how a Pacific climate mode could cascade into geopolitical crises."
        ),
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/c/c8/Corrientes-Ocean-World.png/640px-Corrientes-Ocean-World.png",
        "local_images": [
            ("images/download.jpg", "desolate landscape of cracked, parched earth stretching toward a distant horizon under a clear blue skys"),
            
        ],
        "accent": "#1D9E75",
        "badge_bg": "#E1F5EE",
        "badge_text": "#085041",
    },
    "1988 North American Drought": {
        "year": 1988,
        "default_indices": ["PDO", "NAO", "AMO"],
        "description": (
            "A La Niña–linked drought caused the worst US crop failures since the 1930s, "
            "while the Mississippi River ran at record lows. That same summer, NASA "
            "scientist James Hansen's Congressional testimony brought climate change to "
            "global public attention for the very first time."
        ),
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/9/90/Drought_area_in_US.png/640px-Drought_area_in_US.png",
        "local_images": [
            ("images/c.jpg", "desolate landscape of cracked, parched earth stretching toward a distant horizon under a clear blue skys"),
            
        ],
        "accent": "#D85A30",
        "badge_bg": "#FAECE7",
        "badge_text": "#712B13",
    },
    "1998 Super El Niño": {
        "year": 1998,
        "default_indices": ["ENSO", "SOI", "PDO"],
        "description": (
            "The strongest El Niño on record up to that point — producing catastrophic "
            "flooding in South America, severe droughts in Southeast Asia and Australia, "
            "and record-breaking coral bleaching across the tropics. 1998 became, and "
            "held for nearly two decades, the hottest year ever recorded."
        ),
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3e/NASA_1998ElNino.jpg/640px-NASA_1998ElNino.jpg",
        "local_images": [
            ("images/g.jpg", "TOPEX/Poseidon satellite: sea-surface height anomaly, 1 Dec 1997 — the warm pool (white/red) towering over the equatorial Pacific at peak El Niño"),
        
        ],
        "accent": "#378ADD",
        "badge_bg": "#E6F1FB",
        "badge_text": "#0C447C",
    },
    "2010 Russia–Pakistan Extremes": {
        "year": 2010,
        "default_indices": ["AMO", "NAO", "ENSO", "PDO"],
        "description": (
            "A remarkable simultaneous double-extreme: a blocking high pressure system "
            "locked record heat over Russia (killing 55,000) while the same circulation "
            "pattern channelled unprecedented monsoon rains into Pakistan, flooding "
            "one-fifth of the country. Scientists identified a La Niña–NAO coupling "
            "as the root driver."
        ),
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/7/7c/Pakistan_floods_2010.jpg/640px-Pakistan_floods_2010.jpg",
        "local_images": [
            ("images/r.jpg", "ERA-I reanalysis: SAT anomaly (K) over Russia, Pakistan rainfall anomaly (mm/day), Z300 geopotential and V300 wind anomalies — 25 Jul–8 Aug 2010"),
        ],
        "accent": "#7F77DD",
        "badge_bg": "#EEEDFE",
        "badge_text": "#3C3489",
    },
}

# ═══════════════════════════════════════════════════════════════════════════════
#  SYNTHETIC DATA GENERATORS  (used when no real .nc file is uploaded)
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data
def generate_climate_indices(start_year: int, end_year: int) -> pd.DataFrame:
    """Return a DataFrame of synthetic but physically plausible climate indices."""
    years = np.arange(start_year, end_year + 1)
    n = len(years)
    t = np.linspace(0, 1, n)
    rng = np.random.default_rng(42)

    # AMO: ~70-year cycle + gentle upward trend
    amo = (0.30 * np.sin(2 * np.pi * t * (n / 70))
           + 0.12 * t
           + 0.04 * rng.standard_normal(n))

    # ENSO: dominant 4.5-yr + secondary 3.2-yr cycle
    enso = (0.40 * np.sin(2 * np.pi * t * (n / 4.5))
            + 0.18 * np.sin(2 * np.pi * t * (n / 3.2))
            + 0.13 * rng.standard_normal(n))

    # PDO: 25-year decadal mode
    pdo = (0.35 * np.sin(2 * np.pi * t * (n / 25))
           + 0.08 * rng.standard_normal(n))

    # NAO: irregular ~8-year signal + noise
    nao = (0.25 * np.sin(2 * np.pi * t * (n / 8))
           + 0.28 * rng.standard_normal(n))

    # SOI: roughly anti-correlated with ENSO
    soi = -0.65 * enso + 0.12 * rng.standard_normal(n)

    return pd.DataFrame({"Year": years, "AMO": amo, "ENSO": enso,
                         "PDO": pdo, "NAO": nao, "SOI": soi})


@st.cache_data
def generate_temperature_map(year: int):
    """Return lat/lon grids + temperature-anomaly array for the given year."""
    lats = np.linspace(-90, 90, 73)
    lons = np.linspace(-180, 180, 145)
    LON, LAT = np.meshgrid(lons, lats)

    baseline_warming = (year - 1900) * 0.009          # ~0.9 °C per century
    polar_amp = (2.2 * np.exp(-((LAT - 78) ** 2) / 300)
                 + 1.5 * np.exp(-((LAT + 78) ** 2) / 300))
    land_signal = (0.25 * np.sin(LON * np.pi / 55)
                   * np.cos(LAT * np.pi / 40))
    rng = np.random.default_rng(year % 100)
    noise = 0.12 * rng.standard_normal(LON.shape)

    anomaly = baseline_warming + 0.35 * polar_amp + land_signal + noise
    return lats, lons, anomaly

# @st.cache_data
def load_temperature_map(variable="sst_trends_ann", region_bounds=None):
    lat, lon, grid = get_spatial_pattern(variable)
    if region_bounds is not None:
        min_lat, max_lat, min_lon, max_lon = region_bounds
        lat_mask = (lat >= min_lat) & (lat <= max_lat)
        lon_mask = (lon >= min_lon) & (lon <= max_lon)
        grid = grid[np.ix_(lat_mask, lon_mask)]
        lat  = lat[lat_mask]
        lon  = lon[lon_mask]
    return lat, lon, grid

@st.cache_data
def generate_amo_spatial_pattern():
    """Return North-Atlantic lat/lon grids + AMO EOF1 loading-factor array."""
    lats = np.linspace(0, 75, 60)
    lons = np.linspace(-120, 20, 90)
    LON, LAT = np.meshgrid(lons, lats)

    pattern = (0.85 * np.exp(-((LAT - 52) ** 2 / 180 + (LON + 28) ** 2 / 700))
               - 0.45 * np.exp(-((LAT - 18) ** 2 / 130 + (LON + 58) ** 2 / 450))
               + 0.25 * np.sin(LAT * np.pi / 38) * np.cos(LON * np.pi / 75))
    return lats, lons, pattern


@st.cache_data
def decadal_averages(df: pd.DataFrame, indices: list[str]) -> pd.DataFrame:
    df = df.copy()
    df["Decade"] = (df["Year"] // 10) * 10
    agg = df.groupby("Decade")[indices].mean().reset_index()
    agg["Label"] = agg["Decade"].astype(str) + "s"
    return agg


# ═══════════════════════════════════════════════════════════════════════════════
#  REAL NetCDF LOADER  (optional, triggers only when user uploads a file)
# ═══════════════════════════════════════════════════════════════════════════════

def load_netcdf(uploaded_file):
    if not XARRAY_AVAILABLE:
        return None, None
    import tempfile, os
    with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name
    try:
        ds = xr.open_dataset(tmp_path, engine="netcdf4")
        return ds, list(ds.data_vars)
    except Exception as e:
        st.error(f"Could not read NetCDF: {e}")
        return None, None


# ═══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
dataset_loaded = False

with st.sidebar:
    st.markdown("## CONTROL PANEL")
    st.divider()

    # Optional real data upload
    nc_file = st.file_uploader("Upload NetCDF (.nc)", type=["nc"],
                                help="Upload ERA5 / CMIP6 / ERSST file")

    real_ds, real_vars = None, None
    if nc_file and XARRAY_AVAILABLE:
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix=".nc") as tmp:
            tmp.write(nc_file.read())
            tmp_path = tmp.name
        ds = load_dataset(tmp_path)
        info = get_dataset_info()
        dataset_loaded = True
        real_vars = info["variables"]
        st.success(f"Dataset loaded: {len(real_vars)} variables")

    st.divider()

    region = st.selectbox("Select Region", list(REGION_BOUNDS.keys()))

    all_indices = list(INDEX_META.keys())
    selected_indices = st.multiselect(
        "Select Climate Indices", all_indices,
        default=["AMO", "PDO", "ENSO", "SOI"]
    )
    if not selected_indices:
        selected_indices = ["AMO"]

    time_range = st.slider(
    "Time Period",
    1900,
    2018 if dataset_loaded else 2023,
    (1900, 2018)
    )

    dataset = st.selectbox("Dataset", list(DATASET_INFO.keys()))

    SPATIAL_VARS_IN_FILE = [
        "sst_trends_ann", "sst_trends_mon",
        "amo_pattern_mon", "pdo_pattern_mon",
        "sst_spatialmean_ann", "sst_spatialstddev_ann",
    ]
    if dataset_loaded and real_vars:
        # Show only spatial variables that exist in the uploaded file
        available_spatial = [v for v in SPATIAL_VARS_IN_FILE if v in real_vars]
        climate_var = st.selectbox(
            "Spatial Variable (maps)",
            available_spatial if available_spatial else real_vars
        )
    else:
        climate_var = "sst_trends_ann"

    st.divider()

    # ── Statistical insights ─────────────────────────────────────────────────
    # ── Statistical insights — real data when loaded, labelled estimates otherwise
    if dataset_loaded:
        # Mean global SST from actual file
        try:
            _sst_df = get_global_sst("sst_global_avg_ann")
            _sst_f  = _sst_df[
                (_sst_df["Year"] >= time_range[0]) &
                (_sst_df["Year"] <= time_range[1])
            ]
            mean_anom     = round(float(_sst_f["Value"].mean()), 2)
            anom_suffix   = "°C  (ERSST mean)"
        except Exception:
            mean_anom     = round((time_range[1] - 1900) * 0.009 + 0.28, 2)
            anom_suffix   = "°C  (est.)"

        # AMO trend from actual file
        try:
            from modules.data_processing import compute_trend
            _amo_df  = get_index_timeseries("amo_timeseries_mon")
            _amo_f   = _amo_df[
                (_amo_df["Year"] >= time_range[0]) &
                (_amo_df["Year"] <= time_range[1])
            ]
            _amo_t   = compute_trend(_amo_f)
            amo_trend_str = f"{_amo_t['slope_per_decade']:+.4f}/decade"
        except Exception:
            amo_trend_str = "+0.12/decade (est.)"

        # ENSO phase from actual nino34 values
        try:
            from modules.data_processing import detect_climate_phase
            _enso_df = get_index_timeseries("nino34")
            _enso_f  = _enso_df[
                (_enso_df["Year"] >= time_range[0]) &
                (_enso_df["Year"] <= time_range[1])
            ]
            phase = detect_climate_phase(_enso_f)
            tag_col = (
                "#e67e22" if "Niño" in phase else
                "#2980b9" if "Niña" in phase else
                "#27ae60"
            )
        except Exception:
            phase, tag_col = "Unavailable", "#888888"

    else:
        # No file uploaded — synthetic estimates, clearly labelled
        mean_anom     = round((time_range[1] - 1900) * 0.009 + 0.28, 2)
        anom_suffix   = "°C  (est.)"
        amo_trend_str = "+0.12/decade (est.)"
        t_norm  = (time_range[1] - 1900) / 118
        enso_raw = 0.4 * np.sin(2 * np.pi * t_norm * (118 / 4.5))
        if enso_raw > 0.15:
            phase, tag_col = "El Niño (est.)", "#e67e22"
        elif enso_raw < -0.15:
            phase, tag_col = "La Niña (est.)", "#2980b9"
        else:
            phase, tag_col = "Neutral (est.)", "#27ae60"

    st.markdown(f"""
<div class="stat-block">
  <div class="stat-lbl">Mean Global SST</div>
  <div class="stat-val">+{mean_anom} {anom_suffix}</div>
</div>
<div class="stat-block">
  <div class="stat-lbl">AMO Trend</div>
  <div class="stat-val">{amo_trend_str}</div>
</div>
<div class="stat-block">
  <div class="stat-lbl">Current ENSO Status</div>
  <div class="stat-val" style="color:{tag_col}">{phase}</div>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  HEADER
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown(
    """
    <h1 style="
        text-align:center;
        font-size:clamp(40px,5vw,70px);
        font-weight:800;
        color:white;
        animation: softGlow 3s ease-in-out infinite alternate;
        margin-top:10px;
    ">
    🌍 PyClimaExplorer: Exploring Global Climate Data
    </h1>

    <style>
    @keyframes softGlow {
        from { text-shadow:0 0 5px rgba(255,255,255,0.3); }
        to { text-shadow:0 0 20px rgba(255,255,255,0.9); }
    }
    </style>
    """,
    unsafe_allow_html=True
)
# ADD after the `with st.sidebar:` block ends, before line 478:
_rb = REGION_BOUNDS[region]  # (min_lat, max_lat, min_lon, max_lon)
# Available for any spatial filter call: ds.sel(lat=slice(_rb[0],_rb[1]), lon=slice(_rb[2],_rb[3]))


tab_dash, tab_analysis, tab_globe, tab_stories, tab_compare, tab_ml, tab_sources, tab_about = st.tabs(
    ["📊 Dashboard", "📈 Analysis", "🌐 3D Globe", "📖 Story Mode",
     "⏮️ Comparison", "🤖 ML Forecast", "🗄️ Data Sources", "ℹ️ About"]
)


# ═══════════════════════════════════════════════════════════════════════════════
#  DATA PREPARATION
# ═══════════════════════════════════════════════════════════════════════════════

df_all = generate_climate_indices(1900, 2023)
df = None

if dataset_loaded:

    index_data = {}

    # Load each selected climate index
    for idx in selected_indices:

        dataset_var = INDEX_MAP.get(idx)

        if dataset_var is None:
            st.warning(f"{idx} not available in this dataset.")
            continue

        try:
            df_idx = get_index_timeseries(dataset_var)
            df_idx = df_idx.rename(columns={"Value": idx})
            index_data[idx] = df_idx

        except Exception as e:
            st.warning(f"Could not load index {idx}: {e}")

    # Merge indices into a single dataframe
    if index_data:

        first_idx = list(index_data.keys())[0]

        # Start dataframe using first index
        df = index_data[first_idx][["Year", first_idx]]

        # Merge remaining indices
        for idx in list(index_data.keys())[1:]:
            df = df.merge(
                index_data[idx][["Year", idx]],
                on="Year",
                how="inner"
            )

        # Apply time range filter
        df = df[
            (df["Year"] >= time_range[0]) &
            (df["Year"] <= time_range[1])
        ].copy()
if df is None:
    # Fallback: use synthetic data for whatever indices the user selected
    keep = ["Year"] + [c for c in selected_indices if c in df_all.columns]
    df = df_all[keep].copy()
    df = df[
        (df["Year"] >= time_range[0]) &
        (df["Year"] <= time_range[1])
    ].copy()

available_indices = [c for c in df.columns if c != "Year"]

# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 1 — DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════

with tab_dash:

    # ── Story Mode Carousel ───────────────────────────────────────────────────
    st.markdown('<p class="section-label">Climate Story Mode</p>',
                unsafe_allow_html=True)

    story_keys = list(story_mode_data.keys())
    total_stories = len(story_keys)

    # Session state initialisation
    if "story_idx" not in st.session_state:
        st.session_state.story_idx = 0
    if "story_detail_idx" not in st.session_state:
        st.session_state.story_detail_idx = 0

    idx = st.session_state.story_idx
    event_name = story_keys[idx]
    event = story_mode_data[event_name]

    # Card: image left, text right
    with st.container():
        img_col, txt_col = st.columns([1, 2], gap="medium")

        with img_col:
            local_imgs = event.get("local_images", [])
            if local_imgs:
                for fname, cap in local_imgs:
                    img_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), fname)
                    if os.path.exists(img_path):
                        st.image(img_path, use_container_width=True, caption=cap)
                    else:
                        st.caption(f"_(image not found: {fname})_")
            else:
                try:
                    st.image(event["image_url"], use_container_width=True)
                except Exception:
                    st.info("Image unavailable")

        with txt_col:
            # Coloured accent bar
            st.markdown(
                f'<div style="height:3px;width:48px;border-radius:2px;'
                f'background:{event["accent"]};margin-bottom:8px;"></div>',
                unsafe_allow_html=True,
            )
            # Year badge
            st.markdown(
                f'<span style="background:{event["badge_bg"]};color:{event["badge_text"]};'
                f'font-size:0.75rem;font-weight:600;padding:3px 10px;border-radius:20px;">'
                f'{event["year"]}</span>',
                unsafe_allow_html=True,
            )
            st.markdown(f"#### {event_name}")
            st.caption("  ·  ".join(event["default_indices"]))
            st.write(event["description"])

            if st.button("📖 Open full story →", key=f"open_story_{idx}"):
                st.session_state.story_detail_idx = idx
                # Streamlit cannot switch tabs programmatically, so we guide user
                st.info("👆 Click the **📖 Story Mode** tab above to view the full analysis.")

    # Navigation row
    nav_l, nav_dots, nav_r = st.columns([1, 8, 1])
    with nav_l:
        if st.button("←", key="car_prev", use_container_width=True):
            st.session_state.story_idx = (idx - 1) % total_stories
            st.rerun()
    with nav_dots:
        dots_html = "".join(
            f'<span style="display:inline-block;'
            f'width:{"20px" if i == idx else "7px"};height:7px;'
            f'border-radius:{"3px" if i == idx else "50%"};'
            f'background:{"#1a3a5c" if i == idx else "#c8d8e8"};'
            f'margin:0 3px;vertical-align:middle;transition:all 0.2s;"></span>'
            for i in range(total_stories)
        )
        st.markdown(
            f'<div style="text-align:center;padding:8px 0;">{dots_html}'
            f'<span style="font-size:0.72rem;color:#8aafcf;margin-left:10px;">'
            f'{idx + 1} / {total_stories}</span></div>',
            unsafe_allow_html=True,
        )
    with nav_r:
        if st.button("→", key="car_next", use_container_width=True):
            st.session_state.story_idx = (idx + 1) % total_stories
            st.rerun()

    st.divider()

    # ── Main dashboard columns ────────────────────────────────────────────────
    left_col, right_col = st.columns([5, 4], gap="medium")

    # ── Climate Index Time Series ─────────────────────────────────────────────
    with left_col:
        st.markdown('<p class="section-label">Climate Index Time Series</p>',
                    unsafe_allow_html=True)
        st.caption("Multidecadal trends of the selected climate indices")

        fig_ts = go.Figure()

        for idx_name in df.columns:

            if idx_name == "Year":
                continue

            fig_ts.add_trace(go.Scatter(
                x=df["Year"],
                y=df[idx_name],
                name=idx_name,
                line=dict(color=INDEX_META[idx_name]["color"], width=1.6),
                hovertemplate=f"<b>{idx_name}</b><br>Year: %{{x}}<br>Value: %{{y:.3f}}<extra></extra>",
            ))

        # Regression trend on primary index
        # primary = selected_indices[0]
        primary = [c for c in df.columns if c != "Year"][0]
        x_arr = df["Year"].values.astype(float)
        y_arr = df[primary].values
        slope, intercept, r_val, p_val, _ = stats.linregress(x_arr, y_arr)
        trend_y = slope * x_arr + intercept

        fig_ts.add_trace(go.Scatter(
            x=df["Year"], y=trend_y,
            name="Regression trend",
            line=dict(color="#27ae60", width=2, dash="dot"),
            hovertemplate="Trend: %{y:.3f}<extra></extra>",
        ))

        # Callout annotations for last data point of each index
        last_yr = df["Year"].max()
        offsets = [(-70, -20), (-70, 10), (-70, -50)]
        for i, idx_name in enumerate(available_indices[:3]):
            val = float(df.loc[df["Year"] == last_yr, idx_name].iloc[0])
            ax, ay = offsets[i % len(offsets)]
            fig_ts.add_annotation(
                x=last_yr, y=val,
                text=f"<b>{idx_name} ({last_yr}): {val:+.2f}</b>",
                showarrow=True, arrowhead=2, arrowsize=1,
                bgcolor=INDEX_META[idx_name]["color"], opacity=0.9,
                font=dict(color="white", size=10),
                arrowcolor=INDEX_META[idx_name]["color"],
                ax=ax, ay=ay,
            )

        fig_ts.update_layout(
            xaxis_title="Time", yaxis_title="Multidecadal trends",
            legend=dict(orientation="h", yanchor="bottom",
                        y=1.02, xanchor="left", x=0),
            height=330, margin=dict(l=50, r=20, t=60, b=40),
            plot_bgcolor="white",
            xaxis=dict(showgrid=True, gridcolor="#ebebeb"),
            yaxis=dict(showgrid=True, gridcolor="#ebebeb", zeroline=True,
                       zerolinecolor="#ccc"),
        )
        st.plotly_chart(fig_ts, use_container_width=True)

        # ── Decadal Average Bar Chart ─────────────────────────────────────────
        st.markdown('<p class="section-label">Climate Oscillation Decadal Average</p>',
                    unsafe_allow_html=True)

        # dec_df = decadal_averages(df, selected_indices)
        indices = [c for c in df.columns if c != "Year"]
        dec_df = decadal_averages(df, indices)

        fig_bar = go.Figure()
        for idx_name in indices:
            fig_bar.add_trace(go.Bar(
                name=idx_name, x=dec_df["Label"], y=dec_df[idx_name],
                marker_color=INDEX_META[idx_name]["color"],
                hovertemplate=(f"<b>{idx_name}</b><br>Decade: %{{x}}<br>"
                               f"Avg: %{{y:.3f}}<extra></extra>"),
            ))

        fig_bar.update_layout(
            barmode="group",
            xaxis_title="Decade", yaxis_title="Value",
            height=290, margin=dict(l=50, r=20, t=30, b=40),
            plot_bgcolor="white",
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor="#ebebeb",
                       zeroline=True, zerolinecolor="#ccc"),
            legend=dict(orientation="h", y=1.08, x=0),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # ── Right column — maps ───────────────────────────────────────────────────
    with right_col:
        DASH_MAP_LABELS = {
            "sst_trends_ann":       ("Global SST Annual Trend",  "Trend (°C/yr)"),
            "sst_trends_mon":       ("Global SST Monthly Trend", "Trend (°C/mon)"),
            "amo_pattern_mon":      ("AMO Spatial Pattern",      "Loading Factor"),
            "pdo_pattern_mon":      ("PDO Spatial Pattern",      "Loading Factor"),
            "sst_spatialmean_ann":  ("SST Mean (Annual)",        "SST (°C)"),
            "sst_spatialstddev_ann":("SST Variability",          "Std Dev (°C)"),
        }
        _map_title, _map_cb = DASH_MAP_LABELS.get(climate_var, ("Spatial Pattern", "Value"))
    # ── Global SST Trend Map ─────────────────────────────────────
        st.markdown(
            f'<p class="section-label">{_map_title}</p>',
            unsafe_allow_html=True
        )

        if dataset_loaded:

            try:
                _rb_use = None if region == "Global" else _rb
                lats, lons, temp_grid = load_temperature_map(climate_var, region_bounds=_rb_use)

                fig_globe = go.Figure(go.Heatmap(
                    z=temp_grid,
                    x=lons,
                    y=lats,
                    colorscale="RdBu_r",
                    zmid=0,
                    colorbar=dict(title=_map_cb),
                    hovertemplate=f"Lat: %{{y:.1f}}°  Lon: %{{x:.1f}}°<br>{_map_cb}: %{{z:.4f}}<extra></extra>",
                ))

                fig_globe.update_layout(
                    xaxis_title="Longitude",
                    yaxis_title="Latitude",
                    height=265,
                    margin=dict(l=50, r=10, t=10, b=40),
                )

                st.plotly_chart(fig_globe, use_container_width=True)

            except Exception as e:
                st.warning(f"Could not load spatial SST trend map: {e}")

        else:
            st.info("Upload a NetCDF dataset to view SST spatial patterns.")

        # ── AMO Spatial Pattern ─────────────────────────────────────
        st.markdown(
            '<p class="section-label">AMO Spatial Pattern</p>',
            unsafe_allow_html=True
        )

        if dataset_loaded:

            try:
                amo_lats, amo_lons, amo_grid = load_temperature_map("amo_pattern_mon")

                fig_amo = go.Figure(go.Heatmap(
                    z=amo_grid,
                    x=amo_lons,
                    y=amo_lats,
                    colorscale="RdBu_r",
                    zmid=0,
                    colorbar=dict(title="Loading Factor"),
                    hovertemplate="Lat: %{y:.1f}°  Lon: %{x:.1f}°<br>Loading: %{z:.3f}<extra></extra>",
                ))

                fig_amo.update_layout(
                    xaxis_title="Longitude",
                    yaxis_title="Latitude",
                    height=265,
                    margin=dict(l=50, r=10, t=10, b=40),
                )

                st.plotly_chart(fig_amo, use_container_width=True)

            except Exception as e:
                st.warning(f"Could not load AMO spatial pattern: {e}")

        else:
            st.info("Upload dataset to visualize AMO spatial patterns.")


# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 2 — ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

with tab_analysis:
    st.markdown("### Advanced Climate Analysis")

    if not dataset_loaded:
        st.info(
            "📂 Upload a NetCDF file in the sidebar to unlock real analysis.",
            icon="📊",
        )
    else:
        col_a, col_b = st.columns(2, gap="medium")

        with col_a:
            st.markdown("#### Linear Trend Analysis")
            records = []
            for idx_name in available_indices:
                y = df[idx_name].values
                x = df["Year"].values.astype(float)
                slope, intercept, r_val, p_val, _ = stats.linregress(x, y)
                records.append({
                    "Index": idx_name,
                    "Trend / decade": f"{slope * 10:+.4f}",
                    "Intercept": f"{intercept:.4f}",
                    "R²": f"{r_val ** 2:.4f}",
                    "p-value": f"{p_val:.4f}",
                    "Significant": "✅" if p_val < 0.05 else "❌",
                })
            st.dataframe(pd.DataFrame(records), use_container_width=True, hide_index=True)

            primary = available_indices[0]
            x = df["Year"].values.astype(float)
            y = df[primary].values
            slope, intercept, r_val, p_val, _ = stats.linregress(x, y)

            fig_trend = go.Figure()
            fig_trend.add_trace(go.Scatter(
                x=df["Year"], y=y, name=primary,
                line=dict(color=INDEX_META[primary]["color"], width=1.4),
                opacity=0.75,
            ))
            fig_trend.add_trace(go.Scatter(
                x=df["Year"], y=slope * x + intercept,
                name="Linear Trend",
                line=dict(color="#c0392b", width=2.2, dash="dash"),
            ))
            fig_trend.update_layout(
                title=dict(
                    text=f"{primary}  |  trend: {slope*10:+.4f}/decade  |  R² = {r_val**2:.3f}",
                    font=dict(size=12)
                ),
                xaxis_title="Year", yaxis_title="Index Value",
                height=300, plot_bgcolor="white",
                margin=dict(l=50, r=20, t=50, b=40),
                legend=dict(orientation="h", y=1.12),
                xaxis=dict(showgrid=True, gridcolor="#ebebeb"),
                yaxis=dict(showgrid=True, gridcolor="#ebebeb"),
            )
            st.plotly_chart(fig_trend, use_container_width=True)

        with col_b:
            st.markdown("#### Correlation Matrix")
            indices = [c for c in df.columns if c != "Year"]
            corr = df[indices].corr()
            fig_corr = go.Figure(go.Heatmap(
                z=corr.values,
                x=corr.columns.tolist(),
                y=corr.index.tolist(),
                colorscale="RdBu_r", zmid=0, zmin=-1, zmax=1,
                text=np.round(corr.values, 2),
                texttemplate="%{text}",
                colorbar=dict(title="r", thickness=12),
                hovertemplate="X: %{x}<br>Y: %{y}<br>r = %{z:.3f}<extra></extra>",
            ))
            fig_corr.update_layout(height=310, margin=dict(l=20, r=20, t=10, b=20))
            st.plotly_chart(fig_corr, use_container_width=True)

            st.markdown("#### Index Comparison (Z-score Normalised)")
            fig_norm = go.Figure()
            for idx_name in available_indices:
                z = (df[idx_name] - df[idx_name].mean()) / df[idx_name].std()
                fig_norm.add_trace(go.Scatter(
                    x=df["Year"], y=z, name=idx_name,
                    line=dict(color=INDEX_META[idx_name]["color"], width=1.5),
                    hovertemplate=f"<b>{idx_name}</b>  z = %{{y:.2f}}<extra></extra>",
                ))
            fig_norm.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
            fig_norm.update_layout(
                xaxis_title="Year", yaxis_title="Z-score",
                height=275, plot_bgcolor="white",
                margin=dict(l=50, r=20, t=10, b=40),
                legend=dict(orientation="h", y=1.12),
                xaxis=dict(showgrid=True, gridcolor="#ebebeb"),
                yaxis=dict(showgrid=True, gridcolor="#ebebeb",
                           zeroline=True, zerolinecolor="#bbb"),
            )
            st.plotly_chart(fig_norm, use_container_width=True)

        st.divider()
        st.markdown("#### Rolling Decadal Mean")
        window = st.slider("Window (years)", 5, 30, 10, key="win")
        fig_roll = go.Figure()
        for idx_name in df.columns:
            if idx_name == "Year":
                continue
            raw = df[idx_name].values
            rolled = pd.Series(raw).rolling(window, center=True).mean()
            fig_roll.add_trace(go.Scatter(
                x=df["Year"], y=raw, name=f"{idx_name} (raw)",
                line=dict(color=INDEX_META[idx_name]["color"], width=0.9),
                opacity=0.35, showlegend=False,
            ))
            fig_roll.add_trace(go.Scatter(
                x=df["Year"], y=rolled, name=f"{idx_name} ({window}-yr mean)",
                line=dict(color=INDEX_META[idx_name]["color"], width=2.5),
            ))
        fig_roll.update_layout(
            xaxis_title="Year", yaxis_title="Index Value",
            height=300, plot_bgcolor="white",
            margin=dict(l=50, r=20, t=10, b=40),
            legend=dict(orientation="h", y=1.1),
            xaxis=dict(showgrid=True, gridcolor="#ebebeb"),
            yaxis=dict(showgrid=True, gridcolor="#ebebeb"),
        )
        st.plotly_chart(fig_roll, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 3 — 3D GLOBE VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════════════

# ── Map of user-facing labels → NetCDF variable names ────────────────────────
GLOBE_SPATIAL_VARS = {
    "SST Annual Trend":          "sst_trends_ann",
    "SST Monthly Trend":         "sst_trends_mon",
    "AMO Spatial Pattern":       "amo_pattern_mon",
    "PDO Spatial Pattern":       "pdo_pattern_mon",
    "SST Mean (Annual)":         "sst_spatialmean_ann",
    "SST Variability (Std Dev)": "sst_spatialstddev_ann",
}

COLORBAR_LABELS = {
    "SST Annual Trend":          "Trend (°C/yr)",
    "SST Monthly Trend":         "Trend (°C/month)",
    "AMO Spatial Pattern":       "Loading Factor",
    "PDO Spatial Pattern":       "Loading Factor",
    "SST Mean (Annual)":         "SST (°C)",
    "SST Variability (Std Dev)": "Std Dev (°C)",
}

with tab_globe:
    st.markdown("### 🌐 3D Globe — Spatial Climate Patterns")
    # CHANGED caption — no longer claims data is synthetic
    st.caption(
        "Upload a NetCDF dataset to visualise real spatial fields. "
        "Drag to rotate · Scroll to zoom · Hover for values."
    )

    # ── Controls ──────────────────────────────────────────────────────────────
    # CHANGED: 4 columns when dataset loaded (added variable selector), 3 otherwise
    if dataset_loaded:
        g_col1, g_col2, g_col3, g_col4 = st.columns([2, 2, 3, 3])
        with g_col1:
            globe_projection = st.selectbox(
                "Projection",
                ["orthographic", "natural earth", "equirectangular",
                 "mercator", "mollweide"],
                key="globe_proj"
            )
        with g_col2:
            globe_colorscale = st.selectbox(
                "Colour Scale",
                ["RdBu_r", "Thermal", "Picnic", "Portland", "Jet"],
                key="globe_cs"
            )
        with g_col3:
            # CHANGED: variable selector only appears when real data is loaded
            globe_var_label = st.selectbox(
                "Spatial Variable",
                list(GLOBE_SPATIAL_VARS.keys()),
                key="globe_var"
            )
        with g_col4:
            # CHANGED: year slider hidden for real data (spatial fields are time-averaged)
            st.caption("📅 Year slider N/A — spatial fields are time-averaged (1900–2018)")
            globe_year = time_range[1]   # keeps downstream stats consistent
    else:
        g_col1, g_col2, g_col3 = st.columns([2, 2, 3])
        with g_col1:
            # Year slider only meaningful for synthetic fallback
            globe_year = st.slider("Year", 1900, 2023, time_range[1], key="globe_year")
        with g_col2:
            globe_projection = st.selectbox(
                "Projection",
                ["orthographic", "natural earth", "equirectangular",
                 "mercator", "mollweide"],
                key="globe_proj"
            )
        with g_col3:
            globe_colorscale = st.selectbox(
                "Colour Scale",
                ["RdBu_r", "Thermal", "Picnic", "Portland", "Jet"],
                key="globe_cs"
            )
        globe_var_label = None

    # ── Resolve the grid — real data or synthetic fallback ────────────────────
    # CHANGED: this entire block replaces the single generate_temperature_map() call
    if dataset_loaded and globe_var_label:
        try:
            lats_g, lons_g, temp_g = get_spatial_pattern(
                GLOBE_SPATIAL_VARS[globe_var_label]
            )
            colorbar_label = COLORBAR_LABELS[globe_var_label]
            globe_title    = f"🌍  {globe_var_label}  ·  ERSST 1900–2018"
            # Dynamic symmetric colour range from real data
            abs_max = float(np.nanmax(np.abs(temp_g)))
            cmin, cmax = -abs_max, abs_max
        except Exception as e:
            st.warning(f"Could not load spatial field: {e}. Showing synthetic fallback.")
            lats_g, lons_g, temp_g = generate_temperature_map(globe_year)
            colorbar_label = "Anomaly (°C)"
            globe_title    = f"🌍  Temperature Anomaly (synthetic)  ·  {globe_year}"
            cmin, cmax     = -2.5, 2.5
    else:
        lats_g, lons_g, temp_g = generate_temperature_map(globe_year)
        colorbar_label = "Anomaly (°C)"
        globe_title    = f"🌍  Temperature Anomaly (synthetic)  ·  {globe_year}"
        cmin, cmax     = -2.5, 2.5

    # NaN-safe flatten — real grids may contain fill values
    # CHANGED: added np.nan_to_num so Scattergeo doesn't choke on NaNs
    temp_g_clean = np.where(np.isfinite(temp_g), temp_g, np.nan)

    LAT_G, LON_G  = np.meshgrid(lats_g, lons_g, indexing="ij")
    flat_lat  = LAT_G[::3, ::3].ravel()
    flat_lon  = LON_G[::3, ::3].ravel()
    flat_temp = temp_g_clean[::3, ::3].ravel()

    # Drop NaN points entirely — Scattergeo renders gaps as missing, not errors
    # CHANGED: NaN filter
    valid     = np.isfinite(flat_temp)
    flat_lat  = flat_lat[valid]
    flat_lon  = flat_lon[valid]
    flat_temp = flat_temp[valid]

    fig_3d = go.Figure()

    fig_3d.add_trace(go.Scattergeo(
        lat=flat_lat,
        lon=flat_lon,
        mode="markers",
        marker=dict(
            size=4,
            color=flat_temp,
            colorscale=globe_colorscale,
            cmin=cmin, cmax=cmax,           # CHANGED: was hardcoded -2.5 / 2.5
            colorbar=dict(
                title=dict(text=colorbar_label, font=dict(size=11)),  # CHANGED: dynamic label
                thickness=14, len=0.65, x=1.01,
            ),
            opacity=0.85,
            symbol="circle",
            line=dict(width=0),
        ),
        hovertemplate=(
            "Lat: %{lat:.1f}°  Lon: %{lon:.1f}°<br>"
            f"{colorbar_label}: " + "%{marker.color:.4f}<extra></extra>"  # CHANGED: dynamic label
        ),
        name="Temp Anomaly",
    ))

    # ── Key city markers overlay ───────────────────────────────────────────────
    cities = {
        "Delhi":      (28.6,  77.2),
        "New York":   (40.7, -74.0),
        "London":     (51.5,  -0.1),
        "Beijing":    (39.9, 116.4),
        "Sydney":    (-33.9, 151.2),
        "São Paulo": (-23.5, -46.6),
        "Cape Town": (-33.9,  18.4),
        "Reykjavik":  (64.1, -21.9),
    }
    city_lats  = [v[0] for v in cities.values()]
    city_lons  = [v[1] for v in cities.values()]
    city_names = list(cities.keys())

    city_anoms = []
    for clat, clon in zip(city_lats, city_lons):
        ilat = int(np.argmin(np.abs(lats_g - clat)))
        ilon = int(np.argmin(np.abs(lons_g - clon)))
        # CHANGED: guard against NaN at nearest gridpoint
        val = float(temp_g_clean[ilat, ilon])
        city_anoms.append(round(val, 4) if np.isfinite(val) else 0.0)

    fig_3d.add_trace(go.Scattergeo(
        lat=city_lats, lon=city_lons,
        mode="markers+text",
        marker=dict(size=8, color="yellow", symbol="star",
                    line=dict(color="black", width=1)),
        text=city_names,
        textposition="top right",
        textfont=dict(size=9, color="white"),
        hovertemplate=(
            "<b>%{text}</b><br>"
            f"{colorbar_label}: " + "%{customdata:.4f}<extra></extra>"  # CHANGED
        ),
        customdata=city_anoms,
        name="Major Cities",
    ))

    # ── Layout ────────────────────────────────────────────────────────────────
    fig_3d.update_geos(
        projection_type=globe_projection,
        showland=True,        landcolor="#2d4a2d",
        showocean=True,       oceancolor="#0a1628",
        showcoastlines=True,  coastlinecolor="#5a8fa0", coastlinewidth=0.7,
        showlakes=True,       lakecolor="#0a1628",
        showrivers=False,
        showframe=False,
        bgcolor="#0d1b2a",
        lataxis_showgrid=True, lonaxis_showgrid=True,
        lataxis_gridcolor="rgba(255,255,255,0.08)",
        lonaxis_gridcolor="rgba(255,255,255,0.08)",
    )
    fig_3d.update_layout(
        title=dict(
            text=globe_title,                          # CHANGED: was hardcoded string
            font=dict(size=15, color="#c8dff0"),
            x=0.5, xanchor="center",
        ),
        geo=dict(bgcolor="#0d1b2a"),
        paper_bgcolor="#0d1b2a",
        plot_bgcolor="#0d1b2a",
        font=dict(color="#c8dff0"),
        height=620,
        margin=dict(l=0, r=0, t=50, b=0),
        legend=dict(
            orientation="h", x=0.5, xanchor="center", y=-0.02,
            font=dict(size=10, color="#c8dff0"),
            bgcolor="rgba(0,0,0,0)",
        ),
    )

    st.plotly_chart(fig_3d, use_container_width=True)

    # ── Stats row below the globe ─────────────────────────────────────────────
    st.divider()
    s1, s2, s3, s4 = st.columns(4)
    # CHANGED: np.nanmean/nanmax/nanmin/nanstd — real grids may have NaN edges
    s1.metric("Global Mean",  f"{np.nanmean(flat_temp):+.4f}")
    s2.metric("Max Value",    f"{np.nanmax(flat_temp):+.4f}",
              delta=f"Lat {lats_g[np.unravel_index(np.nanargmax(temp_g_clean), temp_g_clean.shape)[0]]:.0f}°")
    s3.metric("Min Value",    f"{np.nanmin(flat_temp):+.4f}")
    s4.metric("Std Dev",      f"{np.nanstd(flat_temp):.4f}")
# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 5 — STORY MODE
# ═══════════════════════════════════════════════════════════════════════════════

with tab_stories:
    st.markdown("### 📖 Climate Story Mode")
    st.caption(
        "Guided tours through history's most dramatic climate events. "
        "Use the carousel on the Dashboard to browse, or pick an event below."
    )

    # Pre-select card that was clicked in the carousel (falls back to 0)
    detail_idx = st.session_state.get("story_detail_idx", 0)
    story_keys_list = list(story_mode_data.keys())

    chosen = st.selectbox(
        "Select a climate event",
        story_keys_list,
        index=detail_idx,
        key="story_select",
    )
    ev = story_mode_data[chosen]

    st.divider()

    # ── Event detail card ─────────────────────────────────────────────────────
    img_c, info_c = st.columns([2, 3], gap="large")

    with img_c:
        local_imgs = ev.get("local_images", [])
        if local_imgs:
            # Show each local image with its own caption
            for fname, cap in local_imgs:
                img_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), fname)
                if os.path.exists(img_path):
                    st.image(img_path, use_container_width=True, caption=cap)
                else:
                    st.caption(f"_(image file not found: {fname})_")
        else:
            # Fall back to remote URL
            try:
                st.image(ev["image_url"], use_container_width=True,
                         caption=f"{chosen} · {ev['year']}")
            except Exception:
                st.info("Image unavailable for this event.")

    with info_c:
        st.markdown(
            f'<div style="height:3px;width:56px;border-radius:2px;'
            f'background:{ev["accent"]};margin-bottom:12px;"></div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<span style="background:{ev["badge_bg"]};color:{ev["badge_text"]};'
            f'font-size:0.78rem;font-weight:600;padding:4px 12px;border-radius:20px;">'
            f'{ev["year"]}</span>',
            unsafe_allow_html=True,
        )
        st.markdown(f"## {chosen}")
        st.write(ev["description"])

        st.markdown("**Relevant climate indices**")
        pills_html = " ".join(
            f'<span style="display:inline-block;background:#f0f5fb;'
            f'color:#1a3a5c;font-size:0.78rem;font-weight:600;'
            f'padding:3px 12px;border-radius:20px;margin:2px;">{i}</span>'
            for i in ev["default_indices"]
        )
        st.markdown(pills_html, unsafe_allow_html=True)

    st.divider()

    # ── Index time series centred on the event year ───────────────────────────
    st.markdown(f"#### Index time series around {ev['year']} (±15 years)")

    # ADDED — data source caption
    if dataset_loaded:
        st.caption("📡 AMO · PDO · ENSO use real uploaded data. NAO and SOI are synthetic estimates.")
    else:
        st.caption("⚠️ No dataset uploaded — all index values shown are synthetic estimates.")

    yr = ev["year"]
    _story_base = df_all.copy()
    if dataset_loaded:
        for _col in [c for c in df.columns if c != "Year"]:
            _story_base = _story_base.merge(
                df[["Year", _col]].rename(columns={_col: _col + "_real"}),
                on="Year", how="left"
            )
            _story_base[_col] = _story_base[_col + "_real"].combine_first(_story_base[_col])
            _story_base.drop(columns=[_col + "_real"], inplace=True)
        _yr_max = 2018   # ADDED
    else:
        _yr_max = 2023   # ADDED

    df_story = _story_base[
        (_story_base["Year"] >= max(1900, yr - 15)) &
        (_story_base["Year"] <= min(_yr_max, yr + 15))   # already correct
    ].copy()

    fig_story = go.Figure()
    for si in ev["default_indices"]:
        if si not in df_story.columns:   # ADDED guard
            continue
        fig_story.add_trace(go.Scatter(
            x=df_story["Year"], y=df_story[si],
            name=si,
            line=dict(color=INDEX_META[si]["color"], width=2),
            hovertemplate=f"<b>{si}</b>  %{{y:.3f}}<extra></extra>",
        ))

    fig_story.add_vline(
        x=yr, line_dash="dash", line_color=ev["accent"], line_width=2,
        annotation_text=f"  {yr}",
        annotation_position="top right",
        annotation_font=dict(color=ev["accent"], size=12),
    )
    fig_story.update_layout(
        height=340, plot_bgcolor="white",
        xaxis_title="Year", yaxis_title="Index Value",
        xaxis=dict(showgrid=True, gridcolor="#ebebeb"),
        yaxis=dict(showgrid=True, gridcolor="#ebebeb",
                   zeroline=True, zerolinecolor="#ccc"),
        margin=dict(l=50, r=20, t=30, b=40),
        legend=dict(orientation="h", y=1.1),
    )
    st.plotly_chart(fig_story, use_container_width=True)

    st.markdown(f"#### Index statistics for {max(1900, yr-15)}–{min(_yr_max, yr+15)}")  # CHANGED 2023 → _yr_max
    stat_records = []
    for si in ev["default_indices"]:
        if si not in df_story.columns:   # ADDED guard
            continue
        s = df_story[si]
        stat_records.append({
            "Index": si,
            "Mean":  f"{s.mean():+.4f}",
            "Std":   f"{s.std():.4f}",
            "Min":   f"{s.min():+.4f}",
            "Max":   f"{s.max():+.4f}",
        })
    st.dataframe(pd.DataFrame(stat_records), use_container_width=True,
                 hide_index=True)
# ═══════════════════════════════════════════════════════════════════════════════
#  TAB — ML FORECAST
# ═══════════════════════════════════════════════════════════════════════════════

with tab_ml:
    st.markdown("### 🤖 ML Climate Index Forecasting")
    st.caption(
        "Polynomial Ridge Regression fitted on historical index data. "
        "Upload a NetCDF file for real data, or use synthetic estimates."
    )

    if not dataset_loaded:
        st.info(
            "📂 No dataset uploaded — forecasting will run on synthetic index estimates. "
            "Upload a NetCDF file in the sidebar for real data.",
            icon="⚠️",
        )

    st.divider()

    # ── Controls ──────────────────────────────────────────────────────────────
    ml_c1, ml_c2, ml_c3 = st.columns([2, 2, 2])

    with ml_c1:
        ml_index = st.selectbox(
            "Select Index to Forecast",
            available_indices,
            key="ml_index"
        )
    with ml_c2:
        forecast_horizon = st.slider(
            "Forecast Horizon (years)", 5, 20, 12, key="ml_horizon"
        )
    with ml_c3:
        show_confidence = st.checkbox(
            "Show uncertainty band", value=True, key="ml_ci"
        )

    st.divider()

    if not dataset_loaded and ml_index not in [k for k,v in INDEX_MAP.items() if v]:
        st.warning(
            f"⚠️ {ml_index} is not available in the ERSST dataset. "
            f"Forecast is running on synthetic estimates."
        )
    # ── Run model ─────────────────────────────────────────────────────────────
    try:
        hist_years, hist_trend, future_years, future_forecast = generate_forecast(
            df, ml_index, forecast_horizon
        )

        # ── Main forecast chart ───────────────────────────────────────────────
        fig_ml = go.Figure()

        # Raw historical values
        fig_ml.add_trace(go.Scatter(
            x=df["Year"],
            y=df[ml_index],
            name=f"{ml_index} (observed)",
            mode="lines",
            line=dict(color=INDEX_META[ml_index]["color"], width=1.5),
            opacity=0.6,
            hovertemplate="<b>Observed</b><br>Year: %{x}<br>Value: %{y:.4f}<extra></extra>",
        ))

        # Model fit on historical data
        fig_ml.add_trace(go.Scatter(
            x=hist_years,
            y=hist_trend,
            name="Model fit (historical)",
            mode="lines",
            line=dict(color="#2ecc71", width=2, dash="dot"),
            hovertemplate="<b>Model fit</b><br>Year: %{x}<br>Value: %{y:.4f}<extra></extra>",
        ))

        # Forecast line
        fig_ml.add_trace(go.Scatter(
            x=future_years,
            y=future_forecast,
            name=f"Forecast ({int(future_years[0])}–{int(future_years[-1])})",
            mode="lines+markers",
            line=dict(color="#e74c3c", width=2.5),
            marker=dict(size=5),
            hovertemplate="<b>Forecast</b><br>Year: %{x}<br>Value: %{y:.4f}<extra></extra>",
        ))

        # Optional uncertainty band — simple ±1 std of residuals
        if show_confidence:
            residuals = df[ml_index].values - hist_trend
            sigma = float(np.std(residuals))
            fig_ml.add_trace(go.Scatter(
                x=np.concatenate([future_years, future_years[::-1]]),
                y=np.concatenate([
                    future_forecast + sigma,
                    (future_forecast - sigma)[::-1]
                ]),
                fill="toself",
                fillcolor="rgba(231, 76, 60, 0.12)",
                line=dict(color="rgba(0,0,0,0)"),
                name="±1σ uncertainty",
                hoverinfo="skip",
            ))

        # Vertical line at forecast start
        fig_ml.add_vline(
            x=int(hist_years[-1]),
            line_dash="dash",
            line_color="#888888",
            line_width=1.5,
            annotation_text="  Forecast start",
            annotation_position="top right",
            annotation_font=dict(size=10, color="#888888"),
        )

        fig_ml.update_layout(
            xaxis_title="Year",
            yaxis_title=f"{ml_index} Value",
            height=400,
            plot_bgcolor="white",
            margin=dict(l=55, r=20, t=40, b=45),
            legend=dict(orientation="h", y=1.12),
            xaxis=dict(showgrid=True, gridcolor="#ebebeb"),
            yaxis=dict(showgrid=True, gridcolor="#ebebeb",
                       zeroline=True, zerolinecolor="#cccccc"),
        )
        st.plotly_chart(fig_ml, use_container_width=True)

        # ── Metrics row ───────────────────────────────────────────────────────
        st.divider()
        m1, m2, m3, m4 = st.columns(4)

        residuals      = df[ml_index].values - hist_trend
        rmse           = float(np.sqrt(np.mean(residuals ** 2)))
        mae            = float(np.mean(np.abs(residuals)))
        forecast_delta = float(future_forecast[-1] - future_forecast[0])
        last_obs       = float(df[ml_index].dropna().iloc[-1])

        m1.metric("RMSE (fit)",        f"{rmse:.4f}")
        m2.metric("MAE (fit)",         f"{mae:.4f}")
        m3.metric(
            f"Forecast change over {forecast_horizon} yr",
            f"{forecast_delta:+.4f}"
        )
        m4.metric(
            f"Forecast {int(future_years[-1])}",
            f"{future_forecast[-1]:+.4f}",
            delta=f"{future_forecast[-1] - last_obs:+.4f} vs last obs"
        )

        # ── Forecast table ────────────────────────────────────────────────────
        st.divider()
        st.markdown("#### Forecast Values")
        forecast_df = pd.DataFrame({
            "Year":     future_years.astype(int),
            "Forecast": np.round(future_forecast, 4),
        })
        if show_confidence:
            forecast_df["Lower (−1σ)"] = np.round(future_forecast - sigma, 4)
            forecast_df["Upper (+1σ)"] = np.round(future_forecast + sigma, 4)

        st.dataframe(forecast_df, use_container_width=True, hide_index=True)

    except Exception as e:
        st.error(f"Forecasting failed: {e}")
# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 5 — DATA SOURCES
# ═══════════════════════════════════════════════════════════════════════════════

with tab_sources:
    st.markdown("### Dataset Information")
    st.info(
        "PyClimaExplorer supports standard **NetCDF (.nc)** files. "
        "Upload a file in the sidebar to switch from synthetic demo data to real observations.",
        icon="📂",
    )

    info = DATASET_INFO[dataset]
    cols = st.columns(3)
    for i, (k, v) in enumerate(info.items()):
        with cols[i % 3]:
            st.metric(k, v)

    st.divider()
    st.markdown("#### How to obtain data")

    guide_data = {
        "Dataset": list(DATASET_INFO.keys()),
        "Direct URL": [d["URL"] for d in DATASET_INFO.values()],
        "Python snippet": [
            "import cdsapi; c = cdsapi.Client(); c.retrieve('reanalysis-era5-single-levels', {...}, 'era5.nc')",
            "!wget https://esgf-node.llnl.gov/...  # use ESGF search API",
            "import xarray as xr; ds = xr.open_dataset('https://www.ncei.noaa.gov/.../ersst.v5.nc')",
        ],
    }
    st.dataframe(pd.DataFrame(guide_data), use_container_width=True, hide_index=True)

    st.divider()
    

# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 4 — COMPARISON MODE
# ═══════════════════════════════════════════════════════════════════════════════
with tab_compare:
    st.markdown("### ⏮️ Comparison Mode")

    # ── Mode selector — NEW ───────────────────────────────────────────────────
    compare_mode = st.radio(
        "Comparison type",
        ["Spatial Variables (real data)", "Index Time Windows (real data)",
         "Year-by-Year Maps (synthetic fallback)"],
        horizontal=True,
        key="cmp_mode"
    )

    st.divider()

    # ═════════════════════════════════════════════════════════════════════════
    #  MODE 1 — Compare two spatial variables from real data
    # ═════════════════════════════════════════════════════════════════════════
    if compare_mode == "Spatial Variables (real data)":

        if not dataset_loaded:
            st.warning("Upload a NetCDF file to use this mode.")
        else:
            st.caption(
                "Select two spatial fields from the dataset. "
                "Both maps share an identical symmetric colour scale."
            )

            SPATIAL_OPTIONS = {
                "SST Annual Trend":          "sst_trends_ann",
                "SST Monthly Trend":         "sst_trends_mon",
                "AMO Spatial Pattern":       "amo_pattern_mon",
                "PDO Spatial Pattern":       "pdo_pattern_mon",
                "SST Mean (Annual)":         "sst_spatialmean_ann",
                "SST Variability (Std Dev)": "sst_spatialstddev_ann",
            }

            left_col, right_col = st.columns(2, gap="large")

            with left_col:
                var_a_label = st.selectbox(
                    "Left variable", list(SPATIAL_OPTIONS.keys()),
                    index=0, key="cmp_var_a"
                )
            with right_col:
                var_b_label = st.selectbox(
                    "Right variable", list(SPATIAL_OPTIONS.keys()),
                    index=2, key="cmp_var_b"          # default: AMO pattern
                )

            try:
                lats_a, lons_a, grid_a = get_spatial_pattern(SPATIAL_OPTIONS[var_a_label])
                lats_b, lons_b, grid_b = get_spatial_pattern(SPATIAL_OPTIONS[var_b_label])

                # Shared symmetric colour range across BOTH grids
                abs_max = max(
                    float(np.nanmax(np.abs(grid_a))),
                    float(np.nanmax(np.abs(grid_b)))
                )
                SHARED_ZMIN = -abs_max
                SHARED_ZMAX =  abs_max

                with left_col:
                    st.markdown(
                        f'<p class="section-label">{var_a_label}</p>',
                        unsafe_allow_html=True
                    )
                    fig_a = go.Figure(go.Heatmap(
                        z=grid_a, x=lons_a, y=lats_a,
                        colorscale="RdBu_r",
                        zmin=SHARED_ZMIN, zmid=0, zmax=SHARED_ZMAX,
                        colorbar=dict(thickness=12, len=0.75,
                                      tickfont=dict(size=9)),
                        hovertemplate=(
                            "Lat: %{y:.1f}°  Lon: %{x:.1f}°<br>"
                            "Value: %{z:.4f}<extra></extra>"
                        ),
                    ))
                    fig_a.update_layout(
                        xaxis_title="Longitude", yaxis_title="Latitude",
                        height=350, plot_bgcolor="white",
                        margin=dict(l=50, r=20, t=30, b=50),
                        title=dict(text=var_a_label, font=dict(size=12),
                                   x=0.5, xanchor="center"),
                    )
                    st.plotly_chart(fig_a, use_container_width=True)
                    st.metric("Mean", f"{np.nanmean(grid_a):+.4f}")

                with right_col:
                    st.markdown(
                        f'<p class="section-label">{var_b_label}</p>',
                        unsafe_allow_html=True
                    )
                    fig_b = go.Figure(go.Heatmap(
                        z=grid_b, x=lons_b, y=lats_b,
                        colorscale="RdBu_r",
                        zmin=SHARED_ZMIN, zmid=0, zmax=SHARED_ZMAX,
                        colorbar=dict(thickness=12, len=0.75,
                                      tickfont=dict(size=9)),
                        hovertemplate=(
                            "Lat: %{y:.1f}°  Lon: %{x:.1f}°<br>"
                            "Value: %{z:.4f}<extra></extra>"
                        ),
                    ))
                    fig_b.update_layout(
                        xaxis_title="Longitude", yaxis_title="Latitude",
                        height=350, plot_bgcolor="white",
                        margin=dict(l=50, r=20, t=30, b=50),
                        title=dict(text=var_b_label, font=dict(size=12),
                                   x=0.5, xanchor="center"),
                    )
                    st.plotly_chart(fig_b, use_container_width=True)
                    st.metric("Mean", f"{np.nanmean(grid_b):+.4f}")

                # Difference map — only valid if grids share the same lat/lon
                if grid_a.shape == grid_b.shape:
                    st.divider()
                    st.markdown("#### Difference Map — Left minus Right")
                    diff = grid_a - grid_b
                    diff_bound = max(float(np.nanmax(np.abs(diff))), 0.001)

                    fig_diff = go.Figure(go.Heatmap(
                        z=diff, x=lons_a, y=lats_a,
                        colorscale="RdBu_r",
                        zmin=-diff_bound, zmid=0, zmax=diff_bound,
                        colorbar=dict(title="Δ", thickness=14,
                                      tickfont=dict(size=9)),
                        hovertemplate=(
                            "Lat: %{y:.1f}°  Lon: %{x:.1f}°<br>"
                            "Δ: %{z:.4f}<extra></extra>"
                        ),
                    ))
                    fig_diff.update_layout(
                        xaxis_title="Longitude", yaxis_title="Latitude",
                        height=320, plot_bgcolor="white",
                        margin=dict(l=50, r=20, t=20, b=50),
                    )
                    st.plotly_chart(fig_diff, use_container_width=True)

                    d1, d2, d3, d4 = st.columns(4)
                    d1.metric("Mean Δ",   f"{float(np.nanmean(diff)):+.4f}")
                    d2.metric("Max Δ",    f"{float(np.nanmax(diff)):+.4f}")
                    d3.metric("Min Δ",    f"{float(np.nanmin(diff)):+.4f}")
                    d4.metric("Cells A > B",
                              f"{int(np.sum(diff > 0))}",
                              delta=f"{100 * np.nanmean(diff > 0):.1f}%")
                else:
                    st.info(
                        "Difference map unavailable — the two variables have "
                        "different grid shapes."
                    )

            except Exception as e:
                st.error(f"Could not load spatial data: {e}")

    # ═════════════════════════════════════════════════════════════════════════
    #  MODE 2 — Compare two time windows of a real index
    # ═════════════════════════════════════════════════════════════════════════
    elif compare_mode == "Index Time Windows (real data)":

        if not dataset_loaded:
            st.warning("Upload a NetCDF file to use this mode.")
        else:
            st.caption(
                "Split one index into two periods and compare their distributions."
            )

            c1, c2, c3 = st.columns([2, 2, 2])
            with c1:
                cmp_index = st.selectbox(
                    "Index", available_indices, key="cmp_idx"
                )
            with c2:
                win_a = st.slider(
                    "Period A", 1900, 2018, (1900, 1958), key="cmp_win_a"
                )
            with c3:
                win_b = st.slider(
                    "Period B", 1900, 2018, (1959, 2018), key="cmp_win_b"
                )

            # df already contains real index data (built in DATA PREPARATION)
            idx_col = cmp_index
            if idx_col not in df.columns:
                st.error(f"{cmp_index} not in loaded dataframe.")
            else:
                period_a = df[(df["Year"] >= win_a[0]) & (df["Year"] <= win_a[1])][["Year", idx_col]]
                period_b = df[(df["Year"] >= win_b[0]) & (df["Year"] <= win_b[1])][["Year", idx_col]]

                left_col, right_col = st.columns(2, gap="large")

                # Shared y-axis range
                y_all = pd.concat([period_a[idx_col], period_b[idx_col]]).dropna()
                y_min, y_max = float(y_all.min()), float(y_all.max())
                pad = (y_max - y_min) * 0.1

                def _period_fig(period_df, label, color):
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=period_df["Year"], y=period_df[idx_col],
                        name=label,
                        line=dict(color=color, width=1.8),
                        hovertemplate=f"<b>{label}</b><br>Year: %{{x}}<br>Value: %{{y:.4f}}<extra></extra>",
                    ))
                    # Trend line
                    x_arr = period_df["Year"].values.astype(float)
                    y_arr = period_df[idx_col].values
                    if len(x_arr) >= 3:
                        slope, intercept, r, p, _ = stats.linregress(x_arr, y_arr)
                        fig.add_trace(go.Scatter(
                            x=period_df["Year"],
                            y=slope * x_arr + intercept,
                            name="Trend",
                            line=dict(color="#c0392b", width=2, dash="dash"),
                        ))
                    fig.update_layout(
                        height=300, plot_bgcolor="white",
                        yaxis=dict(range=[y_min - pad, y_max + pad],
                                   showgrid=True, gridcolor="#ebebeb"),
                        xaxis=dict(showgrid=True, gridcolor="#ebebeb"),
                        margin=dict(l=50, r=20, t=40, b=40),
                        title=dict(text=label, font=dict(size=12),
                                   x=0.5, xanchor="center"),
                        legend=dict(orientation="h", y=1.12),
                    )
                    return fig

                with left_col:
                    label_a = f"{cmp_index}  {win_a[0]}–{win_a[1]}"
                    st.plotly_chart(
                        _period_fig(period_a, label_a, "#1f77b4"),
                        use_container_width=True
                    )
                    ma = period_a[idx_col].mean()
                    sa = period_a[idx_col].std()
                    st.metric(f"Mean  ({win_a[0]}–{win_a[1]})", f"{ma:+.4f}")
                    st.metric("Std Dev", f"{sa:.4f}")

                with right_col:
                    label_b = f"{cmp_index}  {win_b[0]}–{win_b[1]}"
                    st.plotly_chart(
                        _period_fig(period_b, label_b, "#ff7f0e"),
                        use_container_width=True
                    )
                    mb = period_b[idx_col].mean()
                    sb = period_b[idx_col].std()
                    st.metric(f"Mean  ({win_b[0]}–{win_b[1]})",
                              f"{mb:+.4f}",
                              delta=f"{mb - ma:+.4f} vs Period A")
                    st.metric("Std Dev", f"{sb:.4f}",
                              delta=f"{sb - sa:+.4f} vs Period A")

                # Distribution comparison
                st.divider()
                st.markdown("#### Value Distribution")
                fig_hist = go.Figure()
                fig_hist.add_trace(go.Histogram(
                    x=period_a[idx_col].dropna(), name=f"Period A  {win_a[0]}–{win_a[1]}",
                    marker_color="#1f77b4", opacity=0.65,
                    hovertemplate="Value: %{x:.3f}<br>Count: %{y}<extra></extra>",
                ))
                fig_hist.add_trace(go.Histogram(
                    x=period_b[idx_col].dropna(), name=f"Period B  {win_b[0]}–{win_b[1]}",
                    marker_color="#ff7f0e", opacity=0.65,
                    hovertemplate="Value: %{x:.3f}<br>Count: %{y}<extra></extra>",
                ))
                fig_hist.update_layout(
                    barmode="overlay",
                    height=260, plot_bgcolor="white",
                    xaxis_title=f"{cmp_index} Value",
                    yaxis_title="Count",
                    margin=dict(l=50, r=20, t=20, b=40),
                    legend=dict(orientation="h", y=1.12),
                )
                st.plotly_chart(fig_hist, use_container_width=True)

    # ═════════════════════════════════════════════════════════════════════════
    #  MODE 3 — Year-by-year synthetic fallback (original behaviour)
    # ═════════════════════════════════════════════════════════════════════════
    else:
        st.caption(
            "Synthetic temperature maps. Upload a NetCDF file to compare real data. "
            "Both maps share an identical colour scale."
        )

        left_col, right_col = st.columns(2, gap="large")

        with left_col:
            baseline_year = st.slider(
                "Select Baseline Year", 1900, 2023, 1950, key="cmp_baseline"
            )
        with right_col:
            comparison_year = st.slider(
                "Select Comparison Year", 1900, 2023, 2023, key="cmp_compare"
            )

        lats_b, lons_b, temp_b = generate_temperature_map(baseline_year)
        lats_c, lons_c, temp_c = generate_temperature_map(comparison_year)

        SHARED_COLORSCALE = "RdBu_r"
        SHARED_ZMIN, SHARED_ZMID, SHARED_ZMAX = -2.5, 0.0, 2.5

        with left_col:
            st.markdown(
                f'<p class="section-label">Baseline · {baseline_year}</p>',
                unsafe_allow_html=True
            )
            fig_base = go.Figure(go.Heatmap(
                z=temp_b, x=lons_b, y=lats_b,
                colorscale=SHARED_COLORSCALE,
                zmin=SHARED_ZMIN, zmid=SHARED_ZMID, zmax=SHARED_ZMAX,
                colorbar=dict(title="°C", thickness=12, len=0.75,
                              tickfont=dict(size=9)),
                hovertemplate=(
                    "Lat: %{y:.1f}°  Lon: %{x:.1f}°<br>"
                    f"Anomaly ({baseline_year}): %{{z:.2f}} °C<extra></extra>"
                ),
            ))
            fig_base.update_layout(
                xaxis_title="Longitude", yaxis_title="Latitude",
                height=380, plot_bgcolor="white",
                margin=dict(l=50, r=20, t=30, b=50),
                title=dict(text=f"Temperature Anomaly — {baseline_year}",
                           font=dict(size=13), x=0.5, xanchor="center"),
            )
            st.plotly_chart(fig_base, use_container_width=True)
            mean_b = float(np.nanmean(temp_b))
            st.metric(f"Global Mean Anomaly ({baseline_year})", f"{mean_b:+.3f} °C")

        with right_col:
            st.markdown(
                f'<p class="section-label">Comparison · {comparison_year}</p>',
                unsafe_allow_html=True
            )
            fig_comp = go.Figure(go.Heatmap(
                z=temp_c, x=lons_c, y=lats_c,
                colorscale=SHARED_COLORSCALE,
                zmin=SHARED_ZMIN, zmid=SHARED_ZMID, zmax=SHARED_ZMAX,
                colorbar=dict(title="°C", thickness=12, len=0.75,
                              tickfont=dict(size=9)),
                hovertemplate=(
                    "Lat: %{y:.1f}°  Lon: %{x:.1f}°<br>"
                    f"Anomaly ({comparison_year}): %{{z:.2f}} °C<extra></extra>"
                ),
            ))
            fig_comp.update_layout(
                xaxis_title="Longitude", yaxis_title="Latitude",
                height=380, plot_bgcolor="white",
                margin=dict(l=50, r=20, t=30, b=50),
                title=dict(text=f"Temperature Anomaly — {comparison_year}",
                           font=dict(size=13), x=0.5, xanchor="center"),
            )
            st.plotly_chart(fig_comp, use_container_width=True)
            mean_c = float(np.nanmean(temp_c))
            st.metric(f"Global Mean Anomaly ({comparison_year})",
                      f"{mean_c:+.3f} °C",
                      delta=f"{mean_c - mean_b:+.3f} °C vs {baseline_year}")

        st.divider()
        st.markdown("#### Difference Map — Comparison minus Baseline")
        diff_grid  = temp_c - temp_b
        diff_bound = max(float(np.nanmax(np.abs(diff_grid))), 0.1)

        fig_diff = go.Figure(go.Heatmap(
            z=diff_grid, x=lons_b, y=lats_b,
            colorscale="RdBu_r",
            zmin=-diff_bound, zmid=0, zmax=diff_bound,
            colorbar=dict(title="Δ °C", thickness=14, tickfont=dict(size=9)),
            hovertemplate=(
                "Lat: %{y:.1f}°  Lon: %{x:.1f}°<br>"
                f"Δ ({comparison_year}−{baseline_year}): %{{z:.2f}} °C<extra></extra>"
            ),
        ))
        fig_diff.update_layout(
            xaxis_title="Longitude", yaxis_title="Latitude",
            height=340, plot_bgcolor="white",
            margin=dict(l=50, r=20, t=20, b=50),
        )
        st.plotly_chart(fig_diff, use_container_width=True)

        st.divider()
        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Mean Δ Anomaly",  f"{float(np.nanmean(diff_grid)):+.3f} °C")
        d2.metric("Max Warming Δ",   f"{float(np.nanmax(diff_grid)):+.2f} °C")
        d3.metric("Max Cooling Δ",   f"{float(np.nanmin(diff_grid)):+.2f} °C")
        d4.metric("Warming Grid Cells",
                  f"{int((diff_grid > 0).sum())}",
                  delta=f"{100 * np.nanmean(diff_grid > 0):.1f}% of globe")

# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 5 — DATA SOURCES
# ═══════════════════════════════════════════════════════════════════════════════

with tab_sources:
    st.markdown("### Dataset Information")
    st.info(
        "PyClimaExplorer supports standard **NetCDF (.nc)** files. "
        "Upload a file in the sidebar to switch from synthetic demo data to real observations.",
        icon="📂",
    )

    info = DATASET_INFO[dataset]
    cols = st.columns(3)
    for i, (k, v) in enumerate(info.items()):
        with cols[i % 3]:
            st.metric(k, v)

    st.divider()
    st.markdown("#### How to obtain data")

    guide_data = {
        "Dataset": list(DATASET_INFO.keys()),
        "Direct URL": [d["URL"] for d in DATASET_INFO.values()],
        "Python snippet": [
            "import cdsapi; c = cdsapi.Client(); c.retrieve('reanalysis-era5-single-levels', {...}, 'era5.nc')",
            "!wget https://esgf-node.llnl.gov/...  # use ESGF search API",
            "import xarray as xr; ds = xr.open_dataset('https://www.ncei.noaa.gov/.../ersst.v5.nc')",
        ],
    }

    st.dataframe(pd.DataFrame(guide_data), use_container_width=True, hide_index=True)

    st.divider()
    st.markdown("#### Data Processing Pipeline")

    st.image(
        "images/pipeline.png",
        caption="Climate Data Processing Pipeline",
        use_container_width=True
    )

    st.code("""
import xarray as xr

# 1. Load
ds = xr.open_dataset("era5_temperature.nc")

# 2. Filter region
ds_region = ds.sel(lat=slice(min_lat, max_lat),
                   lon=slice(min_lon, max_lon))

# 3. Filter time
ds_time = ds_region.sel(time=slice("1980", "2023"))

# 4. Extract variable
ta = ds_time["t2m"]

# 5. Compute anomaly
climatology = ta.sel(time=slice("1991", "2020")).mean("time")
anomaly = ta - climatology

# 6. Convert to Pandas for Plotly
df = anomaly.to_dataframe().reset_index()
""", language="python")

# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 6 — ABOUT
# ═══════════════════════════════════════════════════════════════════════════════

with tab_about:
    c1, c2 = st.columns([3, 2])

    with c1:
        st.markdown("""
## PyClimaExplorer

An **interactive climate data exploration dashboard** built for  
**TECHNEX '26 — Hack It Out Hackathon** at IIT (BHU) Varanasi.

The tool converts complex multi-dimensional NetCDF climate datasets into
compelling, interactive visual narratives accessible to both researchers
and the general public.

### Tech Stack
| Layer | Library |
|---|---|
| Web framework | Streamlit |
| Climate data I/O | Xarray |
| Data manipulation | Pandas · NumPy |
| Visualisation | Plotly |
| Statistics | SciPy |

### Dataset
- **ERSST v5** — NOAA Extended Reconstructed Sea Surface Temperature  
  Coverage: 1900–2018 · Resolution: 2° × 2°
        """)

    with c2:
        st.markdown("### Implemented Features")
        features = [
            "📈 Multi-index time series with regression overlays",
            "🗺️ Spatial pattern maps (SST trend, AMO, PDO)",
            "📊 Decadal average bar chart",
            "📉 Linear trend analysis with significance testing",
            "🔗 Pearson correlation matrix",
            "⚖️ Z-score normalised index comparison",
            "🔄 Rolling mean with configurable window",
            "🌐 3D interactive globe (real spatial fields)",
            "⏮️ Comparison mode — spatial & time-window",
            "📖 Story Mode — guided historical climate events",
            "📂 Real NetCDF upload via Xarray",
            "🌍 Region filter for spatial maps",
        ]
        for f in features:
            st.markdown(f"- {f}")
