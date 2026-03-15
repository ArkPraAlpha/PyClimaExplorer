# 🌍 PyClimaExplorer

### Interactive Climate Data Visualization & Analysis Dashboard

**HackItOut Hackathon – TECHNEX '26 | IIT (BHU) Varanasi**

PyClimaExplorer is an interactive web-based dashboard designed to explore, analyze, and visualize large-scale climate datasets. The platform allows users to interact with climate indices, spatial temperature patterns, and long-term climate trends using an intuitive interface.

The application supports NetCDF climate datasets and provides powerful visualization and statistical tools to help researchers, students, and enthusiasts understand complex climate patterns.

---

# 🚀 Features

### 📊 Interactive Climate Dashboard

* Explore major climate indices such as:

  * AMO (Atlantic Multidecadal Oscillation)
  * ENSO (El Niño–Southern Oscillation)
  * PDO (Pacific Decadal Oscillation)
  * NAO (North Atlantic Oscillation)
  * SOI (Southern Oscillation Index)

### 📈 Time-Series Climate Analysis

* Interactive plots for climate indices
* Regression trend analysis
* Decadal climate oscillation averages
* Rolling trend analysis

### 🌎 Spatial Climate Visualization

* Global Sea Surface Temperature (SST) patterns
* Climate anomaly heatmaps
* Region-specific climate exploration

### 📖 Climate Story Mode

Explore historical climate events with contextual analysis:

* 1930s Dust Bowl
* 1972–73 El Niño
* 1998 Super El Niño
* 2010 Russia–Pakistan climate extremes

### 🔬 Statistical Climate Analysis

* Climate anomaly calculation
* Decadal averaging
* Correlation matrix across climate indices
* Rolling mean trend analysis

### 🤖 Machine Learning Forecasting

* Polynomial Ridge Regression model
* Forecast future climate index values
* Visual comparison between historical trends and predicted values

---

# 🧠 Tech Stack

| Category            | Technologies   |
| ------------------- | -------------- |
| Language            | Python         |
| Web Framework       | Streamlit      |
| Data Processing     | Pandas, NumPy  |
| Climate Data        | NetCDF, Xarray |
| Visualization       | Plotly         |
| Machine Learning    | Scikit-learn   |
| Scientific Analysis | SciPy          |

---

# 📂 Project Structure

```
PyClimaExplorer
│
├── app_new.py            # Main Streamlit application
│
├── modules
│   ├── data_loader.py        # Load NetCDF climate datasets
│   ├── data_processing.py    # Statistical analysis functions
│   ├── visualization.py      # Plotly visualization utilities
│   └── ml_model.py           # Climate forecasting model
│
├── dataset
│   ├── ERSST_v5_1.cvdp_data.1979-2018       # Example NetCDF dataset
│   └── ERSST_v5.cvdp_data.1900-2018
├── screenshots
│   └── dashboard.png
│
├── requirements.txt
└── README.md
```

---

# 📦 Installation

Clone the repository:

```bash
git clone https://github.com/ArkPraAlpha/PyClimaExplorer.git
cd PyClimaExplorer
```

Create virtual environment (recommended):

```bash
python -m venv .venv
```

Activate environment:

**Windows**

```bash
.venv\Scripts\activate
```

**Linux / Mac**

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# ▶️ Run the Application

Start the Streamlit dashboard:

```bash
streamlit run app_new.py
```

Then open the browser at:

```
http://localhost:8501
```

---

# 📊 Supported Climate Data

The dashboard supports **NetCDF climate datasets** including:

* ERA5 Reanalysis
* CMIP6 Climate Models
* ERSST Sea Surface Temperature

Users can upload `.nc` files directly through the interface for analysis.

---

# 🧪 Example Analysis Capabilities

The platform enables:

* Climate index time-series visualization
* Spatial SST anomaly mapping
* Climate oscillation correlation analysis
* Decadal climate pattern exploration
* Machine learning-based climate trend forecasting

---

# 🎯 Hackathon Objective

The goal of this project was to develop **PyClimaExplorer**, a rapid-prototype climate data visualization platform that allows interactive exploration of climate model outputs.

The tool helps bridge the gap between complex climate datasets and accessible scientific visualization.

---

# 👨‍💻 Team

HackItOut Hackathon Team

ARK PRAJAPATI
ASTHA GUPTA
SHASHANK KUMAR SHUKLA
KASHAF NOOR

---

# ⭐ Acknowledgements

* NOAA ERSST Dataset
* ECMWF ERA5 Reanalysis
* WCRP CMIP6 Climate Models
* TECHNEX '26 – IIT (BHU) Varanasi Hackathon
