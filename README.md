# 🦠 EpiTrack — Epidemic Spread Prediction

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-Model-FF6600?style=for-the-badge)
![TensorFlow](https://img.shields.io/badge/TensorFlow-LSTM-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)


**CodeCure Biohackathon · Track C · AI + Epidemiology**

*Predicting disease spread using machine learning, deep learning, and classical epidemiological models*

</div>

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Problem Statement](#-problem-statement)
- [Architecture](#-architecture)
- [Folder Structure](#-folder-structure)
- [Datasets](#-datasets)
- [Models](#-models)
- [Feature Engineering](#-feature-engineering)
- [Setup & Installation](#-setup--installation)
- [How to Run](#-how-to-run)
- [Dashboard](#-dashboard)
- [Model Performance](#-model-performance)
- [Deliverables](#-deliverables)
- [Tech Stack](#-tech-stack)
- [Team](#-team)

---

## 🌍 Overview

**EpiTrack** is a full-stack epidemic spread prediction system built for the CodeCure Biohackathon (Track C). It combines three fundamentally different modelling approaches — gradient boosting, deep learning, and classical compartmental epidemiology — to forecast disease spread, estimate outbreak risk, and identify the key drivers of transmission.

All outputs are packaged into a single interactive Streamlit dashboard with a global risk map, real-time R₀ tracking, and SHAP-based feature importance analysis.

---

## 🎯 Problem Statement

Infectious diseases like COVID-19 can spread exponentially before health systems have time to respond. The core challenge is:

> **Can we predict where an outbreak is heading before it peaks?**

Early forecasting gives health authorities the lead time to:
- Allocate medical resources (beds, ventilators, staff) to high-risk regions
- Implement or lift restrictions at the right time
- Identify outbreak hotspots before they become crisis zones

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     DATA SOURCES                        │
│  Johns Hopkins COVID-19  │  Our World in Data  │  Google Mobility  │
└──────────────┬──────────────────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────────────┐
│              PREPROCESSING  (src/preprocessing.py)      │
│  Daily new cases · 7-day rolling avg · Missing value    │
│  forward-fill · Leading zero removal                    │
└──────────────┬──────────────────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────────────┐
│           FEATURE ENGINEERING  (src/features.py)        │
│  Lag features · Rolling stats · Growth rate             │
│  R₀ estimate · Mobility index · Calendar features       │
└──────────────┬──────────────────────────────────────────┘
               │
       ┌───────┼───────┐
       ▼       ▼       ▼
  ┌─────────┐ ┌──────┐ ┌──────────┐
  │XGBoost  │ │ LSTM │ │SIR/SEIR  │
  │+SHAP    │ │ GRU  │ │scipy.opt │
  └────┬────┘ └──┬───┘ └────┬─────┘
       └─────────┼──────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────┐
│            EVALUATION  (src/evaluation.py)              │
│         RMSE  ·  MAE  ·  MAPE  ·  Forecast plots       │
└──────────────┬──────────────────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────────────┐
│         STREAMLIT DASHBOARD  (app/dashboard.py)         │
│  Forecast chart · R₀ timeline · Growth rate bars        │
│  SHAP importance · Global choropleth risk map           │
└─────────────────────────────────────────────────────────┘
```

---

## 📁 Folder Structure

```
epidemic-spread-prediction/
│
├── 📂 data/
│   ├── raw/                          ← Download datasets here (see below)
│   └── processed/
│       ├── merged_features.csv       ← Auto-generated after running notebook 02
│       └── country_splits/           ← Per-country CSV files
│
├── 📂 notebooks/
│   ├── 01_EDA.ipynb                  ← Exploratory data analysis
│   ├── 02_preprocessing.ipynb        ← Cleaning + feature engineering
│   ├── 03_model_LSTM.ipynb           ← LSTM training + evaluation
│   ├── 04_model_XGBoost.ipynb        ← XGBoost training + SHAP analysis
│   ├── 05_model_SIR.ipynb            ← SIR/SEIR fitting + R₀ computation
│   └── 06_evaluation.ipynb           ← Model comparison + leaderboard
│
├── 📂 src/
│   ├── __init__.py
│   ├── data_loader.py                ← Load & merge all 3 datasets
│   ├── preprocessing.py              ← Clean, smooth, forward-fill
│   ├── features.py                   ← Full feature engineering pipeline
│   ├── evaluation.py                 ← RMSE, MAE, MAPE + forecast plots
│   └── models/
│       ├── __init__.py
│       ├── lstm_model.py             ← Stacked LSTM (Keras/TensorFlow)
│       ├── xgboost_model.py          ← XGBoost regressor + SHAP
│       └── sir_model.py              ← SIR & SEIR ODE fitting
│
├── 📂 app/
│   ├── dashboard.py                  ← Main Streamlit application
│   ├── risk_map.py                   ← Plotly choropleth risk map
│   └── assets/
│       └── style.css                 ← Custom Streamlit theme
│
├── 📂 models/
│   └── saved/                        ← Trained model files (.h5, .pkl)
│
├── 📂 outputs/
│   ├── plots/                        ← Forecast plots, SHAP charts
│   └── reports/                      ← Risk map HTML, evaluation reports
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 📦 Datasets

Download the following files and place them in `data/raw/`:

### 1. Johns Hopkins COVID-19 Time Series *(Primary — Required)*
| | |
|---|---|
| **File name** | `time_series_covid19_confirmed_global.csv` |
| **Download** | [GitHub — CSSEGISandData/COVID-19](https://github.com/CSSEGISandData/COVID-19/blob/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv) |
| **Contains** | Daily cumulative confirmed cases for 200+ countries from Jan 2020 |
| **Used for** | Core training signal — actual case counts the models predict |

### 2. Our World in Data COVID-19 Dataset *(Secondary — Required)*
| | |
|---|---|
| **File name** | `owid-covid-data.csv` |
| **Download** | [GitHub — owid/covid-19-data](https://github.com/owid/covid-19-data/blob/master/public/data/owid-covid-data.csv) |
| **Contains** | Vaccination rates, stringency index, population density, median age, GDP |
| **Used for** | Contextual features that explain why cases rise or fall |

### 3. Google Community Mobility Reports *(Optional)*
| | |
|---|---|
| **File name** | `Global_Mobility_Report.csv` |
| **Download** | [google.com/covid19/mobility](https://www.google.com/covid19/mobility/) |
| **Contains** | Daily % change in movement at workplaces, transit, retail vs baseline |
| **Used for** | Mobility index feature — leading indicator of transmission changes |

> **Note:** The dashboard and all models work without the Google Mobility file. It will be skipped automatically if not present.

---

## 🤖 Models

### Model 1 — XGBoost ⭐ *Best overall*
```
Type        : Gradient Boosting (tabular ML)
File        : src/models/xgboost_model.py
Input       : 17 engineered features (lag, rolling, mobility, vaccination...)
Output      : Predicted daily case count for next N days
Strength    : Highest accuracy + SHAP interpretability
Weakness    : Degrades beyond ~14-day forecasting horizon
```

### Model 2 — LSTM
```
Type        : Long Short-Term Memory Neural Network (deep learning)
File        : src/models/lstm_model.py
Input       : Sliding window of 30 days of case counts
Output      : 7-day ahead forecast
Strength    : Captures wave dynamics and long-range temporal patterns
Weakness    : Needs longer data history, slower to train
```

### Model 3 — SIR / SEIR
```
Type        : Compartmental epidemiological model (differential equations)
File        : src/models/sir_model.py
Input       : Cumulative case counts + population size
Output      : Fitted curve + R₀ estimate + long-range projection
Strength    : Biologically interpretable, works with sparse data
Weakness    : Assumes homogeneous population, lower raw accuracy
Key output  : R₀ = β/γ  →  R₀ > 1 = growing,  R₀ < 1 = declining
```

---

## 🔬 Feature Engineering

All features are built in `src/features.py` and stored in `data/processed/merged_features.csv`:

| Feature | Type | Description |
|---|---|---|
| `lag_7d`, `lag_14d`, `lag_21d` | Lag | Cases from 1, 2, 3 weeks ago |
| `roll_mean_7d`, `roll_mean_14d` | Rolling | Short and medium-term trend |
| `roll_std_7d`, `roll_std_14d` | Rolling | Signal volatility / noise |
| `growth_rate_7d` | Derived | % change vs 7 days ago |
| `r0_estimate` | Epidemiological | Reproduction number estimate |
| `mobility_index` | External | Composite movement score |
| `stringency_index` | External | Government response score |
| `total_vaccinations_per_hundred` | External | Vaccine coverage % |
| `population_density` | Static | People per km² |
| `median_age` | Static | Country median age |
| `day_of_week`, `month`, `week_of_year` | Calendar | Seasonality + reporting patterns |

---

## ⚙️ Setup & Installation

### Prerequisites
- Python 3.10 or higher
- pip
- Git

### Step 1 — Clone the repository
```bash
git clone https://github.com/your-team/epidemic-spread-prediction.git
cd epidemic-spread-prediction
```

### Step 2 — Create virtual environment
```bash
python -m venv venv
```

### Step 3 — Activate virtual environment
```bash
# Windows
venv\Scripts\activate

# Mac / Linux
source venv/bin/activate
```

### Step 4 — Install dependencies
```bash
pip install -r requirements.txt
```

> ⚠️ TensorFlow is 385MB. If download times out, run:
> ```bash
> pip install tensorflow --timeout=1000
> ```
> Or install without TensorFlow (LSTM notebook will be skipped):
> ```bash
> pip install pandas numpy matplotlib seaborn scipy scikit-learn xgboost shap plotly folium streamlit
> ```

### Step 5 — Download datasets
Place the 3 dataset files in `data/raw/` as described in the [Datasets](#-datasets) section above.

---

## 🚀 How to Run

### Option A — Run Everything (Recommended)

Run notebooks in order, then launch the dashboard:

```bash
# Register the venv as a Jupyter kernel first
python -m ipykernel install --user --name=epitrack

# Open Jupyter
jupyter notebook notebooks/
```

Run in this exact order:
```
01_EDA.ipynb              ← Understand the data
02_preprocessing.ipynb    ← Clean + engineer features  (saves to data/processed/)
03_model_LSTM.ipynb       ← Train LSTM                 (saves to models/saved/)
04_model_XGBoost.ipynb    ← Train XGBoost + SHAP       (saves to models/saved/)
05_model_SIR.ipynb        ← Fit SIR/SEIR + compute R₀
06_evaluation.ipynb       ← Compare all models
```

Then launch the dashboard:
```bash
streamlit run app/dashboard.py
```

### Option B — Dashboard Only (Quick Start)

If you just want to see the dashboard with live model training:

```bash
streamlit run app/dashboard.py
```

The dashboard will preprocess data and train models on-demand when you click **Run Prediction**.

### Option C — Generate Risk Map Only

```bash
python -c "
from src.data_loader import load_jhu, load_owid, merge_all
from src.preprocessing import run_full_pipeline
from src.features import build_all_features
from app.risk_map import compute_risk_scores, build_choropleth

jhu = load_jhu()
owid = load_owid()
df = build_all_features(run_full_pipeline(merge_all(jhu, owid)))
risk = compute_risk_scores(df)
build_choropleth(risk, save_html=True)
print('Risk map saved to outputs/reports/risk_map.html')
"
```

---

## 🖥️ Dashboard

Open `http://localhost:8501` after running `streamlit run app/dashboard.py`.

### What It Shows

| Panel | Description |
|---|---|
| **Sidebar** | Country selector, model selector, forecast horizon slider (7–60 days), Run Prediction button |
| **4 KPI Cards** | Daily new cases, estimated R₀, cumulative confirmed, vaccination rate |
| **Forecast Chart** | Actual cases + XGBoost predictions + SIR 30-day projection overlaid |
| **R₀ Timeline** | Reproduction number over time — dashed red line at R₀ = 1 threshold |
| **Growth Rate Bars** | Red = outbreak expanding, green = declining — week-over-week momentum |
| **SHAP Importance** | Which features drove the XGBoost prediction most strongly |
| **Global Risk Map** | Choropleth world map — green (declining) to red (surging) by 14-day growth rate |

---

## 📊 Model Performance

*Fill in after running notebook 06 on your chosen test country:*

| Model | RMSE | MAE | MAPE | Notes |
|---|---|---|---|---|
| XGBoost | — | — | — | Best overall accuracy |
| LSTM | — | — | — | Best wave detection |
| SIR | — | — | — | Best interpretability |

**Evaluation method:** Held-out test set = last 30 days per country. MAPE is the fairest metric for cross-country comparison since it is scale-independent.

---

## ✅ Deliverables

| # | Deliverable | Status | Location |
|---|---|---|---|
| 1 | GitHub Repository | ✅ | This repo |
| 2 | Outbreak Prediction Model | ✅ | `src/models/` |
| 3 | Interactive Epidemic Dashboard | ✅ | `app/dashboard.py` |
| 4 | Global Risk Map | ✅ | `app/risk_map.py` + `outputs/reports/risk_map.html` |
| 5 | Feature Importance Analysis (SHAP) | ✅ | `src/models/xgboost_model.py` + notebook 04 |

---

## 🛠️ Tech Stack

| Category | Technology |
|---|---|
| Language | Python 3.10+ |
| Data processing | pandas, numpy |
| Machine learning | scikit-learn, XGBoost |
| Deep learning | TensorFlow / Keras |
| Epidemiological modelling | scipy (odeint, optimize) |
| Interpretability | SHAP |
| Visualisation | Plotly, Matplotlib, Seaborn, Folium |
| Dashboard | Streamlit |
| Notebooks | Jupyter |

---

## 📜 Data Sources & Citations

- **Johns Hopkins CSSE** — Dong E, Du H, Gardner L. *An interactive web-based dashboard to track COVID-19 in real time.* Lancet Inf Dis. 2020.
- **Our World in Data** — Hannah Ritchie et al. *Coronavirus Pandemic (COVID-19).* OurWorldInData.org. 2020.
- **Google LLC** — *Google COVID-19 Community Mobility Reports.* google.com/covid19/mobility. 2020.

---

## 👥 Team

**CodeCure — Track C Submission**
> CodeCure Biohackathon · Epidemic Spread Prediction · AI + Epidemiology

---



<div align="center">
  <sub>Built for CodeCure Biohackathon · Track C · Epidemic Spread Prediction</sub>
</div>
