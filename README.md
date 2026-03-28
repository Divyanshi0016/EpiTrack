# 🦠 EpiTrack — Epidemic Spread Prediction

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-Model-FF6600?style=for-the-badge)
![TensorFlow](https://img.shields.io/badge/TensorFlow-LSTM-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o--mini-412991?style=for-the-badge&logo=openai&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-00c896?style=for-the-badge)

**CodeCure Biohackathon · Track C · AI + Epidemiology**

*Predicting disease spread using machine learning, deep learning, classical epidemiological models, and an AI-powered chatbot assistant*

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
- [AI Assistant — EpiBot](#-ai-assistant--epibot)
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

All outputs are packaged into an interactive multi-page Streamlit application featuring a forecast dashboard, a global risk map, SHAP-based feature importance analysis, and **EpiBot** — an AI-powered epidemic assistant that answers natural language queries about outbreak risk, safe travel, and country-specific situations.

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
┌──────────────────────────────────────────────────────────────┐
│                        DATA SOURCES                          │
│  Johns Hopkins COVID-19  │  Our World in Data  │  Google Mobility  │
└───────────────┬──────────────────────────────────────────────┘
                │
                ▼
┌──────────────────────────────────────────────────────────────┐
│               PREPROCESSING  (src/preprocessing.py)          │
│   Daily new cases · 7-day rolling avg · Missing value        │
│   forward-fill · Leading zero removal                        │
└───────────────┬──────────────────────────────────────────────┘
                │
                ▼
┌──────────────────────────────────────────────────────────────┐
│            FEATURE ENGINEERING  (src/features.py)            │
│   Lag features · Rolling stats · Growth rate                 │
│   R₀ estimate · Mobility index · Calendar features           │
└───────────────┬──────────────────────────────────────────────┘
                │
        ┌───────┼───────┐
        ▼       ▼       ▼
   ┌─────────┐ ┌──────┐ ┌──────────┐
   │XGBoost  │ │ LSTM │ │SIR/SEIR  │
   │+SHAP    │ │      │ │scipy.opt │
   └────┬────┘ └──┬───┘ └────┬─────┘
        └─────────┼──────────┘
                  │
                  ▼
┌──────────────────────────────────────────────────────────────┐
│               EVALUATION  (src/evaluation.py)                │
│            RMSE  ·  MAE  ·  MAPE  ·  Forecast plots         │
└───────────────┬──────────────────────────────────────────────┘
                │
        ┌───────┴────────────────┐
        ▼                        ▼
┌──────────────────┐   ┌─────────────────────────────────────┐
│  RISK SUMMARY    │   │     STREAMLIT MULTI-PAGE APP         │
│  (src/chatbot/   │   │  ┌─────────────┐ ┌───────────────┐  │
│  risk_summary.py)│──▶│  │  Dashboard  │ │  AI Assistant │  │
│  risk_summary.csv│   │  │  app/       │ │  EpiBot       │  │
└──────────────────┘   │  │  dashboard  │ │  GPT-4o-mini  │  │
                        │  │  .py        │ │  + Fallback   │  │
                        │  └─────────────┘ └───────────────┘  │
                        └─────────────────────────────────────┘
```

---

## 📁 Folder Structure

```
epidemic-spread-prediction/
│
├── 📂 .streamlit/
│   └── secrets.toml              ← API key config (never committed)
│
├── 📂 app/
│   ├── Home.py                   ← Multi-page app entry point
│   ├── dashboard.py              ← Forecast dashboard (standalone)
│   ├── risk_map.py               ← Plotly choropleth risk map
│   ├── assets/
│   │   └── style.css             ← Custom Streamlit theme
│   └── pages/
│       └── AI_Assistant.py       ← EpiBot chatbot page
│
├── 📂 data/
│   ├── raw/                      ← Download datasets here
│   └── processed/
│       ├── features.csv          ← Auto-generated after notebook 02
│       ├── risk_summary.csv      ← Auto-generated by risk_summary.py
│       └── country_splits/       ← Per-country CSV files
│
├── 📂 notebooks/
│   ├── 01_EDA.ipynb              ← Exploratory data analysis
│   ├── 02_preprocessing.ipynb    ← Cleaning + feature engineering
│   ├── 03_model_LSTM.ipynb       ← LSTM training + evaluation
│   ├── 04_model_XGBoost.ipynb    ← XGBoost training + SHAP analysis
│   ├── 05_model_SIR.ipynb        ← SIR/SEIR fitting + R₀ computation
│   └── 06_evaluation.ipynb       ← Model comparison + leaderboard
│
├── 📂 src/
│   ├── __init__.py
│   ├── data_loader.py            ← Load & merge all 3 datasets
│   ├── preprocessing.py          ← Clean, smooth, forward-fill
│   ├── features.py               ← Full feature engineering pipeline
│   ├── evaluation.py             ← RMSE, MAE, MAPE + forecast plots
│   ├── chatbot/
│   │   ├── __init__.py
│   │   ├── risk_summary.py       ← Builds risk_summary.csv
│   │   ├── context_builder.py    ← Builds LLM context from CSV
│   │   └── engine.py             ← OpenAI + offline fallback engine
│   └── models/
│       ├── __init__.py
│       ├── lstm_model.py         ← Stacked LSTM (Keras/TensorFlow)
│       ├── xgboost_model.py      ← XGBoost regressor + SHAP
│       └── sir_model.py          ← SIR & SEIR ODE fitting
│
├── 📂 models/
│   └── saved/                    ← Trained model files (.h5, .pkl)
│
├── 📂 outputs/
│   ├── plots/                    ← Forecast plots, SHAP charts
│   └── reports/                  ← Risk map HTML, evaluation reports
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
Input       : Sliding window of 14–21 days of case counts (auto-adjusted)
Output      : Next-day case forecast
Strength    : Captures wave dynamics and long-range temporal patterns
Weakness    : Needs longer data history, slower to train
Fixes       : Auto seq_len reduction, validation_split guard, batch_size clip
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

All features are built in `src/features.py` and stored in `data/processed/features.csv`:

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

## 💬 AI Assistant — EpiBot

EpiBot is a smart epidemic assistant built into the dashboard that answers natural language questions using real data from the prediction pipeline.

### Features
- 📊 Summarises global outbreak situation
- 🔥 Identifies high-risk countries with R₀ and growth rate data
- 🟢 Suggests safe travel destinations
- 🌍 Answers country-specific queries
- 📄 Generates downloadable outbreak reports
- 💬 Maintains chat history across the session

### Two Modes
| Mode | When | How |
|---|---|---|
| **OpenAI GPT-4o-mini** | API key provided | Full AI responses using your data as context |
| **Offline Fallback** | No API key | Rule-based engine, covers all common queries, zero cost |

> The offline engine handles all hackathon demo scenarios without any API key.

### Example Queries
```
"Which countries are safe to travel to?"
"Show me all high risk countries"
"What is the situation in India?"
"Give me a global outbreak summary"
"Which countries are improving?"
"Explain what R0 means"
"Generate a full epidemic report"
```

### How It Works
```
User query
    ↓
context_builder.py  ←  risk_summary.csv  ←  features.csv
    ↓
engine.py (OpenAI GPT-4o-mini OR Fallback)
    ↓
Structured response in Streamlit chat UI
```

### Setup (Optional — for full AI mode)
Add your OpenAI API key to `.streamlit/secrets.toml`:
```toml
OPENAI_API_KEY = "sk-your-key-here"
```
Or paste it directly in the sidebar when the app is running.

---

## ⚙️ Setup & Installation

### Prerequisites
- Python 3.10 or higher
- pip
- Git

### Step 1 — Clone the repository
```bash
git clone https://github.com/Divyanshi0016/EpiTrack.git
cd EpiTrack
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

> ⚠️ **TensorFlow is 385MB.** If the download times out:
> ```bash
> pip install tensorflow --timeout=1000
> ```
> Or skip TensorFlow entirely (LSTM notebook optional):
> ```bash
> pip install pandas numpy matplotlib seaborn scipy scikit-learn xgboost shap plotly folium streamlit openai
> ```

### Step 5 — Download datasets
Place the 3 dataset files in `data/raw/` as described in the [Datasets](#-datasets) section above.

---

## 🚀 How to Run

### Option A — Full Multi-Page App *(Recommended)*

```bash
streamlit run app/Home.py
```

Opens at `http://localhost:8501` with two pages:
- **Dashboard** — forecast charts, R₀, SHAP, risk map
- **AI Assistant** — EpiBot chatbot

### Option B — Dashboard Only

```bash
streamlit run app/dashboard.py
```

### Option C — Run Notebooks First (for trained models)

```bash
# Register venv as Jupyter kernel
python -m ipykernel install --user --name=epitrack

# Open Jupyter
jupyter notebook notebooks/
```

Run in this order:
```
02_preprocessing.ipynb    ← Required — generates features.csv
04_model_XGBoost.ipynb    ← Required — best model + SHAP
05_model_SIR.ipynb        ← Required — R₀ estimates
03_model_LSTM.ipynb       ← Optional — needs TensorFlow
06_evaluation.ipynb       ← Optional — model comparison table
01_EDA.ipynb              ← Optional — data exploration
```

### Option D — Generate Risk Summary for EpiBot

Run once after notebook 02 to enable the AI Assistant:
```bash
python -m src.chatbot.risk_summary
```

---

## 🖥️ Dashboard

### Page 1 — Forecast Dashboard
| Panel | Description |
|---|---|
| **Sidebar** | Country selector, model selector, forecast horizon slider (7–60 days), Run Prediction button |
| **4 KPI Cards** | Daily new cases, estimated R₀, cumulative confirmed, vaccination rate |
| **Forecast Chart** | Actual cases + XGBoost predictions + SIR 30-day projection overlaid |
| **R₀ Timeline** | Reproduction number over time — dashed red line at R₀ = 1 threshold |
| **Growth Rate Bars** | Red = outbreak expanding, green = declining — week-over-week momentum |
| **SHAP Importance** | Which features drove the XGBoost prediction most strongly |
| **Global Risk Map** | Choropleth world map — green (declining) to red (surging) by 14-day growth rate |

### Page 2 — AI Assistant (EpiBot)
| Panel | Description |
|---|---|
| **Sidebar** | API key input, live risk snapshot, Refresh Data button, Report Generator |
| **Quick Questions** | 6 one-click suggested queries for instant demo |
| **Chat Interface** | Full conversation history with user and bot messages |
| **Report Generator** | Generates downloadable markdown outbreak report |

---

## 📊 Model Performance

*Fill in after running notebook 06:*

| Model | RMSE | MAE | MAPE | Notes |
|---|---|---|---|---|
| XGBoost | — | — | — | Best overall accuracy |
| LSTM | — | — | — | Best wave detection |
| SIR | — | — | — | Best interpretability |

**Evaluation:** Held-out test set = last 30 days per country. MAPE is scale-independent and fairest for cross-country comparison.

---

## ✅ Deliverables

| # | Deliverable | Status | Location |
|---|---|---|---|
| 1 | GitHub Repository | ✅ | This repo |
| 2 | Outbreak Prediction Model | ✅ | `src/models/` — XGBoost, LSTM, SIR/SEIR |
| 3 | Interactive Epidemic Dashboard | ✅ | `app/dashboard.py` + `app/Home.py` |
| 4 | Global Risk Map | ✅ | `app/risk_map.py` + `outputs/reports/risk_map.html` |
| 5 | Feature Importance (SHAP) | ✅ | `src/models/xgboost_model.py` + notebook 04 |
| 6 | AI Epidemic Assistant | ✅ | `src/chatbot/` + `app/pages/AI_Assistant.py` |
| 7 | Automated Report Generator | ✅ | `src/chatbot/engine.py` → `generate_report()` |

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
| Dashboard | Streamlit (multi-page) |
| AI Assistant | OpenAI GPT-4o-mini + offline fallback |
| Notebooks | Jupyter |

---

## 🐛 Known Issues & Fixes

| Issue | Fix Applied |
|---|---|
| `Taiwan*` causes Windows filename error | Strip illegal chars in `preprocessing.py` |
| LSTM `ValueError: 0 samples` | Auto seq_len reduction + validation_split guard |
| `United States` not found in dataset | Country name validation — use `'US'` |
| `ModuleNotFoundError: src.chatbot` | `sys.path` fix in `AI_Assistant.py` |
| `st.switch_page` error in Home.py | Removed — sidebar navigation used instead |
| Merge conflict on first GitHub push | Resolved with `git pull --allow-unrelated-histories` |

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
