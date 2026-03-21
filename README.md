# 🦠 EpiTrack — Epidemic Spread Prediction
### CodeCure AI Hackathon · Track C

Predicts epidemic spread and outbreak risk using COVID-19 time-series data,
machine learning models (LSTM, XGBoost), and a classical SIR/SEIR compartmental model.

---

## 📁 Project Structure

```
epidemic-spread-prediction/
├── data/
│   ├── raw/                    # Downloaded CSVs (gitignored)
│   └── processed/              # Cleaned feature CSVs + country splits
├── notebooks/
│   ├── 01_EDA.ipynb            # Exploratory data analysis
│   ├── 02_preprocessing.ipynb  # Cleaning + feature engineering
│   ├── 03_model_LSTM.ipynb     # LSTM training & evaluation
│   ├── 04_model_XGBoost.ipynb  # XGBoost training, SHAP analysis
│   ├── 05_model_SIR.ipynb      # SIR/SEIR compartmental model + R₀
│   └── 06_evaluation.ipynb     # Cross-model comparison
├── src/
│   ├── data_loader.py          # Dataset download & merge
│   ├── preprocessing.py        # Cleaning pipeline
│   ├── features.py             # Feature engineering (lag, R₀, growth)
│   ├── evaluation.py           # RMSE, MAE, MAPE, plots
│   └── models/
│       ├── lstm_model.py       # Keras LSTM forecaster
│       ├── xgboost_model.py    # XGBoost + SHAP explainability
│       └── sir_model.py        # SIR / SEIR ODE models
├── app/
│   ├── dashboard.py            # 🖥️  Streamlit interactive dashboard
│   ├── risk_map.py             # Plotly choropleth risk map
│   └── assets/style.css
├── models/saved/               # Serialized trained models
├── outputs/plots/              # Generated charts
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Download datasets & build features
```bash
cd notebooks
jupyter nbconvert --to notebook --execute 01_EDA.ipynb
jupyter nbconvert --to notebook --execute 02_preprocessing.ipynb
```

### 3. Train models
```bash
jupyter nbconvert --to notebook --execute 03_model_LSTM.ipynb
jupyter nbconvert --to notebook --execute 04_model_XGBoost.ipynb
jupyter nbconvert --to notebook --execute 05_model_SIR.ipynb
```

### 4. Launch the dashboard
```bash
streamlit run app/dashboard.py
```

---

## 📊 Datasets

| Dataset | Source | Type |
|---------|--------|------|
| Johns Hopkins COVID-19 | [GitHub](https://github.com/CSSEGISandData/COVID-19) | Primary |
| Our World in Data | [GitHub](https://github.com/owid/covid-19-data) | Secondary |
| Google Mobility Reports | [Google](https://www.google.com/covid19/mobility/) | Optional |

---

## 🤖 Models

| Model | Type | Use Case |
|-------|------|----------|
| **LSTM** | Deep Learning | Sequential time-series forecast |
| **XGBoost** | Gradient Boosting | Tabular feature-based prediction + SHAP |
| **SIR/SEIR** | Compartmental ODE | Epidemiologically interpretable, R₀ estimation |

---

## 📈 Features Engineered

- **Lag features** — cases 7, 14, 21 days ago
- **Rolling statistics** — 7d / 14d / 30d mean, std, max
- **Growth rate & acceleration** — week-over-week change
- **R₀ estimate** — reproduction number via ratio method
- **CFR** — case fatality rate
- **Vaccination rate** (lagged 14 days)
- **Stringency index**, mobility composite
- **Temporal** — day of week, month, days since outbreak

---

## 🎯 Evaluation Metrics

| Model | RMSE | MAE | MAPE | R² |
|-------|------|-----|------|----|
| XGBoost | — | — | — | — |
| LSTM | — | — | — | — |
| SIR | — | — | — | — |

*Run notebook 06 to populate this table.*

---

## 🏆 Key Deliverables (Hackathon Checklist)

- [x] GitHub repository with clean structure
- [x] Outbreak prediction model (3 approaches)
- [x] Interactive epidemic dashboard (Streamlit)
- [x] Risk map of disease spread (Plotly choropleth)
- [x] Feature importance via SHAP
- [x] R₀ estimation and SIR/SEIR modeling

---

*EpiTrack · CodeCure AI Hackathon — Track C · Epidemic Spread Prediction*
