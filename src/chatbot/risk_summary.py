"""
risk_summary.py
---------------
Generates data/processed/risk_summary.csv from the processed features CSV.
Run once after notebook 02:
    python -m src.chatbot.risk_summary
"""

import pandas as pd
import numpy as np
import os

# Paths relative to project root
_HERE         = os.path.dirname(os.path.abspath(__file__))
_ROOT         = os.path.abspath(os.path.join(_HERE, '..', '..'))
PROCESSED_DIR = os.path.join(_ROOT, 'data', 'processed')
OUTPUT_PATH   = os.path.join(PROCESSED_DIR, 'risk_summary.csv')


def classify_risk(risk_score: float) -> str:
    if risk_score > 0.7:
        return "High Risk"
    elif risk_score > 0.4:
        return "Moderate Risk"
    else:
        return "Low Risk"


def classify_trend(growth_rate: float) -> str:
    if growth_rate > 0.15:
        return "Rising"
    elif growth_rate < -0.10:
        return "Declining"
    else:
        return "Stable"

def compute_risk_score(row: pd.Series) -> float:
    score = 0.0

    gr = float(row.get('growth_rate_7d', 0) or 0)
    score += np.clip(gr, -0.5, 1.0) * 0.35

    r0 = float(row.get('r0_estimate', 1) or 1)
    score += np.clip((r0 - 0.5) / 2.5, 0, 1) * 0.35

    vax = float(row.get('total_vaccinations_per_hundred', 0) or 0)
    score += (1 - np.clip(vax / 200, 0, 1)) * 0.20

    st = float(row.get('stringency_index', 50) or 50)
    score += (1 - np.clip(st / 100, 0, 1)) * 0.10

    return round(float(np.clip(score, 0, 1)), 4)



def build_risk_summary(features_path: str = None) -> pd.DataFrame:
    """
    Read features.csv, compute latest risk per country,
    save risk_summary.csv and return the DataFrame.
    """
    if features_path is None:
        features_path = os.path.join(PROCESSED_DIR, 'features.csv')
        alt_path = os.path.join(PROCESSED_DIR, 'merged_features.csv')
        if not os.path.exists(features_path) and os.path.exists(alt_path):
            features_path = alt_path

    if not os.path.exists(features_path):
        raise FileNotFoundError(
            f"Processed features file not found at:\n  {features_path}\n"
            "Please run notebook 02 first."
        )

    print(f"  Loading features: {features_path}")
    df = pd.read_csv(features_path, parse_dates=['date'])

    # Get the most recent row per country
    latest = (df.sort_values('date')
                .groupby('country')
                .tail(1)
                .reset_index(drop=True))

    rows = []
    for _, row in latest.iterrows():
        risk_score = compute_risk_score(row)

        # Safe population handling
        pop = row.get('population', 0)
        if pd.isna(pop) or pop is None:
            pop = 0
        else:
            pop = int(float(pop))   # safe conversion

        rows.append({
            'country':      str(row['country']),
            'date':         str(row['date'].date()) if pd.api.types.is_datetime64_any_dtype(row.get('date')) else str(row['date']),
            'cases_14d':    int(row.get('roll_mean_14d', 0) or 0),
            'new_cases':    int(row.get('new_cases_jhu_smooth', 0) or 0),
            'growth_rate':  round(float(row.get('growth_rate_7d', 0) or 0), 4),
            'r0_estimate':  round(float(row.get('r0_estimate', 1.0) or 1.0), 3),
            'vax_rate':     round(float(row.get('total_vaccinations_per_hundred', 0) or 0), 1),
            'stringency':   round(float(row.get('stringency_index', 0) or 0), 1),
            'population':   pop,
            'risk_score':   risk_score,
            'trend':        classify_trend(float(row.get('growth_rate_7d', 0) or 0)),
            'status':       classify_risk(risk_score),
        })

    summary = pd.DataFrame(rows).sort_values('risk_score', ascending=False).reset_index(drop=True)

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    summary.to_csv(OUTPUT_PATH, index=False)
    print(f"  Saved: {OUTPUT_PATH}  ({len(summary)} countries)")
    return summary


if __name__ == '__main__':
    df = build_risk_summary()
    print(df[['country', 'risk_score', 'trend', 'status']].head(20).to_string())
