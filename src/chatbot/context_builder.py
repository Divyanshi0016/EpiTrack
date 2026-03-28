"""
context_builder.py
------------------
Builds a structured context string from risk_summary.csv
that is injected into every LLM prompt.
"""

import pandas as pd
import os
from datetime import datetime

_HERE        = os.path.dirname(os.path.abspath(__file__))
_ROOT        = os.path.abspath(os.path.join(_HERE, '..', '..'))
SUMMARY_PATH = os.path.join(_ROOT, 'data', 'processed', 'risk_summary.csv')


def load_summary() -> pd.DataFrame:
    if not os.path.exists(SUMMARY_PATH):
        raise FileNotFoundError(
            f"risk_summary.csv not found at {SUMMARY_PATH}.\n"
            "Run:  python -m src.chatbot.risk_summary"
        )
    return pd.read_csv(SUMMARY_PATH)


def build_context(df: pd.DataFrame = None, top_n: int = 10) -> str:
    if df is None:
        df = load_summary()

    high_risk = df[df['status'] == 'High Risk'].head(top_n)
    moderate  = df[df['status'] == 'Moderate Risk'].head(top_n)
    low_risk  = df[df['status'] == 'Low Risk'].head(top_n)
    rising    = df[df['trend']  == 'Rising'].head(6)
    declining = df[df['trend']  == 'Declining'].head(6)

    n_high = len(df[df['status'] == 'High Risk'])
    n_mod  = len(df[df['status'] == 'Moderate Risk'])
    n_low  = len(df[df['status'] == 'Low Risk'])

    def fmt(sub):
        if sub.empty:
            return "None"
        return "; ".join(
            f"{r['country']} (risk={r['risk_score']:.2f}, R0={r['r0_estimate']:.2f}, trend={r['trend']})"
            for _, r in sub.iterrows()
        )

    return f"""
=== EPIDEMIC SITUATION REPORT ===
Generated : {datetime.now().strftime('%Y-%m-%d')}
Monitored : {len(df)} countries
High Risk : {n_high} | Moderate: {n_mod} | Low Risk: {n_low}

HIGH RISK  : {fmt(high_risk)}
MODERATE   : {fmt(moderate)}
SAFE       : {fmt(low_risk)}
RISING     : {fmt(rising)}
DECLINING  : {fmt(declining)}
=== END ===
""".strip()


def get_country_detail(country: str, df: pd.DataFrame = None) -> str:
    if df is None:
        df = load_summary()

    match = df[df['country'].str.lower() == country.lower()]
    if match.empty:
        match = df[df['country'].str.lower().str.contains(country.lower(), na=False)]
    if match.empty:
        return f"No data found for '{country}'."

    r = match.iloc[0]
    return (
        f"Country     : {r['country']}\n"
        f"Status      : {r['status']}\n"
        f"Trend       : {r['trend']}\n"
        f"Risk Score  : {r['risk_score']:.2f} / 1.00\n"
        f"R0 Estimate : {r['r0_estimate']:.2f} "
        f"({'spreading' if r['r0_estimate'] > 1 else 'declining'})\n"
        f"New Cases   : {int(r['new_cases']):,} / day (7d avg)\n"
        f"Growth Rate : {r['growth_rate']*100:+.1f}% vs last week\n"
        f"Vaccination : {r['vax_rate']:.1f} per 100 people\n"
        f"Stringency  : {r['stringency']:.0f} / 100\n"
    )


def get_safe_countries(df: pd.DataFrame = None, top_n: int = 10) -> list:
    if df is None:
        df = load_summary()
    return df[df['status'] == 'Low Risk'].head(top_n)['country'].tolist()


def get_high_risk_countries(df: pd.DataFrame = None, top_n: int = 10) -> list:
    if df is None:
        df = load_summary()
    return df[df['status'] == 'High Risk'].head(top_n)['country'].tolist()
