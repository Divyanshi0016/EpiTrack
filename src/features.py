"""
features.py
-----------
Engineers predictive features on top of the cleaned DataFrame:
  - Lag features (7, 14, 21 days)
  - Rolling statistics (mean, std, min, max)
  - Growth rate & acceleration
  - Estimated reproduction number R0
  - Case fatality rate (CFR)
  - Mobility composite (if columns present)
  - Day-of-week, month, days-since-outbreak
"""

import numpy as np
import pandas as pd


def add_lag_features(df: pd.DataFrame,
                     col: str = 'new_cases_7d',
                     lags: list = None) -> pd.DataFrame:
    """Add lag columns for a given series within each country group."""
    if lags is None:
        lags = [7, 14, 21]
    for lag in lags:
        df[f'{col}_lag{lag}'] = (
            df.groupby('country')[col]
              .shift(lag)
        )
    return df


def add_rolling_stats(df: pd.DataFrame,
                      col: str = 'new_cases_7d',
                      windows: list = None) -> pd.DataFrame:
    """Add rolling mean, std, max for multiple window sizes."""
    if windows is None:
        windows = [7, 14, 30]
    for w in windows:
        grp = df.groupby('country')[col]
        df[f'{col}_rmean{w}'] = grp.transform(
            lambda x: x.rolling(w, min_periods=1).mean())
        df[f'{col}_rstd{w}']  = grp.transform(
            lambda x: x.rolling(w, min_periods=1).std().fillna(0))
        df[f'{col}_rmax{w}']  = grp.transform(
            lambda x: x.rolling(w, min_periods=1).max())
    return df


def add_growth_rate(df: pd.DataFrame,
                    col: str = 'new_cases_7d') -> pd.DataFrame:
    """
    Growth rate = (today - 7 days ago) / (7 days ago + 1).
    Acceleration = growth_rate - growth_rate_lag7.
    """
    lag7 = df.groupby('country')[col].shift(7)
    df['growth_rate'] = (df[col] - lag7) / (lag7 + 1)
    df['growth_rate'] = df['growth_rate'].clip(-5, 50)   # cap outliers
    df['growth_accel'] = (
        df['growth_rate'] - df.groupby('country')['growth_rate'].shift(7)
    )
    return df


def estimate_r0(df: pd.DataFrame,
                col: str = 'new_cases_7d',
                serial_interval: int = 5) -> pd.DataFrame:
    """
    Simple R0 estimate using the ratio method:
        R0 ≈ (cases in week t) / (cases in week t - serial_interval)
    A value > 1 means the epidemic is growing.
    """
    lag = df.groupby('country')[col].shift(serial_interval)
    df['r0_estimate'] = (df[col] / (lag + 1)).clip(0, 10)
    return df


def add_cfr(df: pd.DataFrame) -> pd.DataFrame:
    """Case Fatality Rate = cumulative deaths / cumulative confirmed * 100."""
    if 'deaths_cumulative' in df.columns and 'confirmed_cumulative' in df.columns:
        df['cfr'] = (
            df['deaths_cumulative'] / (df['confirmed_cumulative'] + 1) * 100
        ).clip(0, 30)
    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract date-based features."""
    df['day_of_week']  = df['date'].dt.dayofweek          # 0=Mon
    df['month']        = df['date'].dt.month
    df['week_of_year'] = df['date'].dt.isocalendar().week.astype(int)

    # Days since first case in each country
    first_date = df.groupby('country')['date'].transform('min')
    df['days_since_start'] = (df['date'] - first_date).dt.days
    return df


def add_mobility_composite(df: pd.DataFrame) -> pd.DataFrame:
    """
    If mobility columns exist, compute a single composite index:
        mob_composite = mean of (retail, grocery, work, transit) channels.
    High negative value → more lockdown-like behaviour.
    """
    mob_cols = ['mob_retail', 'mob_grocery', 'mob_work', 'mob_transit']
    available = [c for c in mob_cols if c in df.columns]
    if available:
        df['mob_composite'] = df[available].mean(axis=1)
    return df


def add_vaccination_rate_lag(df: pd.DataFrame) -> pd.DataFrame:
    """Lag vaccination rate by 14 days (immunity takes time)."""
    col = 'total_vaccinations_per_hundred'
    if col in df.columns:
        df['vax_lag14'] = df.groupby('country')[col].shift(14).fillna(0)
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all feature engineering steps in order."""
    print("  Building features...")
    df = add_lag_features(df)
    df = add_rolling_stats(df)
    df = add_growth_rate(df)
    df = estimate_r0(df)
    df = add_cfr(df)
    df = add_time_features(df)
    df = add_mobility_composite(df)
    df = add_vaccination_rate_lag(df)

    # Drop rows where lag features are still NaN (first 21 days per country)
    df = df.dropna(subset=['new_cases_7d_lag21'])
    df = df.reset_index(drop=True)
    print(f"  Feature matrix shape: {df.shape}")
    return df


def get_feature_columns() -> list:
    """Return the canonical list of ML feature columns."""
    return [
        'new_cases_7d_lag7', 'new_cases_7d_lag14', 'new_cases_7d_lag21',
        'new_cases_7d_rmean7', 'new_cases_7d_rstd7',
        'new_cases_7d_rmean14', 'new_cases_7d_rstd14',
        'new_cases_7d_rmean30',
        'growth_rate', 'growth_accel',
        'r0_estimate',
        'cfr',
        'day_of_week', 'month', 'days_since_start',
        'population_density', 'median_age', 'hospital_beds_per_thousand',
        'stringency_index', 'new_tests_per_thousand', 'vax_lag14',
    ]


if __name__ == '__main__':
    import os, sys
    sys.path.insert(0, os.path.dirname(__file__))
    from data_loader import build_master
    from preprocessing import clean

    df = build_master()
    df = clean(df)
    df = build_features(df)
    out = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'features.csv')
    df.to_csv(out, index=False)
    print(f"Saved features → {out}")
