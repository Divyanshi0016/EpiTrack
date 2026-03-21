"""
preprocessing.py
----------------
Cleans the master DataFrame and engineers base numeric features.
"""

import os
import numpy as np
import pandas as pd


PROCESSED_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')


def remove_leading_zeros(df: pd.DataFrame) -> pd.DataFrame:
    """For each country, drop rows before the first non-zero confirmed case."""
    def _trim(g):
        first = g['confirmed_cumulative'].ne(0).idxmax()
        return g.loc[first:]
    return df.groupby('country', group_keys=False).apply(_trim)


def compute_daily_new(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert cumulative confirmed/deaths into daily new counts.
    Clips negatives (data corrections) to 0.
    """
    df = df.sort_values(['country', 'date'])
    df['new_cases'] = (
        df.groupby('country')['confirmed_cumulative']
          .diff()
          .clip(lower=0)
    )
    df['new_deaths'] = (
        df.groupby('country')['deaths_cumulative']
          .diff()
          .clip(lower=0)
    )
    return df


def smooth_rolling(df: pd.DataFrame,
                   cols: list = None,
                   window: int = 7) -> pd.DataFrame:
    """Apply a rolling mean to reduce day-of-week reporting noise."""
    if cols is None:
        cols = ['new_cases', 'new_deaths']
    for col in cols:
        if col in df.columns:
            df[f'{col}_7d'] = (
                df.groupby('country')[col]
                  .transform(lambda x: x.rolling(window, min_periods=1).mean())
            )
    return df


def fill_owid_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    OWID columns update weekly; forward-fill within each country
    then backfill remaining NaNs.
    """
    owid_cols = [
        'total_vaccinations_per_hundred',
        'people_fully_vaccinated_per_hundred',
        'stringency_index',
        'new_tests_per_thousand',
    ]
    for col in owid_cols:
        if col in df.columns:
            df[col] = (
                df.groupby('country')[col]
                  .transform(lambda x: x.ffill().bfill())
            )
    return df


def fill_static_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Static OWID features (population, density, etc.) are constant per country.
    Forward-fill then backward-fill within each country group.
    """
    static_cols = [
        'population', 'population_density', 'median_age',
        'hospital_beds_per_thousand', 'life_expectancy',
    ]
    for col in static_cols:
        if col in df.columns:
            df[col] = (
                df.groupby('country')[col]
                  .transform(lambda x: x.ffill().bfill())
            )
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Master cleaning pipeline."""
    df = remove_leading_zeros(df)
    df = compute_daily_new(df)
    df = smooth_rolling(df)
    df = fill_owid_features(df)
    df = fill_static_features(df)
    # Drop rows where new_cases is still NaN (very first row per country)
    df = df.dropna(subset=['new_cases'])
    df = df.sort_values(['country', 'date']).reset_index(drop=True)
    print(f"  After cleaning: {df.shape}")
    return df


def save_processed(df: pd.DataFrame, fname: str = 'merged_features.csv'):
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    out = os.path.join(PROCESSED_DIR, fname)
    df.to_csv(out, index=False)
    print(f"  Saved processed → {out}")


def save_country_splits(df: pd.DataFrame):
    """Save one CSV per country into data/processed/country_splits/."""
    split_dir = os.path.join(PROCESSED_DIR, 'country_splits')
    os.makedirs(split_dir, exist_ok=True)
    for country, g in df.groupby('country'):
        safe = country.replace(' ', '_').replace('/', '-')
        g.to_csv(os.path.join(split_dir, f'{safe}.csv'), index=False)
    print(f"  Saved {df['country'].nunique()} country split files.")


if __name__ == '__main__':
    from data_loader import build_master
    master = build_master(download=True)
    cleaned = clean(master)
    save_processed(cleaned)
    save_country_splits(cleaned)
