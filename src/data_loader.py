"""
data_loader.py
--------------
Loads and merges the three datasets for Track C:
  1. Johns Hopkins COVID-19 time series (primary)
  2. Our World in Data COVID-19 dataset  (secondary)
  3. Google Mobility Reports             (optional)
"""

import os
import pandas as pd
import numpy as np
import requests

# ── Paths ────────────────────────────────────────────────────────────────────
RAW_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')

JHU_URL = (
    "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/"
    "csse_covid_19_data/csse_covid_19_time_series/"
    "time_series_covid19_confirmed_global.csv"
)
JHU_DEATHS_URL = (
    "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/"
    "csse_covid_19_data/csse_covid_19_time_series/"
    "time_series_covid19_deaths_global.csv"
)
OWID_URL = (
    "https://raw.githubusercontent.com/owid/covid-19-data/master/"
    "public/data/owid-covid-data.csv"
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _download(url: str, dest: str) -> str:
    """Download a file if it does not already exist locally."""
    os.makedirs(RAW_DIR, exist_ok=True)
    fpath = os.path.join(RAW_DIR, dest)
    if not os.path.exists(fpath):
        print(f"  Downloading {dest} ...")
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        with open(fpath, 'wb') as f:
            f.write(r.content)
        print(f"  Saved → {fpath}")
    else:
        print(f"  Found cached: {dest}")
    return fpath


# ── JHU loader ───────────────────────────────────────────────────────────────

def load_jhu_cases(download: bool = True) -> pd.DataFrame:
    """
    Load JHU confirmed cases. Returns long-format DataFrame:
        columns: country, date, confirmed_cumulative
    """
    fpath = _download(JHU_URL, 'jhu_confirmed.csv') if download else \
            os.path.join(RAW_DIR, 'jhu_confirmed.csv')

    df = pd.read_csv(fpath)

    # Drop Province/State, Lat, Long — aggregate to country level
    df = df.drop(columns=['Province/State', 'Lat', 'Long'], errors='ignore')
    df = df.groupby('Country/Region').sum(numeric_only=True).reset_index()
    df = df.rename(columns={'Country/Region': 'country'})

    # Wide → long
    df = df.melt(id_vars='country', var_name='date', value_name='confirmed_cumulative')
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['country', 'date']).reset_index(drop=True)
    return df


def load_jhu_deaths(download: bool = True) -> pd.DataFrame:
    """Load JHU deaths, same shape as load_jhu_cases."""
    fpath = _download(JHU_DEATHS_URL, 'jhu_deaths.csv') if download else \
            os.path.join(RAW_DIR, 'jhu_deaths.csv')

    df = pd.read_csv(fpath)
    df = df.drop(columns=['Province/State', 'Lat', 'Long'], errors='ignore')
    df = df.groupby('Country/Region').sum(numeric_only=True).reset_index()
    df = df.rename(columns={'Country/Region': 'country'})
    df = df.melt(id_vars='country', var_name='date', value_name='deaths_cumulative')
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['country', 'date']).reset_index(drop=True)
    return df


# ── OWID loader ───────────────────────────────────────────────────────────────

def load_owid(download: bool = True) -> pd.DataFrame:
    """
    Load Our World in Data COVID dataset.
    Selects key columns useful for feature engineering:
        country, date, total_vaccinations_per_hundred,
        stringency_index, population, hospital_beds_per_thousand,
        new_tests_per_thousand
    """
    fpath = _download(OWID_URL, 'owid-covid-data.csv') if download else \
            os.path.join(RAW_DIR, 'owid-covid-data.csv')

    keep_cols = [
        'location', 'date',
        'total_vaccinations_per_hundred',
        'people_fully_vaccinated_per_hundred',
        'stringency_index',
        'population',
        'population_density',
        'median_age',
        'hospital_beds_per_thousand',
        'new_tests_per_thousand',
        'life_expectancy',
    ]
    df = pd.read_csv(fpath, usecols=[c for c in keep_cols if c in
                     pd.read_csv(fpath, nrows=0).columns])
    df = df.rename(columns={'location': 'country'})
    df['date'] = pd.to_datetime(df['date'])

    # Drop aggregated regions like "World", "Asia", "Europe"
    excluded = {'World', 'Asia', 'Europe', 'Africa', 'North America',
                'South America', 'Oceania', 'European Union',
                'High income', 'Low income', 'Lower middle income',
                'Upper middle income', 'International'}
    df = df[~df['country'].isin(excluded)]
    return df


# ── Mobility loader ───────────────────────────────────────────────────────────

def load_mobility(fpath: str = None) -> pd.DataFrame | None:
    """
    Load Google Mobility CSV (optional).
    Pass the local file path; returns None if file not found.
    """
    if fpath is None:
        fpath = os.path.join(RAW_DIR, 'Global_Mobility_Report.csv')
    if not os.path.exists(fpath):
        print("  Mobility file not found — skipping (optional dataset).")
        return None

    keep = ['country_region', 'sub_region_1', 'date',
            'retail_and_recreation_percent_change_from_baseline',
            'grocery_and_pharmacy_percent_change_from_baseline',
            'workplaces_percent_change_from_baseline',
            'residential_percent_change_from_baseline',
            'transit_stations_percent_change_from_baseline']

    df = pd.read_csv(fpath, usecols=keep, parse_dates=['date'])
    # Keep only national-level rows (sub_region_1 is null)
    df = df[df['sub_region_1'].isna()].drop(columns='sub_region_1')
    df = df.rename(columns={
        'country_region': 'country',
        'retail_and_recreation_percent_change_from_baseline': 'mob_retail',
        'grocery_and_pharmacy_percent_change_from_baseline': 'mob_grocery',
        'workplaces_percent_change_from_baseline': 'mob_work',
        'residential_percent_change_from_baseline': 'mob_residential',
        'transit_stations_percent_change_from_baseline': 'mob_transit',
    })
    return df


# ── Master merge ─────────────────────────────────────────────────────────────

def build_master(download: bool = True,
                 mobility_path: str = None) -> pd.DataFrame:
    """
    Download (or load from cache) all datasets and return a single merged
    DataFrame keyed on (country, date).
    """
    print("Loading JHU cases...")
    cases = load_jhu_cases(download)
    print("Loading JHU deaths...")
    deaths = load_jhu_deaths(download)
    print("Loading OWID...")
    owid = load_owid(download)

    # Merge cases + deaths
    df = cases.merge(deaths, on=['country', 'date'], how='left')

    # Merge OWID
    df = df.merge(owid, on=['country', 'date'], how='left')

    # Optional mobility
    mob = load_mobility(mobility_path)
    if mob is not None:
        df = df.merge(mob, on=['country', 'date'], how='left')

    print(f"  Master shape: {df.shape}")
    return df


if __name__ == '__main__':
    master = build_master(download=True)
    out = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'master_raw.csv')
    master.to_csv(out, index=False)
    print(f"Saved master → {out}")
