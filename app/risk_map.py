"""
risk_map.py
-----------
Generates a Plotly choropleth risk map from the features DataFrame.
Risk score = 0.5 * normalised(r0_estimate) + 0.3 * normalised(growth_rate)
             + 0.2 * normalised(new_cases_7d)
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


# ISO-3 mapping (partial — common countries)
COUNTRY_ISO = {
    'United States': 'USA', 'US': 'USA',
    'United Kingdom': 'GBR',
    'India': 'IND', 'Brazil': 'BRA', 'France': 'FRA',
    'Germany': 'DEU', 'Russia': 'RUS', 'Italy': 'ITA',
    'Spain': 'ESP', 'Canada': 'CAN', 'Australia': 'AUS',
    'Japan': 'JPN', 'China': 'CHN', 'South Korea': 'KOR',
    'Mexico': 'MEX', 'Argentina': 'ARG', 'South Africa': 'ZAF',
    'Nigeria': 'NGA', 'Pakistan': 'PAK', 'Indonesia': 'IDN',
    'Turkey': 'TUR', 'Saudi Arabia': 'SAU', 'Iran': 'IRN',
    'Poland': 'POL', 'Netherlands': 'NLD', 'Belgium': 'BEL',
    'Sweden': 'SWE', 'Switzerland': 'CHE', 'Portugal': 'PRT',
    'Greece': 'GRC', 'Israel': 'ISR', 'Thailand': 'THA',
    'Malaysia': 'MYS', 'Philippines': 'PHL', 'Vietnam': 'VNM',
    'Colombia': 'COL', 'Chile': 'CHL', 'Peru': 'PER',
}


def _normalise(s: pd.Series) -> pd.Series:
    mn, mx = s.min(), s.max()
    if mx - mn < 1e-9:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - mn) / (mx - mn)


def compute_risk_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes the features DataFrame (one row per country-date).
    Returns a country-level DataFrame with a 0-100 risk score
    computed on the most recent date per country.
    """
    latest = df.sort_values('date').groupby('country').last().reset_index()

    cols_needed = ['r0_estimate', 'growth_rate', 'new_cases_7d']
    for c in cols_needed:
        if c not in latest.columns:
            latest[c] = 0.0

    risk = (
        0.5 * _normalise(latest['r0_estimate'].fillna(0).clip(0)) +
        0.3 * _normalise(latest['growth_rate'].fillna(0).clip(0)) +
        0.2 * _normalise(latest['new_cases_7d'].fillna(0).clip(0))
    ) * 100

    latest['risk_score'] = risk.round(1)

    # Categorise
    def _cat(x):
        if x >= 67: return 'High'
        if x >= 33: return 'Medium'
        return 'Low'

    latest['risk_level'] = latest['risk_score'].apply(_cat)
    latest['iso3'] = latest['country'].map(COUNTRY_ISO)
    return latest[['country', 'iso3', 'risk_score', 'risk_level',
                   'r0_estimate', 'growth_rate', 'new_cases_7d',
                   'confirmed_cumulative']].dropna(subset=['iso3'])


def build_choropleth(risk_df: pd.DataFrame) -> go.Figure:
    """Return a Plotly Figure of the world risk map."""
    fig = px.choropleth(
        risk_df,
        locations='iso3',
        color='risk_score',
        hover_name='country',
        hover_data={
            'risk_score': ':.1f',
            'risk_level': True,
            'r0_estimate': ':.2f',
            'growth_rate': ':.2f',
            'new_cases_7d': ':,.0f',
        },
        color_continuous_scale=[
            [0.0, '#0d1b2a'],
            [0.3, '#00c896'],
            [0.6, '#ffb830'],
            [1.0, '#ff4d6d'],
        ],
        range_color=(0, 100),
        title='',
        labels={'risk_score': 'Risk Score'},
    )
    fig.update_layout(
        paper_bgcolor='#050a0f',
        plot_bgcolor='#050a0f',
        geo=dict(
            bgcolor='#050a0f',
            showframe=False,
            showcoastlines=True,
            coastlinecolor='#1a3040',
            showland=True,
            landcolor='#0a1520',
            showocean=True,
            oceancolor='#050a0f',
            showlakes=False,
            lakecolor='#050a0f',
            projection_type='natural earth',
        ),
        coloraxis_colorbar=dict(
            title='Risk',
            titlefont=dict(color='#7a9e90', size=11),
            tickfont=dict(color='#7a9e90', size=10),
            bgcolor='#0d1b2a',
            bordercolor='#1a3040',
            borderwidth=1,
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        font=dict(color='#d0e8e0'),
    )
    return fig
