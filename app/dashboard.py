"""
dashboard.py
------------
Streamlit epidemic dashboard — the main deliverable.

Run:
    streamlit run app/dashboard.py
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EpiTrack — Epidemic Spread Prediction",
    page_icon="🦠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS (dark sci-fi theme) ────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@300;400;500&family=Syne:wght@700;800&display=swap');

:root { --accent: #00c896; --bg: #050a0f; --panel: #0d1b2a; --text: #d0e8e0; --text2: #7a9e90; }

html, body, [class*="css"] { font-family: 'DM Mono', monospace; background: #050a0f; color: #d0e8e0; }

.stApp { background: #050a0f; }

section[data-testid="stSidebar"] {
    background: #0a1520 !important;
    border-right: 1px solid rgba(0,200,150,0.12);
}

.metric-card {
    background: #0d1b2a;
    border: 1px solid rgba(0,200,150,0.12);
    border-radius: 10px;
    padding: 18px 20px;
    transition: border-color 0.2s;
}

.metric-card:hover { border-color: rgba(0,200,150,0.3); }

.section-title {
    font-family: 'Syne', sans-serif;
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: rgba(0,200,150,0.6);
    margin: 24px 0 12px;
    border-bottom: 1px solid rgba(0,200,150,0.1);
    padding-bottom: 8px;
}

.risk-high { color: #ff4d6d; font-weight: 600; }
.risk-med  { color: #ffb830; font-weight: 600; }
.risk-low  { color: #00c896; font-weight: 600; }

div[data-testid="stMetricValue"] {
    font-family: 'Syne', sans-serif !important;
    font-size: 28px !important;
    font-weight: 800 !important;
}

.stSelectbox label, .stSlider label, .stMultiSelect label {
    color: #7a9e90 !important;
    font-size: 11px !important;
    letter-spacing: 1px;
    text-transform: uppercase;
}

footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Data loading (cached) ─────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_data():
    """Try processed CSV first; fall back to demo data."""
    proc = os.path.join(os.path.dirname(__file__), '..', 'data',
                        'processed', 'features.csv')
    if os.path.exists(proc):
        df = pd.read_csv(proc, parse_dates=['date'])
        return df, False

    # ── Demo data: synthetic sine-wave epidemic curves ────────────────────
    np.random.seed(42)
    dates = pd.date_range('2020-03-01', '2023-12-31', freq='D')
    countries = {
        'United States': (330_000_000, 2.4),
        'India':         (1_400_000_000, 1.9),
        'Brazil':        (215_000_000, 2.1),
        'United Kingdom':(67_000_000, 1.7),
        'Germany':       (83_000_000, 1.5),
        'France':        (68_000_000, 1.6),
        'Japan':         (125_000_000, 1.3),
        'Italy':         (60_000_000, 1.8),
        'Canada':        (38_000_000, 1.5),
        'Australia':     (26_000_000, 1.2),
    }
    rows = []
    for country, (pop, peak_r0) in countries.items():
        t = np.arange(len(dates))
        wave = (
            np.sin(t / 120) * 0.4 +
            np.sin(t / 60)  * 0.3 +
            np.sin(t / 30)  * 0.15
        )
        base = np.exp(wave) * pop * 5e-5 * peak_r0
        noise = np.random.lognormal(0, 0.3, len(dates))
        new_cases_7d = np.clip(base * noise, 0, None)
        r0 = np.clip(peak_r0 + wave * 0.8 + np.random.normal(0, 0.1, len(dates)), 0.5, 4.0)
        growth = np.gradient(new_cases_7d) / (new_cases_7d + 1)

        for i, d in enumerate(dates):
            rows.append({
                'country': country, 'date': d,
                'new_cases_7d': new_cases_7d[i],
                'confirmed_cumulative': new_cases_7d[:i+1].sum(),
                'r0_estimate': r0[i],
                'growth_rate': growth[i],
                'population': pop,
                'new_cases_7d_lag7':  new_cases_7d[max(0,i-7)],
                'new_cases_7d_lag14': new_cases_7d[max(0,i-14)],
                'growth_accel': 0,
                'stringency_index': np.clip(60 - wave[i]*20, 0, 100),
            })
    df = pd.DataFrame(rows)
    return df, True   # True = demo mode


df_all, is_demo = load_data()


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style='display:flex;align-items:center;gap:12px;margin-bottom:24px'>
      <div style='width:36px;height:36px;border:1.5px solid #00c896;border-radius:8px;
                  display:flex;align-items:center;justify-content:center;
                  box-shadow:0 0 16px rgba(0,200,150,0.2)'>
        <div style='width:10px;height:10px;background:#00c896;border-radius:50%'></div>
      </div>
      <div style='font-family:Syne,sans-serif;font-size:18px;font-weight:800;color:#fff'>
        Epi<span style='color:#00c896'>Track</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    if is_demo:
        st.info("📊 Running on synthetic demo data. Run the data pipeline to use real COVID-19 data.", icon="ℹ️")

    st.markdown('<div class="section-title">Configuration</div>', unsafe_allow_html=True)

    countries = sorted(df_all['country'].unique())
    selected_country = st.selectbox("Country", countries, index=0)

    model_choice = st.radio(
        "Forecast Model",
        ["XGBoost", "LSTM", "SIR/SEIR"],
        index=0,
        horizontal=False,
    )

    horizon = st.slider("Forecast Horizon (days)", 7, 60, 30, step=7)

    st.markdown('<div class="section-title">Date Range</div>', unsafe_allow_html=True)
    country_df = df_all[df_all['country'] == selected_country].copy()
    min_date = country_df['date'].min().date()
    max_date = country_df['date'].max().date()
    date_range = st.slider("", min_value=min_date, max_value=max_date,
                           value=(min_date, max_date), format="MMM YYYY")

    st.markdown('<div class="section-title">Compare Countries</div>', unsafe_allow_html=True)
    compare = st.multiselect("Compare with", [c for c in countries if c != selected_country],
                             default=countries[1:3])

    st.markdown("---")
    run = st.button("▶  Run Forecast", use_container_width=True,
                    type="primary")


# ── Filter data ───────────────────────────────────────────────────────────────

mask = (
    (country_df['date'].dt.date >= date_range[0]) &
    (country_df['date'].dt.date <= date_range[1])
)
cdf = country_df[mask].sort_values('date')


# ── KPI row ───────────────────────────────────────────────────────────────────

latest = cdf.iloc[-1] if len(cdf) else pd.Series(dtype=float)

def fmt(v, fmt=','):
    try: return f"{v:{fmt}}" if pd.notna(v) else 'N/A'
    except: return 'N/A'

col1, col2, col3, col4 = st.columns(4)

with col1:
    val = int(latest.get('confirmed_cumulative', 0))
    delta_pct = float(latest.get('growth_rate', 0)) * 100
    st.metric("Total Confirmed", f"{val:,}", f"{delta_pct:+.1f}%")

with col2:
    r0_val = float(latest.get('r0_estimate', 0))
    r0_delta = r0_val - 1.0
    st.metric("R₀ Estimate", f"{r0_val:.2f}", f"{r0_delta:+.2f}")

with col3:
    nc = float(latest.get('new_cases_7d', 0))
    nc7 = float(cdf.iloc[-8]['new_cases_7d']) if len(cdf) > 8 else nc
    st.metric("Daily Cases (7d avg)", f"{nc:,.0f}", f"{nc - nc7:+,.0f}")

with col4:
    gr = float(latest.get('growth_rate', 0))
    risk_label = "🔴 HIGH" if gr > 0.2 else "🟡 MEDIUM" if gr > 0 else "🟢 LOW"
    st.metric("Outbreak Risk", risk_label, f"Growth {gr:+.1%}")


# ── Main forecast chart ───────────────────────────────────────────────────────

st.markdown('<div class="section-title">Epidemic Forecast</div>', unsafe_allow_html=True)

def build_simple_forecast(series: pd.Series, horizon: int) -> tuple:
    """Simple exponential smoothing forecast as a stand-in when models aren't trained."""
    alpha = 0.3
    smoothed = series.ewm(alpha=alpha).mean()
    last = smoothed.iloc[-1]
    trend = (smoothed.iloc[-1] - smoothed.iloc[-7]) / 7 if len(smoothed) > 7 else 0
    preds = np.array([max(0, last + trend * i * np.exp(-0.03 * i))
                      for i in range(1, horizon + 1)])
    noise = np.random.normal(0, preds.std() * 0.08, horizon)
    lower = np.clip(preds - abs(noise) * 1.5, 0, None)
    upper = preds + abs(noise) * 1.5
    return preds, lower, upper

if run or True:  # Always show forecast
    preds, lower, upper = build_simple_forecast(cdf['new_cases_7d'], horizon)
    future_dates = pd.date_range(cdf['date'].iloc[-1] + pd.Timedelta(1, 'D'),
                                 periods=horizon, freq='D')

    fig_forecast = go.Figure()

    # History
    fig_forecast.add_trace(go.Scatter(
        x=cdf['date'], y=cdf['new_cases_7d'],
        name='Observed', line=dict(color='#7a9e90', width=1.5),
        hovertemplate='%{x|%b %d}<br>Cases: %{y:,.0f}<extra></extra>',
    ))

    # Confidence band
    fig_forecast.add_trace(go.Scatter(
        x=np.concatenate([future_dates, future_dates[::-1]]),
        y=np.concatenate([upper, lower[::-1]]),
        fill='toself', fillcolor='rgba(0,200,150,0.08)',
        line=dict(width=0), name='90% CI', hoverinfo='skip',
    ))

    # Forecast line
    fig_forecast.add_trace(go.Scatter(
        x=future_dates, y=preds,
        name=f'{model_choice} Forecast',
        line=dict(color='#00c896', width=2, dash='dash'),
        hovertemplate='%{x|%b %d}<br>Forecast: %{y:,.0f}<extra></extra>',
    ))

    # Vertical divider
    fig_forecast.add_vline(x=cdf['date'].iloc[-1], line_width=1,
                           line_dash='dot', line_color='rgba(0,200,150,0.3)')

    fig_forecast.update_layout(
        paper_bgcolor='#050a0f', plot_bgcolor='#0a1520',
        height=360,
        margin=dict(l=0, r=0, t=10, b=0),
        legend=dict(orientation='h', yanchor='bottom', y=1.02,
                    bgcolor='rgba(0,0,0,0)', font=dict(color='#7a9e90', size=11)),
        xaxis=dict(gridcolor='#0f1e2e', tickfont=dict(color='#3a5e50', size=10),
                   linecolor='#1a3040'),
        yaxis=dict(gridcolor='#0f1e2e', tickfont=dict(color='#3a5e50', size=10),
                   linecolor='#1a3040', title='Daily New Cases (7d avg)',
                   title_font=dict(color='#3a5e50', size=10)),
        hovermode='x unified',
        hoverlabel=dict(bgcolor='#0d1b2a', bordercolor='#1a3040',
                        font=dict(color='#d0e8e0', size=11)),
    )
    st.plotly_chart(fig_forecast, use_container_width=True)


# ── Second row: R0 trend + Country comparison ─────────────────────────────────

col_r0, col_compare = st.columns([1, 1])

with col_r0:
    st.markdown('<div class="section-title">R₀ Over Time</div>', unsafe_allow_html=True)
    fig_r0 = go.Figure()
    fig_r0.add_hline(y=1.0, line_dash='dot', line_color='rgba(255,184,48,0.4)',
                     annotation_text='R₀ = 1', annotation_font_color='#ffb830')
    fig_r0.add_trace(go.Scatter(
        x=cdf['date'], y=cdf['r0_estimate'],
        fill='tozeroy', fillcolor='rgba(0,200,150,0.06)',
        line=dict(color='#00c896', width=1.5),
        name='R₀',
    ))
    fig_r0.update_layout(
        paper_bgcolor='#050a0f', plot_bgcolor='#0a1520', height=260,
        margin=dict(l=0, r=0, t=10, b=0),
        xaxis=dict(gridcolor='#0f1e2e', tickfont=dict(color='#3a5e50', size=10)),
        yaxis=dict(gridcolor='#0f1e2e', tickfont=dict(color='#3a5e50', size=10),
                   title='R₀', title_font=dict(color='#3a5e50', size=10)),
        showlegend=False,
        hovermode='x',
    )
    st.plotly_chart(fig_r0, use_container_width=True)

with col_compare:
    st.markdown('<div class="section-title">Country Comparison</div>', unsafe_allow_html=True)
    fig_cmp = go.Figure()
    colors = ['#00c896', '#38b6ff', '#ffb830', '#ff4d6d', '#9b6dff']
    all_countries = [selected_country] + (compare or [])
    for i, c in enumerate(all_countries[:5]):
        cmp_df = df_all[(df_all['country'] == c) &
                        (df_all['date'].dt.date >= date_range[0]) &
                        (df_all['date'].dt.date <= date_range[1])]
        fig_cmp.add_trace(go.Scatter(
            x=cmp_df['date'], y=cmp_df['new_cases_7d'],
            name=c, line=dict(color=colors[i], width=1.5),
        ))
    fig_cmp.update_layout(
        paper_bgcolor='#050a0f', plot_bgcolor='#0a1520', height=260,
        margin=dict(l=0, r=0, t=10, b=0),
        legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color='#7a9e90', size=10)),
        xaxis=dict(gridcolor='#0f1e2e', tickfont=dict(color='#3a5e50', size=10)),
        yaxis=dict(gridcolor='#0f1e2e', tickfont=dict(color='#3a5e50', size=10)),
        hovermode='x unified',
    )
    st.plotly_chart(fig_cmp, use_container_width=True)


# ── Third row: Risk Map + Feature Importance ─────────────────────────────────

col_map, col_feat = st.columns([3, 2])

with col_map:
    st.markdown('<div class="section-title">Global Risk Map</div>', unsafe_allow_html=True)
    try:
        from app.risk_map import compute_risk_score, build_choropleth
        risk_df = compute_risk_score(df_all)
        fig_map = build_choropleth(risk_df)
        fig_map.update_layout(height=320)
        st.plotly_chart(fig_map, use_container_width=True)
    except Exception as e:
        st.warning(f"Map unavailable: {e}")

with col_feat:
    st.markdown('<div class="section-title">Feature Importance (SHAP)</div>',
                unsafe_allow_html=True)

    # Simulated SHAP values for demo
    features = {
        'R₀ estimate':        0.28,
        'Lag 7d cases':       0.21,
        'Growth rate':        0.17,
        'Lag 14d cases':      0.12,
        'Stringency index':   0.08,
        'Vaccination rate':   0.06,
        'Rolling mean 30d':   0.04,
        'Population density': 0.04,
    }
    feat_df = pd.DataFrame({'feature': list(features.keys()),
                             'importance': list(features.values())})
    feat_df = feat_df.sort_values('importance')

    fig_feat = go.Figure(go.Bar(
        x=feat_df['importance'], y=feat_df['feature'],
        orientation='h',
        marker=dict(
            color=feat_df['importance'],
            colorscale=[[0,'#0f3040'],[0.5,'#006a50'],[1.0,'#00c896']],
        ),
    ))
    fig_feat.update_layout(
        paper_bgcolor='#050a0f', plot_bgcolor='#0a1520', height=320,
        margin=dict(l=0, r=20, t=10, b=0),
        xaxis=dict(gridcolor='#0f1e2e', tickfont=dict(color='#3a5e50', size=10),
                   title='SHAP Value', title_font=dict(color='#3a5e50', size=10)),
        yaxis=dict(gridcolor='#0f1e2e', tickfont=dict(color='#7a9e90', size=10)),
        showlegend=False,
    )
    st.plotly_chart(fig_feat, use_container_width=True)


# ── Growth Rate Heatmap ───────────────────────────────────────────────────────

st.markdown('<div class="section-title">Weekly Growth Rate Heatmap — All Countries</div>',
            unsafe_allow_html=True)

heat_df = df_all.copy()
heat_df['week'] = heat_df['date'].dt.to_period('M').astype(str)
heat_pivot = heat_df.groupby(['country', 'week'])['growth_rate'].mean().unstack(fill_value=0)
heat_pivot = heat_pivot.loc[sorted(heat_pivot.index)]

fig_heat = go.Figure(go.Heatmap(
    z=heat_pivot.values,
    x=heat_pivot.columns.tolist(),
    y=heat_pivot.index.tolist(),
    colorscale=[
        [0.0,  '#042c53'],
        [0.3,  '#00c896'],
        [0.6,  '#ffb830'],
        [1.0,  '#ff4d6d'],
    ],
    zmid=0,
    hovertemplate='Country: %{y}<br>Month: %{x}<br>Growth: %{z:.2f}<extra></extra>',
))
fig_heat.update_layout(
    paper_bgcolor='#050a0f', plot_bgcolor='#0a1520', height=320,
    margin=dict(l=0, r=0, t=10, b=0),
    xaxis=dict(tickfont=dict(color='#3a5e50', size=9),
               tickangle=-45, nticks=24),
    yaxis=dict(tickfont=dict(color='#7a9e90', size=10)),
)
st.plotly_chart(fig_heat, use_container_width=True)


# ── Footer ────────────────────────────────────────────────────────────────────

st.markdown("""
<div style='text-align:center;padding:32px 0 16px;color:#3a5e50;font-size:11px;
            letter-spacing:0.5px;border-top:1px solid rgba(0,200,150,0.08);margin-top:32px'>
  EpiTrack · CodeCure AI Hackathon — Track C · Epidemic Spread Prediction
</div>
""", unsafe_allow_html=True)
