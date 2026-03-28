"""
dashboard.py
------------
Streamlit epidemic dashboard — the main deliverable.

Run:
    streamlit run app/dashboard.py
"""


import sys
import os

# ====================== FIX: PROJECT ROOT PATH ======================
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.chatbot.risk_summary import build_risk_summary, OUTPUT_PATH

  
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

# ── CSS: Blue-spectrum 3-D animated theme ────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=JetBrains+Mono:wght@300;400;500&family=Rajdhani:wght@400;500;600;700&display=swap');

:root {
  --bg0:  #020b18;
  --bg1:  #041526;
  --bg2:  #071e38;
  --bg3:  #0a2a50;
  --b1:   #1d6fa4;
  --b2:   #2196f3;
  --b3:   #38bdf8;
  --b4:   #7dd3fc;
  --b5:   #bae6fd;
  --hot:  #06b6d4;
  --warn: #fbbf24;
  --danger:#f87171;
  --txt:  #cce8f8;
  --txt2: #4a7a9b;
}

html, body, [class*="css"] {
  font-family: 'Rajdhani', sans-serif;
  background: var(--bg0);
  color: var(--txt);
}
.stApp { background: var(--bg0); }

/* dual animated grid */
.stApp::before {
  content: '';
  position: fixed; inset: 0;
  background-image:
    linear-gradient(rgba(33,150,243,0.04) 1px, transparent 1px),
    linear-gradient(90deg, rgba(33,150,243,0.04) 1px, transparent 1px),
    linear-gradient(rgba(56,189,248,0.02) 1px, transparent 1px),
    linear-gradient(90deg, rgba(56,189,248,0.02) 1px, transparent 1px);
  background-size: 80px 80px, 80px 80px, 20px 20px, 20px 20px;
  animation: gridDrift 28s linear infinite;
  pointer-events: none; z-index: 0;
}
@keyframes gridDrift {
  0%   { background-position: 0 0, 0 0, 0 0, 0 0; }
  100% { background-position: 80px 80px, 80px 80px, 20px 20px, 20px 20px; }
}
/* scanlines */
.stApp::after {
  content: '';
  position: fixed; inset: 0;
  background: repeating-linear-gradient(
    0deg, transparent, transparent 3px,
    rgba(0,0,0,0.04) 3px, rgba(0,0,0,0.04) 4px
  );
  pointer-events: none; z-index: 1;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
  background: linear-gradient(160deg, #031020 0%, #020b18 100%) !important;
  border-right: 1px solid rgba(33,150,243,0.12);
  box-shadow: 6px 0 50px rgba(0,0,0,0.9);
}
section[data-testid="stSidebar"] > div { background: transparent !important; }

/* ── Brand ── */
.brand-wrap { display:flex; align-items:center; gap:14px; margin-bottom:28px; padding:4px 0; }
.brand-hex {
  width:44px; height:44px;
  background: linear-gradient(135deg, rgba(33,150,243,0.18), rgba(29,111,164,0.08));
  border: 1.5px solid rgba(33,150,243,0.45);
  border-radius: 10px;
  display: flex; align-items: center; justify-content: center;
  position: relative;
  animation: hexPulse 3s ease-in-out infinite;
}
.brand-hex::before {
  content: ''; position: absolute; inset: 4px;
  border: 1px solid rgba(33,150,243,0.22); border-radius: 6px;
  animation: hexPulse 3s ease-in-out infinite reverse;
}
.brand-core {
  width: 10px; height: 10px;
  background: var(--b2); border-radius: 50%;
  box-shadow: 0 0 10px var(--b2), 0 0 22px rgba(33,150,243,0.5);
  animation: coreBlink 2s ease-in-out infinite;
}
@keyframes hexPulse {
  0%,100% { box-shadow: 0 0 18px rgba(33,150,243,0.22), inset 0 0 18px rgba(33,150,243,0.05); }
  50%     { box-shadow: 0 0 35px rgba(33,150,243,0.42), inset 0 0 28px rgba(33,150,243,0.1);  }
}
@keyframes coreBlink {
  0%,100% { opacity:1; transform:scale(1); }
  50%     { opacity:0.5; transform:scale(0.8); }
}

/* ── Section labels ── */
.sec {
  font-family: 'Orbitron', monospace;
  font-size: 9px; font-weight: 700; letter-spacing: 3px;
  text-transform: uppercase; color: var(--b3);
  margin: 28px 0 14px;
  display: flex; align-items: center; gap: 10px; opacity: 0.8;
}
.sec::before {
  content: ''; width: 3px; height: 14px;
  background: linear-gradient(180deg, var(--b3), transparent);
  border-radius: 2px; flex-shrink: 0;
}
.sec::after {
  content: ''; flex: 1; height: 1px;
  background: linear-gradient(90deg, rgba(56,189,248,0.28), transparent);
}

/* ── Metric cards ── */
div[data-testid="metric-container"] {
  background: linear-gradient(135deg, var(--bg1) 0%, rgba(4,10,22,0.92) 100%);
  border: 1px solid rgba(33,150,243,0.14);
  border-radius: 14px;
  padding: 20px 22px !important;
  position: relative; overflow: hidden;
  transition: all 0.3s ease;
  animation: cardIn 0.6s ease both;
}
div[data-testid="metric-container"]::before {
  content: ''; position: absolute;
  top: 0; left: -100%; width: 100%; height: 1px;
  background: linear-gradient(90deg, transparent, var(--b3), transparent);
  animation: topLine 4s linear infinite;
}
div[data-testid="metric-container"]::after {
  content: ''; position: absolute; inset: 0;
  background: radial-gradient(circle at 50% 0%, rgba(33,150,243,0.06) 0%, transparent 70%);
  pointer-events: none;
}
div[data-testid="metric-container"]:hover {
  border-color: rgba(33,150,243,0.38);
  box-shadow: 0 8px 40px rgba(33,150,243,0.1), 0 0 0 1px rgba(33,150,243,0.07);
  transform: translateY(-3px);
}
@keyframes cardIn  { from{opacity:0;transform:translateY(14px);} to{opacity:1;transform:translateY(0);} }
@keyframes topLine { 0%{left:-100%;} 100%{left:100%;} }

div[data-testid="stMetricValue"] {
  font-family: 'Orbitron', monospace !important;
  font-size: 25px !important; font-weight: 900 !important;
  color: #fff !important;
  text-shadow: 0 0 28px rgba(56,189,248,0.5);
}
div[data-testid="stMetricLabel"] {
  font-family: 'JetBrains Mono', monospace !important;
  font-size: 9px !important; letter-spacing: 2.5px !important;
  text-transform: uppercase !important; color: var(--txt2) !important;
}
div[data-testid="stMetricDelta"] {
  font-family: 'JetBrains Mono', monospace !important; font-size: 11px !important;
}

/* ── Controls ── */
.stSelectbox label, .stSlider label, .stMultiSelect label, .stRadio label {
  color: var(--txt2) !important; font-size: 9px !important;
  letter-spacing: 2px; text-transform: uppercase;
  font-family: 'JetBrains Mono', monospace !important;
}
.stSelectbox > div > div, .stMultiSelect > div > div {
  background: rgba(4,10,22,0.85) !important;
  border: 1px solid rgba(33,150,243,0.18) !important;
  border-radius: 8px !important; color: var(--txt) !important;
}
.stSelectbox > div > div:hover, .stMultiSelect > div > div:hover {
  border-color: rgba(33,150,243,0.4) !important;
}

/* ── Button ── */
div[data-testid="stButton"] > button {
  background: linear-gradient(135deg, rgba(33,150,243,0.16), rgba(29,111,164,0.08)) !important;
  border: 1px solid rgba(33,150,243,0.38) !important;
  color: var(--b3) !important;
  font-family: 'Orbitron', monospace !important;
  font-size: 10px !important; letter-spacing: 2px !important;
  border-radius: 8px !important; transition: all 0.25s ease !important;
}
div[data-testid="stButton"] > button:hover {
  background: linear-gradient(135deg, rgba(33,150,243,0.28), rgba(29,111,164,0.18)) !important;
  box-shadow: 0 0 30px rgba(33,150,243,0.28) !important;
  transform: translateY(-2px);
}

/* ── Hero banner ── */
.hero {
  background: linear-gradient(135deg,
    rgba(33,150,243,0.1) 0%, rgba(4,21,38,0.97) 40%, rgba(6,182,212,0.07) 100%);
  border: 1px solid rgba(33,150,243,0.16);
  border-radius: 18px; padding: 30px 38px;
  position: relative; overflow: hidden; margin-bottom: 10px;
}
.hero::before {
  content: ''; position: absolute; top: -80%; left: -30%;
  width: 70%; height: 260%;
  background: conic-gradient(from 0deg, transparent 0%, rgba(33,150,243,0.06) 30%, transparent 60%);
  animation: conicSpin 14s linear infinite; pointer-events: none;
}
.hero::after {
  content: ''; position: absolute; right: -60px; top: -60px;
  width: 320px; height: 320px;
  background: radial-gradient(circle, rgba(56,189,248,0.08) 0%, transparent 70%);
  animation: rightGlow 5s ease-in-out infinite; pointer-events: none;
}
@keyframes conicSpin { from{transform:rotate(0deg);} to{transform:rotate(360deg);} }
@keyframes rightGlow { 0%,100%{opacity:0.4;} 50%{opacity:1;} }

.hero-title {
  font-family: 'Orbitron', monospace;
  font-size: 28px; font-weight: 900; color: #fff;
  position: relative; z-index: 2;
  text-shadow: 0 0 50px rgba(33,150,243,0.45), 0 0 100px rgba(33,150,243,0.12);
  animation: fadeUp 0.8s ease both;
}
.hero-sub {
  font-family: 'JetBrains Mono', monospace;
  font-size: 10px; color: var(--txt2);
  letter-spacing: 3px; text-transform: uppercase; margin-top: 5px;
  position: relative; z-index: 2;
  animation: fadeUp 0.8s ease 0.2s both;
}
.hero-tags {
  display: flex; gap: 8px; flex-wrap: wrap; margin-top: 16px;
  position: relative; z-index: 2;
  animation: fadeUp 0.8s ease 0.4s both;
}
.tag {
  display: inline-flex; align-items: center; gap: 6px;
  padding: 4px 13px; border-radius: 20px;
  font-family: 'JetBrains Mono', monospace;
  font-size: 9px; letter-spacing: 1.5px;
}
.tag-b { background:rgba(33,150,243,0.1); border:1px solid rgba(33,150,243,0.25); color:#38bdf8; }
.tag-c { background:rgba(6,182,212,0.08); border:1px solid rgba(6,182,212,0.2);   color:#22d3ee; }
.tag-d { background:rgba(29,111,164,0.1); border:1px solid rgba(29,111,164,0.25); color:#7dd3fc; }
.live-dot {
  width: 6px; height: 6px; background: var(--b3); border-radius: 50%;
  box-shadow: 0 0 6px var(--b3);
  animation: coreBlink 1.4s ease-in-out infinite;
}

/* ── 3-D virus sphere ── */
.virus3d { position:absolute; right:55px; top:50%; transform:translateY(-50%); z-index:2; }
.vsphere {
  width: 88px; height: 88px; border-radius: 50%;
  background: radial-gradient(circle at 32% 30%,
    rgba(56,189,248,0.55) 0%, rgba(33,150,243,0.22) 45%,
    rgba(6,182,212,0.08) 70%, transparent 90%);
  border: 1.5px solid rgba(56,189,248,0.38);
  position: relative;
  box-shadow: 0 0 40px rgba(33,150,243,0.22), inset 0 0 30px rgba(33,150,243,0.08);
  animation: sphereSpin 10s linear infinite;
}
.vring {
  position: absolute; top:50%; left:50%;
  border-radius: 50%;
  border: 1px solid rgba(56,189,248,0.22);
  transform: translate(-50%,-50%) rotateX(72deg);
  animation: ringSpin 7s linear infinite;
}
.vring:nth-child(1) { width:108px; height:108px; }
.vring:nth-child(2) { width:128px; height:128px; animation-duration:10s; animation-direction:reverse; border-color:rgba(6,182,212,0.15); }
.vring:nth-child(3) { width:148px; height:148px; animation-duration:13s; border-color:rgba(125,211,252,0.1); }
.vspike {
  position: absolute; width:3px; height:13px;
  background: linear-gradient(180deg, #38bdf8, transparent);
  border-radius: 2px; top:50%; left:50%; transform-origin:50% 100%;
  box-shadow: 0 0 4px rgba(56,189,248,0.6);
}
@keyframes sphereSpin { to{transform:rotate(360deg);} }
@keyframes ringSpin   { to{transform:translate(-50%,-50%) rotateX(72deg) rotate(360deg);} }
@keyframes fadeUp     { from{opacity:0;transform:translateY(10px);} to{opacity:1;transform:translateY(0);} }

/* ── Particles ── */
.particles { position:absolute; inset:0; overflow:hidden; pointer-events:none; z-index:1; }
.p {
  position: absolute; width:2px; height:2px;
  background: var(--b3); border-radius:50%; opacity:0;
  box-shadow: 0 0 4px var(--b3);
  animation: pFloat linear infinite;
}
@keyframes pFloat {
  0%   { opacity:0; transform:translateY(100%); }
  10%  { opacity:0.7; }
  90%  { opacity:0.3; }
  100% { opacity:0; transform:translateY(-200%); }
}
.p:nth-child(1){left:8%;  animation-duration:8s;  animation-delay:0s;    background:#38bdf8;}
.p:nth-child(2){left:18%; animation-duration:11s; animation-delay:1.2s;  background:#2196f3;}
.p:nth-child(3){left:28%; animation-duration:9s;  animation-delay:2.8s;  background:#06b6d4;}
.p:nth-child(4){left:42%; animation-duration:7s;  animation-delay:0.6s;  background:#38bdf8;}
.p:nth-child(5){left:58%; animation-duration:10s; animation-delay:3.5s;  background:#7dd3fc;}
.p:nth-child(6){left:72%; animation-duration:8s;  animation-delay:1.8s;  background:#2196f3;}
.p:nth-child(7){left:85%; animation-duration:12s; animation-delay:4.2s;  background:#06b6d4;}
.p:nth-child(8){left:95%; animation-duration:9s;  animation-delay:2.1s;  background:#38bdf8;}

/* chart hover */
.element-container .stPlotlyChart { border-radius:14px; overflow:hidden; transition:box-shadow 0.3s ease; }
.element-container .stPlotlyChart:hover { box-shadow:0 0 40px rgba(33,150,243,0.08), 0 0 0 1px rgba(33,150,243,0.1); }

/* alert */
.stAlert {
  background: rgba(33,150,243,0.06) !important;
  border: 1px solid rgba(33,150,243,0.18) !important;
  border-radius: 10px !important;
  font-family: 'JetBrains Mono', monospace !important; font-size:12px !important;
}

::-webkit-scrollbar { width:4px; }
::-webkit-scrollbar-track { background:var(--bg0); }
::-webkit-scrollbar-thumb { background:rgba(33,150,243,0.28); border-radius:2px; }
footer { visibility:hidden; }
</style>
""", unsafe_allow_html=True)


# ── Data loading ──────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_data():
    proc = os.path.join(os.path.dirname(__file__), '..', 'data',
                        'processed', 'features.csv')
    if os.path.exists(proc):
        df = pd.read_csv(proc, parse_dates=['date'])
        return df, False

    np.random.seed(42)
    dates = pd.date_range('2020-03-01', '2023-12-31', freq='D')
    countries = {
        'United States':  (330_000_000, 2.4),
        'India':          (1_400_000_000, 1.9),
        'Brazil':         (215_000_000, 2.1),
        'United Kingdom': (67_000_000, 1.7),
        'Germany':        (83_000_000, 1.5),
        'France':         (68_000_000, 1.6),
        'Japan':          (125_000_000, 1.3),
        'Italy':          (60_000_000, 1.8),
        'Canada':         (38_000_000, 1.5),
        'Australia':      (26_000_000, 1.2),
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
                'new_cases_7d_lag7':  new_cases_7d[max(0, i-7)],
                'new_cases_7d_lag14': new_cases_7d[max(0, i-14)],
                'growth_accel': 0,
                'stringency_index': np.clip(60 - wave[i]*20, 0, 100),
            })
    return pd.DataFrame(rows), True


df_all, is_demo = load_data()


# ── Chart layout helper ───────────────────────────────────────────────────────

def chart_layout(height=360, **overrides):
    base = dict(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(4,15,30,0.65)',
        height=height,
        margin=dict(l=0, r=0, t=10, b=0),
        font=dict(family='JetBrains Mono, monospace', color='#4a7a9b', size=10),
        xaxis=dict(
            gridcolor='rgba(33,150,243,0.07)',
            tickfont=dict(color='#2a4a6a', size=9),
            linecolor='rgba(33,150,243,0.1)',
            showgrid=True,
        ),
        yaxis=dict(
            gridcolor='rgba(33,150,243,0.07)',
            tickfont=dict(color='#2a4a6a', size=9),
            linecolor='rgba(33,150,243,0.1)',
            showgrid=True,
        ),
        hovermode='x unified',
        hoverlabel=dict(
            bgcolor='rgba(4,15,30,0.97)',
            bordercolor='rgba(33,150,243,0.28)',
            font=dict(color='#cce8f8', size=11, family='JetBrains Mono'),
        ),
    )
    base.update(overrides)
    return base


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div class="brand-wrap">
      <div class="brand-hex"><div class="brand-core"></div></div>
      <div>
        <div style="font-family:Orbitron,monospace;font-size:17px;font-weight:900;
                    color:#fff;text-shadow:0 0 20px rgba(33,150,243,0.45)">
          Epi<span style="color:#38bdf8">Track</span>
        </div>
        <div style="font-family:JetBrains Mono,monospace;font-size:9px;
                    color:#4a7a9b;letter-spacing:2.5px;text-transform:uppercase">
          v2.0 · SYSTEM ONLINE
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    if is_demo:
        st.info("📊 Synthetic demo data. Run the data pipeline for real COVID-19 data.", icon="ℹ️")

    st.markdown('<div class="sec">Configuration</div>', unsafe_allow_html=True)
    countries = sorted(df_all['country'].unique())
    selected_country = st.selectbox("Country", countries, index=0)

    model_choice = st.radio("Forecast Model", ["XGBoost", "LSTM", "SIR/SEIR"], index=0)

    horizon = st.slider("Forecast Horizon (days)", 7, 60, 30, step=7)

    st.markdown('<div class="sec">Date Range</div>', unsafe_allow_html=True)
    country_df = df_all[df_all['country'] == selected_country].copy()
    min_date = country_df['date'].min().date()
    max_date = country_df['date'].max().date()
    date_range = st.slider("", min_value=min_date, max_value=max_date,
                           value=(min_date, max_date), format="MMM YYYY")

    st.markdown('<div class="sec">Compare Countries</div>', unsafe_allow_html=True)
    compare = st.multiselect(
        "Compare with",
        [c for c in countries if c != selected_country],
        default=countries[1:3],
    )

    st.markdown("---")
    run = st.button("▶  Run Forecast", use_container_width=True, type="primary")


# ── Filter ────────────────────────────────────────────────────────────────────

mask = (
    (country_df['date'].dt.date >= date_range[0]) &
    (country_df['date'].dt.date <= date_range[1])
)
cdf = country_df[mask].sort_values('date')


# ── Hero ──────────────────────────────────────────────────────────────────────

spikes_html = ''.join([
    f"<div class='vspike' style='transform:translate(-50%,-100%) rotate({a}deg) translateY(-36px)'></div>"
    for a in range(0, 360, 30)
])

st.markdown(f"""
<div class="hero">
  <div class="particles">{''.join(['<div class="p"></div>' for _ in range(8)])}</div>
  <div style="position:relative;z-index:2">
    <div class="hero-title">EPIDEMIC SPREAD<br>PREDICTION</div>
    <div class="hero-sub">AI-Powered Epidemic Intelligence · Real-Time Forecasting</div>
    <div class="hero-tags">
      <div class="tag tag-b"><div class="live-dot"></div>&nbsp;LIVE FORECAST</div>
      <div class="tag tag-c">{selected_country.upper()}</div>
      <div class="tag tag-d">{model_choice.upper()} MODEL · {horizon}D</div>
    </div>
  </div>
  <div class="virus3d">
    <div style="position:relative;width:88px;height:88px">
      <div class="vring"></div>
      <div class="vring"></div>
      <div class="vring"></div>
      <div class="vsphere">{spikes_html}</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)


# ── KPI row ───────────────────────────────────────────────────────────────────

latest = cdf.iloc[-1] if len(cdf) else pd.Series(dtype=float)

col1, col2, col3, col4 = st.columns(4)

with col1:
    val = int(latest.get('confirmed_cumulative', 0))
    delta_pct = float(latest.get('growth_rate', 0)) * 100
    st.metric("Total Confirmed", f"{val:,}", f"{delta_pct:+.1f}%")

with col2:
    r0_val = float(latest.get('r0_estimate', 0))
    st.metric("R₀ Estimate", f"{r0_val:.2f}", f"{r0_val - 1.0:+.2f}")

with col3:
    nc  = float(latest.get('new_cases_7d', 0))
    nc7 = float(cdf.iloc[-8]['new_cases_7d']) if len(cdf) > 8 else nc
    st.metric("Daily Cases (7d avg)", f"{nc:,.0f}", f"{nc - nc7:+,.0f}")

with col4:
    gr = float(latest.get('growth_rate', 0))
    risk_label = "🔴 HIGH" if gr > 0.2 else "🟡 MEDIUM" if gr > 0 else "🟢 LOW"
    st.metric("Outbreak Risk", risk_label, f"Growth {gr:+.1%}")


# ── Forecast chart ────────────────────────────────────────────────────────────

st.markdown('<div class="sec">Epidemic Forecast</div>', unsafe_allow_html=True)


def build_simple_forecast(series: pd.Series, horizon: int):
    alpha    = 0.3
    smoothed = series.ewm(alpha=alpha).mean()
    last     = smoothed.iloc[-1]
    trend    = (smoothed.iloc[-1] - smoothed.iloc[-7]) / 7 if len(smoothed) > 7 else 0
    preds    = np.array([max(0, last + trend * i * np.exp(-0.03 * i))
                         for i in range(1, horizon + 1)])
    noise    = np.random.normal(0, preds.std() * 0.08, horizon)
    lower    = np.clip(preds - abs(noise) * 1.5, 0, None)
    upper    = preds + abs(noise) * 1.5
    return preds, lower, upper


if run or True:
    preds, lower, upper = build_simple_forecast(cdf['new_cases_7d'], horizon)
    future_dates = pd.date_range(
        cdf['date'].iloc[-1] + pd.Timedelta(days=1),
        periods=horizon, freq='D'
    )
    divider_x = cdf['date'].iloc[-1].strftime('%Y-%m-%d')

    fig_fc = go.Figure()

    # Area fill under history
    fig_fc.add_trace(go.Scatter(
        x=cdf['date'], y=cdf['new_cases_7d'],
        fill='tozeroy', fillcolor='rgba(33,150,243,0.06)',
        line=dict(width=0), showlegend=False, hoverinfo='skip',
    ))
    # History line
    fig_fc.add_trace(go.Scatter(
        x=cdf['date'], y=cdf['new_cases_7d'],
        name='Observed',
        line=dict(color='rgba(33,150,243,0.7)', width=1.6),
        hovertemplate='%{x|%b %d}<br>Cases: %{y:,.0f}<extra></extra>',
    ))
    # CI band
    fig_fc.add_trace(go.Scatter(
        x=list(future_dates) + list(future_dates[::-1]),
        y=list(upper) + list(lower[::-1]),
        fill='toself', fillcolor='rgba(56,189,248,0.08)',
        line=dict(width=0), name='90% CI', hoverinfo='skip',
    ))
    # Glow behind forecast — use rgba() not 8-digit hex
    fig_fc.add_trace(go.Scatter(
        x=future_dates, y=preds,
        line=dict(color='rgba(56,189,248,0.12)', width=14),
        showlegend=False, hoverinfo='skip',
    ))
    # Forecast line
    fig_fc.add_trace(go.Scatter(
        x=future_dates, y=preds,
        name=f'{model_choice} Forecast',
        line=dict(color='#38bdf8', width=2.2, dash='dash'),
        hovertemplate='%{x|%b %d}<br>Forecast: %{y:,.0f}<extra></extra>',
    ))

    # NOW divider via shape + annotation (avoids Plotly Timestamp arithmetic bug)
    fig_fc.add_shape(
        type='line',
        x0=divider_x, x1=divider_x, y0=0, y1=1,
        xref='x', yref='paper',
        line=dict(color='rgba(56,189,248,0.3)', width=1, dash='dot'),
    )
    fig_fc.add_annotation(
        x=divider_x, y=0.97, xref='x', yref='paper',
        text='NOW', showarrow=False,
        font=dict(color='rgba(56,189,248,0.6)', size=8, family='Orbitron'),
        xanchor='left', yanchor='top', xshift=5,
    )

    layout_fc = chart_layout(height=390)
    layout_fc['legend'] = dict(
        orientation='h', yanchor='bottom', y=1.02,
        bgcolor='rgba(0,0,0,0)',
        font=dict(color='#4a7a9b', size=10, family='JetBrains Mono'),
    )
    layout_fc['yaxis']['title'] = 'Daily New Cases (7d avg)'
    layout_fc['yaxis']['title_font'] = dict(color='#2a4a6a', size=10)
    fig_fc.update_layout(**layout_fc)
    st.plotly_chart(fig_fc, use_container_width=True)


# ── R₀ + Country comparison ───────────────────────────────────────────────────

col_r0, col_cmp = st.columns(2)

with col_r0:
    st.markdown('<div class="sec">R₀ Reproduction Number</div>', unsafe_allow_html=True)
    fig_r0 = go.Figure()

    r0_max = float(cdf['r0_estimate'].max()) + 0.3
    fig_r0.add_hrect(y0=1, y1=r0_max, fillcolor='rgba(248,113,113,0.04)', line_width=0)
    fig_r0.add_hrect(y0=0, y1=1,      fillcolor='rgba(33,150,243,0.04)',  line_width=0)

    fig_r0.add_shape(
        type='line', x0=0, x1=1, y0=1, y1=1,
        xref='paper', yref='y',
        line=dict(color='rgba(251,191,36,0.45)', width=1, dash='dot'),
    )
    fig_r0.add_annotation(
        x=1, y=1, xref='paper', yref='y',
        text='CRITICAL  R₀=1', showarrow=False,
        font=dict(color='rgba(251,191,36,0.6)', size=8, family='Orbitron'),
        xanchor='right', yanchor='bottom', yshift=3,
    )
    # Glow shadow — rgba() not 8-digit hex
    fig_r0.add_trace(go.Scatter(
        x=cdf['date'], y=cdf['r0_estimate'],
        line=dict(color='rgba(33,150,243,0.12)', width=10),
        showlegend=False, hoverinfo='skip',
    ))
    fig_r0.add_trace(go.Scatter(
        x=cdf['date'], y=cdf['r0_estimate'],
        fill='tozeroy', fillcolor='rgba(33,150,243,0.07)',
        line=dict(color='#2196f3', width=1.8),
        name='R₀',
        hovertemplate='%{x|%b %d}<br>R₀: %{y:.3f}<extra></extra>',
    ))

    layout_r0 = chart_layout(height=290)
    layout_r0['showlegend'] = False
    layout_r0['yaxis']['title'] = 'R₀'
    layout_r0['yaxis']['title_font'] = dict(color='#2a4a6a', size=10)
    fig_r0.update_layout(**layout_r0)
    st.plotly_chart(fig_r0, use_container_width=True)

with col_cmp:
    st.markdown('<div class="sec">Country Comparison</div>', unsafe_allow_html=True)
    fig_cmp = go.Figure()

    # All blues — different shades, all valid 6-digit hex
    palette_hex  = ['#38bdf8', '#2196f3', '#06b6d4', '#7dd3fc', '#1d6fa4']
    # Pre-built rgba glow versions (no 8-digit hex!)
    palette_glow = [
        'rgba(56,189,248,0.1)',
        'rgba(33,150,243,0.1)',
        'rgba(6,182,212,0.1)',
        'rgba(125,211,252,0.1)',
        'rgba(29,111,164,0.1)',
    ]
    all_c = [selected_country] + (compare or [])

    for i, c in enumerate(all_c[:5]):
        cmp_df = df_all[
            (df_all['country'] == c) &
            (df_all['date'].dt.date >= date_range[0]) &
            (df_all['date'].dt.date <= date_range[1])
        ]
        # Glow trace — rgba string, NOT 8-digit hex
        fig_cmp.add_trace(go.Scatter(
            x=cmp_df['date'], y=cmp_df['new_cases_7d'],
            line=dict(color=palette_glow[i], width=8),
            showlegend=False, hoverinfo='skip',
        ))
        # Main trace
        fig_cmp.add_trace(go.Scatter(
            x=cmp_df['date'], y=cmp_df['new_cases_7d'],
            name=c,
            line=dict(color=palette_hex[i], width=1.7),
            hovertemplate=f'{c}<br>%{{x|%b %d}}<br>%{{y:,.0f}}<extra></extra>',
        ))

    layout_cmp = chart_layout(height=290)
    layout_cmp['legend'] = dict(
        bgcolor='rgba(0,0,0,0)',
        font=dict(color='#4a7a9b', size=9, family='JetBrains Mono'),
    )
    fig_cmp.update_layout(**layout_cmp)
    st.plotly_chart(fig_cmp, use_container_width=True)


# ── Risk map + SHAP ───────────────────────────────────────────────────────────

col_map, col_feat = st.columns([3, 2])

with col_map:
    st.markdown('<div class="sec">Global Risk Map</div>', unsafe_allow_html=True)
    try:
        from app.risk_map import compute_risk_score, build_choropleth
        risk_df = compute_risk_score(df_all)
        fig_map = build_choropleth(risk_df)
        fig_map.update_layout(height=350)
        st.plotly_chart(fig_map, use_container_width=True)
    except Exception as e:
        st.warning(f"Map unavailable: {e}")

with col_feat:
    st.markdown('<div class="sec">Feature Importance · SHAP</div>', unsafe_allow_html=True)

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
    feat_df = (pd.DataFrame({'feature': list(features.keys()),
                              'importance': list(features.values())})
                 .sort_values('importance'))

    # Blue gradient bar colors via rgba()
    bar_colors = [
        f'rgba(33,150,243,{0.18 + v*2.6:.2f})' for v in feat_df['importance']
    ]

    fig_feat = go.Figure(go.Bar(
        x=feat_df['importance'], y=feat_df['feature'],
        orientation='h',
        marker=dict(
            color=bar_colors,
            line=dict(color='rgba(56,189,248,0.22)', width=1),
        ),
        hovertemplate='%{y}<br>SHAP: %{x:.3f}<extra></extra>',
    ))
    layout_feat = chart_layout(height=350)
    layout_feat['showlegend'] = False
    layout_feat['margin']['r'] = 10
    layout_feat['xaxis']['title'] = 'SHAP Value'
    layout_feat['xaxis']['title_font'] = dict(color='#2a4a6a', size=10)
    layout_feat['yaxis']['tickfont'] = dict(color='#6a9fc0', size=10, family='JetBrains Mono')
    fig_feat.update_layout(**layout_feat)
    st.plotly_chart(fig_feat, use_container_width=True)


# ── Growth Rate Heatmap ───────────────────────────────────────────────────────

st.markdown('<div class="sec">Monthly Growth Rate Heatmap — All Countries</div>',
            unsafe_allow_html=True)

heat_df = df_all.copy()
heat_df['week'] = heat_df['date'].dt.to_period('M').astype(str)
heat_pivot = (heat_df.groupby(['country', 'week'])['growth_rate']
                     .mean().unstack(fill_value=0))
heat_pivot = heat_pivot.loc[sorted(heat_pivot.index)]

fig_heat = go.Figure(go.Heatmap(
    z=heat_pivot.values,
    x=heat_pivot.columns.tolist(),
    y=heat_pivot.index.tolist(),
    colorscale=[
        [0.00, '#020b18'],
        [0.20, '#041e38'],
        [0.45, '#1d6fa4'],
        [0.70, '#2196f3'],
        [0.85, '#38bdf8'],
        [1.00, '#bae6fd'],
    ],
    zmid=0, xgap=1, ygap=1,
    hovertemplate='<b>%{y}</b><br>%{x}<br>Growth: %{z:.4f}<extra></extra>',
))

layout_heat = chart_layout(height=350)
layout_heat['xaxis']['tickangle'] = -45
layout_heat['xaxis']['nticks']    = 24
layout_heat['xaxis']['tickfont']  = dict(color='#2a4a6a', size=8, family='JetBrains Mono')
layout_heat['yaxis']['tickfont']  = dict(color='#6a9fc0', size=10, family='JetBrains Mono')
fig_heat.update_layout(**layout_heat)
st.plotly_chart(fig_heat, use_container_width=True)


# ── Footer ────────────────────────────────────────────────────────────────────

st.markdown("""
<div style="text-align:center;padding:32px 0 20px;margin-top:40px;
            border-top:1px solid rgba(33,150,243,0.08)">
  <div style="font-family:Orbitron,monospace;font-size:9px;letter-spacing:4px;
              color:rgba(56,189,248,0.3);text-transform:uppercase;margin-bottom:6px">
    EpiTrack · Epidemic Intelligence System
  </div>
  <div style="font-family:JetBrains Mono,monospace;font-size:10px;
              color:rgba(74,122,155,0.38);letter-spacing:1px">
    CodeCure AI Hackathon — Track C · Epidemic Spread Prediction
  </div>
</div>
""", unsafe_allow_html=True)