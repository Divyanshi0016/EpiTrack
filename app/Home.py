"""
Home.py
-------
Entry point for EpiTrack multi-page Streamlit app.
Run with:  streamlit run app/Home.py
"""
import streamlit as st

st.set_page_config(
    page_title="EpiTrack",
    page_icon="🦠",
    layout="wide"
)

st.markdown("""
<style>
  .stApp { background: #050a0f; }
  section[data-testid="stSidebar"] { background: #0a1520 !important; }
  .card {
    background: #0d1b2a;
    border: 1px solid rgba(0,200,150,0.2);
    border-radius: 12px;
    padding: 24px;
    margin: 10px 0;
  }
</style>
""", unsafe_allow_html=True)

st.markdown("# 🦠 EpiTrack")
st.markdown("### Epidemic Spread Prediction System — CodeCure Biohackathon · Track C")
st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="card">
      <h3 style="color:#00c896">📊 Dashboard</h3>
      <p style="color:#7a9e90">
        Forecast charts, R₀ timeline, growth rate analysis,
        SHAP feature importance, and global risk map.
      </p>
      <p style="color:#3a5e50; font-size:12px">
        👈 Click <b>Dashboard</b> in the sidebar
      </p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="card">
      <h3 style="color:#00c896">🤖 AI Assistant</h3>
      <p style="color:#7a9e90">
        Chat with EpiBot — ask about safe travel, high-risk areas,
        country-specific risk, and get AI-generated outbreak reports.
      </p>
      <p style="color:#3a5e50; font-size:12px">
        👈 Click <b>AI Assistant</b> in the sidebar
      </p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#3a5e50; font-size:12px">
  EpiTrack · CodeCure Biohackathon · Track C ·
  Data: Johns Hopkins + OWID + Google Mobility
</div>
""", unsafe_allow_html=True)
