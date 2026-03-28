"""
AI_Assistant.py
---------------
EpiBot — Smart Epidemic AI Assistant page.
Part of the EpiTrack multi-page Streamlit app.
"""

import streamlit as st
import pandas as pd
import os
import sys

# ====================== PROJECT ROOT PATH FIX ======================
# This ensures "from src.chatbot..." works reliably
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))

if project_root not in sys.path:
    sys.path.insert(0, project_root)

# ====================== IMPORTS ======================
from src.chatbot.risk_summary    import build_risk_summary, OUTPUT_PATH
from src.chatbot.context_builder import (
    load_summary, get_safe_countries, get_high_risk_countries
)
from src.chatbot.engine import get_engine, generate_report
# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EpiBot — AI Assistant",
    page_icon="🤖",
    layout="wide"
)

st.markdown("""
<style>
  .stApp { background: #050a0f; }
  section[data-testid="stSidebar"] { background: #0a1520 !important; }
  .metric-card {
    background: #0d1b2a;
    border: 1px solid rgba(0,200,150,0.15);
    border-radius: 10px;
    padding: 14px;
    margin: 6px 0;
    color: #d0e8e0;
  }
</style>
""", unsafe_allow_html=True)

# ── Load risk data ────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading risk data...")
def get_risk_df():
    if os.path.exists(OUTPUT_PATH):
        return pd.read_csv(OUTPUT_PATH)
    try:
        return build_risk_summary()
    except FileNotFoundError:
        return None

df = get_risk_df()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🤖 EpiBot Settings")
    st.markdown("---")

    api_key = st.text_input(
        "OpenAI API Key (optional)",
        type="password",
        placeholder="sk-... (leave blank for offline mode)",
        help="Without a key EpiBot uses the built-in offline engine."
    )

    st.markdown("---")
    st.markdown("### 📊 Live Risk Snapshot")

    if df is not None:
        n_high = len(df[df['status'] == 'High Risk'])
        n_mod  = len(df[df['status'] == 'Moderate Risk'])
        n_low  = len(df[df['status'] == 'Low Risk'])
        n_rise = len(df[df['trend']  == 'Rising'])

        st.markdown(f"""
        <div class="metric-card">
          🔴 <b>High Risk:</b> {n_high} countries<br>
          🟡 <b>Moderate:</b>  {n_mod} countries<br>
          🟢 <b>Low Risk:</b>  {n_low} countries<br>
          📈 <b>Rising:</b>    {n_rise} countries
        </div>
        """, unsafe_allow_html=True)

        st.markdown("**⚠️ Top 5 High Risk:**")
        for c in get_high_risk_countries(df, 5):
            st.markdown(f"  🔴 {c}")

        st.markdown("**✅ Top 5 Safe:**")
        for c in get_safe_countries(df, 5):
            st.markdown(f"  🟢 {c}")
    else:
        st.warning("Risk data not found. Run notebook 02 then click Refresh.")

    st.markdown("---")

    if st.button("🔄 Refresh Risk Data", use_container_width=True):
        try:
            with st.spinner("Rebuilding..."):
                build_risk_summary()
                st.cache_data.clear()
            st.success("Refreshed!")
            st.rerun()
        except Exception as e:
            st.error(f"Error: {e}")

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

    st.markdown("---")
    if st.button("📄 Generate Report", use_container_width=True, type="primary"):
        if df is not None:
            with st.spinner("Generating..."):
                engine = get_engine(api_key or None)
                st.session_state.report = generate_report(df=df, engine=engine)
        else:
            st.error("No risk data available.")

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("# 🤖 EpiBot — Smart Epidemic AI Assistant")
st.markdown(
    "Ask me anything about epidemic risk, safe travel, outbreaks, or specific countries."
)

if api_key and api_key.strip():
    st.success("🔗 Connected to GPT-4o-mini")
else:
    st.info(
        "💡 Running offline (rule-based engine). "
        "Add your OpenAI API key in the sidebar for full AI responses."
    )

# ── Report display ────────────────────────────────────────────────────────────
if st.session_state.get('report'):
    with st.expander("📄 Generated Report", expanded=True):
        st.markdown(st.session_state.report)
        st.download_button(
            "⬇️ Download Report (.md)",
            data=st.session_state.report,
            file_name="epitrack_report.md",
            mime="text/markdown"
        )
    if st.button("✖ Close Report"):
        st.session_state.report = None
        st.rerun()
    st.markdown("---")

# ── Suggested queries ─────────────────────────────────────────────────────────
st.markdown("**💬 Quick questions:**")
suggested = [
    "Which countries are safe to travel?",
    "Show global outbreak summary",
    "Which countries are high risk?",
    "What is R0 and why does it matter?",
    "Which countries are improving?",
    "Tell me about India's situation",
]
cols = st.columns(3)
for i, q in enumerate(suggested):
    if cols[i % 3].button(q, key=f"sq_{i}", use_container_width=True):
        st.session_state.pending_query = q

# ── Chat state init ───────────────────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ── Display chat history ──────────────────────────────────────────────────────
for role, msg in st.session_state.chat_history:
    with st.chat_message("user" if role == "user" else "assistant",
                         avatar="🧑" if role == "user" else "🤖"):
        st.markdown(msg)

# ── Chat input ────────────────────────────────────────────────────────────────
user_input = st.chat_input("Ask about epidemic risk, travel safety, outbreaks...")

# Handle suggested query
if "pending_query" in st.session_state:
    user_input = st.session_state.pending_query
    del st.session_state.pending_query

if user_input:
    if df is None:
        st.error(
            "Risk data not available.\n\n"
            "1. Run notebook `02_preprocessing.ipynb` first\n"
            "2. Then click **Refresh Risk Data** in the sidebar"
        )
    else:
        with st.chat_message("user", avatar="🧑"):
            st.markdown(user_input)
        st.session_state.chat_history.append(("user", user_input))

        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Analysing..."):
                try:
                    engine   = get_engine(api_key or None)
                    response = engine.ask(
                        user_input,
                        df=df,
                        chat_history=st.session_state.chat_history[:-1]
                    )
                except Exception as e:
                    response = f"Error: {e}"
            st.markdown(response)

        st.session_state.chat_history.append(("assistant", response))

# ── Welcome message ───────────────────────────────────────────────────────────
if not st.session_state.chat_history:
    with st.chat_message("assistant", avatar="🤖"):
        if df is not None:
            n_high    = len(df[df['status'] == 'High Risk'])
            safe_list = get_safe_countries(df, 3)
            st.markdown(
                f"Hello! I'm **EpiBot** 👋\n\n"
                f"Monitoring **{len(df)} countries**. "
                f"Currently **{n_high} high-risk** countries.\n\n"
                f"Safest destinations: **{', '.join(safe_list)}**\n\n"
                f"Ask me anything about the global epidemic situation!"
            )
        else:
            st.markdown(
                "Hello! I'm **EpiBot** 👋\n\n"
                "I need the risk data to work. "
                "Run notebook 02 then click **Refresh Risk Data** in the sidebar."
            )
