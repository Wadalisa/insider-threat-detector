import streamlit as st
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from shared import GLOBAL_CSS, sidebar_logo

st.set_page_config(
    page_title="InsiderEyeShield",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

# ── Pages ──────────────────────────────────────────────────────────────────────
dashboard    = st.Page("pages/01_dashboard.py",    title="SOC Dashboard")
investigation= st.Page("pages/02_investigation.py",title="Investigation")
analytics    = st.Page("pages/03_analytics.py",    title="Analytics")
live         = st.Page("pages/04_live.py",         title="Live Monitoring")
reports      = st.Page("pages/05_reports.py",      title="Reports")

pg = st.navigation([dashboard, investigation, analytics, live, reports])

# ── Shared sidebar ─────────────────────────────────────────────────────────────
sidebar_logo()

with st.sidebar:
    st.markdown("**⚙️ Model Info**")
    st.markdown("- **Model:** RF + LR Ensemble")
    st.markdown("- **Recall:** 80.0%")
    st.markdown("- **F1:** 71.1%")
    st.markdown("- **AUC:** 0.9626")
    st.caption("⚠️ Optimised for Recall — minimising missed threats.")
    st.divider()
    st.markdown("**🎯 Risk Thresholds**")
    st.markdown("- 🟢 **Low** — P < 30%")
    st.markdown("- 🟡 **Medium** — P 30–60%")
    st.markdown("- 🟠 **High** — P 60–80%")
    st.markdown("- 🔴 **Critical** — P > 80%")

pg.run()
