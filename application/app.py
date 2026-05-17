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

# Hide default nav
st.markdown("""
<style>
[data-testid="stSidebarNav"] {
    display: none !important;
}
</style>
""", unsafe_allow_html=True)

#----------navigation bar--------------------------------------------------------------------------
dashboard     = st.Page("pages/01_dashboard.py",    title="SOC Dashboard",  icon="📊")
investigation = st.Page("pages/02_investigation.py",title="Investigation",   icon="🕵️")
analytics     = st.Page("pages/03_analytics.py",    title="Analytics",       icon="📈")
live          = st.Page("pages/04_live.py",         title="Live Monitoring", icon="📡")
reports       = st.Page("pages/05_reports.py",      title="Reports",         icon="📋")

pg = st.navigation([dashboard, investigation, analytics, live, reports], position="hidden")

# ---------- custom nav ----------
def nav_button(label, icon_file, page_key):
    icon_path = os.path.join(os.path.dirname(__file__), "asset", icon_file)
    col1, col2 = st.sidebar.columns([1, 3])
    with col1:
        if os.path.exists(icon_path):
            st.image(icon_path, use_container_width=True)
    with col2:
        return st.button(label, key=page_key, use_container_width=True)

if nav_button("SOC Dashboard",  "tab_dashboard.png",    "nav_dash"):  st.switch_page("pages/01_dashboard.py")
if nav_button("Investigation",  "tab_investigation.png","nav_inv"):   st.switch_page("pages/02_investigation.py")
if nav_button("Analytics",      "tab_batch.png",        "nav_ana"):   st.switch_page("pages/03_analytics.py")
if nav_button("Live Monitoring","tab_monitoring.png",   "nav_live"):  st.switch_page("pages/04_live.py")
if nav_button("Reports",        "tab_report.png",       "nav_rep"):   st.switch_page("pages/05_reports.py")


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
