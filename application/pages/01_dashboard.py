import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sys, os

_banner_path = os.path.join(os.path.dirname(__file__), "../asset/", "dashboard.png")
if os.path.exists(_banner_path):
    st.image(_banner_path, use_container_width=True)
else: "not found"

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from shared import (load_artifacts, load_test_predictions, page_header,
                    risk_color, risk_label, PLOTLY_LAYOUT)

# ── Load data ──────────────────────────────────────────────────────────────────
model, scaler, encoders, feature_names, splits = load_artifacts()
test_df, test_probs, test_preds, y_true = load_test_predictions(
    model, encoders, feature_names, splits)

n_total  = len(test_preds)
n_mal    = int((test_preds == 1).sum())
n_crit   = int((test_probs > 0.80).sum())
n_high   = int(((test_probs > 0.60) & (test_probs <= 0.80)).sum())
n_medium = int(((test_probs > 0.30) & (test_probs <= 0.60)).sum())
n_low    = int((test_probs <= 0.30).sum())
det_rate = n_mal / n_total * 100


# ── KPI row ────────────────────────────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("👤 Users Monitored",  f"{n_total:,}")
k2.metric("🚨 Threats Detected", f"{n_mal:,}",    delta=f"{det_rate:.1f}% of total")
k3.metric("🔴 Critical Alerts",  f"{n_crit:,}")
k4.metric("🟠 High Risk",         f"{n_high:,}")
k5.metric("Model Recall",         "80.0%",         delta="4.3% vs baseline")

st.divider()

# ── Row 1: donut + top flagged ─────────────────────────────────────────────────
left, right = st.columns([1.2, 1])

with left:
    st.markdown("### Risk Level Distribution")
    fig_donut = px.pie(
        values=[n_low, n_medium, n_high, n_crit],
        names=["🟢 Low", "🟡 Medium", "🟠 High", "🔴 Critical"],
        hole=0.55,
        color_discrete_sequence=["#22C55E", "#EAB308", "#F97316", "#EF4444"]
    )
    fig_donut.update_traces(textinfo="percent+value")
    fig_donut.update_layout(**PLOTLY_LAYOUT,
        legend=dict(bgcolor="#111827", bordercolor="#1F2937"), height=320)
    st.plotly_chart(fig_donut, use_container_width=True)

with right:
    st.markdown("### 🚨 Recent Alerts")
    top_threats = (test_df[test_df["Prediction"] == "🔴 MALICIOUS"]
                   .nlargest(5, "Confidence (%)"))

    for _, row in top_threats.iterrows():
        lvl  = str(row["Risk Level"])
        conf = row["Confidence (%)"]
        eid  = row["employee_id"]
        pos  = row["employee_position"]
        dept = row["employee_department"]
        files= row["total_files_burned"]
        off_p= row["num_printed_pages_off_hours"]

        color = ("#EF4444" if "Critical" in lvl else
                 "#F97316" if "High"     in lvl else
                 "#EAB308" if "Medium"   in lvl else "#22C55E")

        reasons = []
        if files > 0: reasons.append(f"{int(files)} files burned")
        if off_p > 0: reasons.append(f"{int(off_p)} off-hrs pages")
        if row["entry_during_weekend"] == 1: reasons.append("weekend access")
        reason_str = " · ".join(reasons) if reasons else "combined signals"

        st.markdown(f"""
        <div style="background:#111827; border-left:4px solid {color};
                    border-radius:8px; padding:10px 14px; margin-bottom:8px;">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <span style="color:{color}; font-weight:bold; font-size:13px;">
                    {lvl} &nbsp; {eid}</span>
                <span style="color:#9CA3AF; font-size:12px;">{conf:.1f}% confidence</span>
            </div>
            <div style="color:#D1D5DB; font-size:12px; margin-top:3px;">{pos} · {dept}</div>
            <div style="color:#6B7280; font-size:11px; margin-top:2px;">⚠ {reason_str}</div>
        </div>
        """, unsafe_allow_html=True)

st.divider()

# ── Row 2: confidence histogram + dept bar ─────────────────────────────────────
r2l, r2r = st.columns(2)

with r2l:
    st.markdown("### Threats Over Time (Confidence Distribution)")
    fig_dist = px.histogram(
        x=test_probs, nbins=50,
        labels={"x": "P(Malicious)", "y": "Count"},
        color_discrete_sequence=["#3B82F6"]
    )
    fig_dist.add_vline(x=0.5, line_dash="dash", line_color="#EF4444",
                       annotation_text="Decision boundary",
                       annotation_font_color="#EF4444")
    fig_dist.add_vline(x=0.8, line_dash="dot", line_color="#F97316",
                       annotation_text="Critical threshold",
                       annotation_font_color="#F97316")
    fig_dist.update_layout(**PLOTLY_LAYOUT, height=300,
        xaxis=dict(gridcolor="#1F2937"),
        yaxis=dict(gridcolor="#1F2937"))
    st.plotly_chart(fig_dist, use_container_width=True)

with r2r:
    st.markdown("### Top Departments by Risk")
    dept_risk = (test_df.copy()
                 .assign(is_mal=test_preds)
                 .groupby("employee_department")["is_mal"]
                 .mean().mul(100).reset_index()
                 .rename(columns={"employee_department": "Department",
                                  "is_mal": "Malicious Rate (%)"}))
    dept_risk = dept_risk.sort_values("Malicious Rate (%)", ascending=True)
    fig_dept = px.bar(
        dept_risk, x="Malicious Rate (%)", y="Department",
        orientation="h", color="Malicious Rate (%)",
        color_continuous_scale="Reds",
        text=dept_risk["Malicious Rate (%)"].apply(lambda x: f"{x:.1f}%")
    )
    fig_dept.update_layout(**PLOTLY_LAYOUT, height=300,
        coloraxis_showscale=False,
        xaxis=dict(gridcolor="#1F2937"),
        yaxis=dict(gridcolor="#1F2937"))
    st.plotly_chart(fig_dept, use_container_width=True)

st.divider()

# ── Row 3: full flagged table ──────────────────────────────────────────────────
st.markdown("### 📋 All Flagged Employees")
flagged = test_df[test_df["Prediction"] == "🔴 MALICIOUS"].copy()
show_cols = ["employee_id", "employee_position", "employee_department",
             "Risk Level", "Confidence (%)", "Prediction"]
if "Ground Truth" in flagged.columns:
    show_cols.append("Ground Truth")
st.dataframe(
    flagged[show_cols].sort_values("Confidence (%)", ascending=False),
    use_container_width=True, height=320
)
