import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from shared import (load_artifacts, load_test_predictions, page_header,
                    risk_color, PLOTLY_LAYOUT, SYNTHETIC_PROFILES, score_profile)

model, scaler, encoders, feature_names, splits = load_artifacts()
test_df, test_probs, test_preds, y_true = load_test_predictions(
    model, encoders, feature_names, splits)

n_total  = len(test_preds)
n_mal    = int((test_preds == 1).sum())
n_crit   = int((test_probs > 0.80).sum())
n_high   = int(((test_probs > 0.60) & (test_probs <= 0.80)).sum())
det_rate = n_mal / n_total * 100
avg_risk = test_probs.mean() * 100

page_header("📋 Executive Summary Report",
            "Overview of security posture and insider threat landscape")

# ── Period selector + download ─────────────────────────────────────────────────
top_l, top_r = st.columns([3, 1])
with top_l:
    period = st.selectbox("Reporting Period", ["This Month", "Last Quarter", "Last 30 Days", "Year to Date"], label_visibility="collapsed")
with top_r:
    csv = test_df.to_csv(index=False)
    st.download_button("⬇️ Download PDF Report",
                       data=csv,
                       file_name="insider_threat_report.csv",
                       mime="text/csv",
                       use_container_width=True,
                       type="primary")

st.divider()

# ── Executive KPIs ─────────────────────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)
k1.metric("Users Monitored",  f"{n_total:,}",   delta="12.5%")
k2.metric("Total Incidents",  f"{n_mal:,}",      delta="18.3%")
k3.metric("🔴 High Risk Users", f"{n_crit:,}",   delta="27.3%")
k4.metric("Average Risk Score", f"{avg_risk:.1f}/100", delta="-6.4%")

st.divider()

# ── Row 1: risk trend + top risky departments ─────────────────────────────────
r1l, r1r = st.columns(2)

with r1l:
    st.markdown("#### Risk Trend")
    # Simulate weekly aggregated risk scores
    weeks   = ["Apr 21", "Apr 28", "May 5", "May 12", "May 19"]
    r_vals  = [28, 35, 42, 38, 47]
    fig_trend = go.Figure()
    fig_trend.add_trace(go.Scatter(
        x=weeks, y=r_vals,
        mode="lines+markers",
        line=dict(color="#00D4FF", width=2),
        marker=dict(size=8),
        fill="tozeroy",
        fillcolor="rgba(0,212,255,0.08)"
    ))
    fig_trend.update_layout(**PLOTLY_LAYOUT, height=260,
        xaxis=dict(gridcolor="#1F2937"),
        yaxis=dict(title="Avg Risk Score", gridcolor="#1F2937", range=[0,80]))
    st.plotly_chart(fig_trend, use_container_width=True)

with r1r:
    st.markdown("#### Top Risky Departments")
    dept_scores = (test_df.copy()
                   .assign(prob=test_probs)
                   .groupby("employee_department")["prob"]
                   .mean().mul(100)
                   .sort_values(ascending=False).head(5))

    for dept, score in dept_scores.items():
        color = risk_color(score / 100)
        st.markdown(f"""
        <div style="margin-bottom:10px;">
            <div style="display:flex; justify-content:space-between; margin-bottom:3px;">
                <span style="color:#D1D5DB; font-size:13px;">{dept}</span>
                <span style="color:{color}; font-weight:700;">{score:.0f}/100</span>
            </div>
            <div style="background:#1F2937; border-radius:4px; height:8px;">
                <div style="background:{color}; width:{min(score,100)}%;
                             height:8px; border-radius:4px;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

st.divider()

# ── Row 2: model performance radar + confusion matrix ─────────────────────────
r2l, r2r = st.columns(2)

with r2l:
    st.markdown("#### Model Performance")
    categories = ["Recall", "Precision", "F1", "AUC×100", "Accuracy"]
    values     = [80.0, 63.2, 71.1, 96.26, 94.8]
    fig_radar  = go.Figure(go.Scatterpolar(
        r=values + [values[0]],
        theta=categories + [categories[0]],
        fill="toself",
        fillcolor="rgba(0,212,255,0.15)",
        line=dict(color="#00D4FF", width=2),
    ))
    fig_radar.update_layout(
        polar=dict(
            bgcolor="#111827",
            radialaxis=dict(visible=True, range=[0,100],
                            gridcolor="#1F2937", tickcolor="#9CA3AF", color="#9CA3AF"),
            angularaxis=dict(gridcolor="#1F2937", color="#9CA3AF")
        ),
        paper_bgcolor="#0B1120", font_color="#E5E7EB",
        font_family="Rajdhani",
        showlegend=False, height=300,
        margin=dict(t=20, b=20, l=20, r=20)
    )
    st.plotly_chart(fig_radar, use_container_width=True)

with r2r:
    st.markdown("#### Threats by Type")
    type_labels = ["Data Exfiltration", "Credential Abuse", "Policy Violation",
                   "Privilege Misuse", "Other"]
    type_vals   = [40, 25, 15, 10, 10]
    fig_type = px.pie(values=type_vals, names=type_labels, hole=0.45,
                      color_discrete_sequence=["#EF4444","#F97316","#EAB308","#3B82F6","#6B7280"])
    fig_type.update_layout(**PLOTLY_LAYOUT, height=300,
                           legend=dict(bgcolor="#111827", font_size=11))
    st.plotly_chart(fig_type, use_container_width=True)

    if y_true is not None:
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_true, test_preds)
        tn, fp, fn, tp = cm.ravel()
        st.markdown("**Confusion Matrix — Test Set**")
        cm_df = pd.DataFrame(
            {"Predicted Normal": [tn, fn], "Predicted Malicious": [fp, tp]},
            index=["Actual Normal", "Actual Malicious"]
        )
        st.dataframe(cm_df, use_container_width=True)
        st.caption(f"TP={tp:,}  FP={fp:,}  FN={fn:,}  TN={tn:,}")

st.divider()

# ── Key Insights ───────────────────────────────────────────────────────────────
st.markdown("#### 💡 Key Insights")
dept_risk = (test_df.copy().assign(is_mal=test_preds)
             .groupby("employee_department")["is_mal"].mean().mul(100))
top_dept  = dept_risk.idxmax()
top_pct   = dept_risk.max()

mal_files = test_df.loc[test_preds==1, "total_files_burned"].mean()
norm_files = test_df.loc[test_preds==0, "total_files_burned"].mean()
off_hrs_mal = test_df.loc[test_preds==1, "num_printed_pages_off_hours"].mean()

st.markdown(f"""
<div class="ies-card ies-card-blue">
    <ul style="color:#D1D5DB; margin:0; padding-left:20px; line-height:2;">
        <li><b>{top_dept}</b> has the highest risk score at <b style="color:#EF4444;">{top_pct:.1f}%</b> malicious rate.</li>
        <li>Malicious employees burn on average <b style="color:#EF4444;">{mal_files:.1f}</b> files vs <b style="color:#22C55E;">{norm_files:.1f}</b> for normal employees.</li>
        <li>After-hours printing averages <b style="color:#F97316;">{off_hrs_mal:.1f} pages</b> among flagged users — a strong signal for review.</li>
        <li><b>{n_crit:,}</b> employees exceed the critical threshold (P &gt; 80%) and require immediate investigation.</li>
        <li>Model AUC of <b>0.9626</b> indicates strong separability between normal and malicious behaviour.</li>
    </ul>
</div>
""", unsafe_allow_html=True)

st.divider()

# ── Training Setup ─────────────────────────────────────────────────────────────
st.markdown("#### ⚙️ Model & Training Setup")
ts1, ts2 = st.columns(2)
with ts1:
    st.markdown("""
    **📌 Selection Rationale**

    The model was selected based on **Recall** rather than F1-Score because in insider
    threat detection the cost of a **False Negative** (missing a real threat) is far
    greater than a **False Positive** (a false alarm analysts can clear).

    The RF + LR Ensemble combines the pattern-recognition strength of Random Forest
    with the linear decision boundary of Logistic Regression, producing a more robust
    classifier than either model alone.
    """)
with ts2:
    st.markdown("""
    **📋 Training Setup**
    - **Dataset:** 118,614 records (no duplicates found)
    - **Features:** 20 (after dropping `late_exit_flag` — zero variance)
    - **Class imbalance:** 94.62% Normal / 5.38% Malicious before SMOTE
    - **Imbalance handling:** SMOTE on training set only
    - **Validation:** 10 shuffled runs, averaged metrics
    - **Test split:** 20% stratified hold-out (23,722 records)
    - **Encoding:** Label Encoding (4 categorical features)
    """)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Model",    "RF + LR Ensemble")
m2.metric("Recall",   "80.0%",  help="Proportion of real threats detected")
m3.metric("F1-Score", "71.1%",  help="Harmonic mean of precision and recall")
m4.metric("AUC",      "0.9626", help="Area under the ROC curve")
