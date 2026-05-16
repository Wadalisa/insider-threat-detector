import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import time, random, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from shared import (load_artifacts, load_test_predictions, page_header,
                    risk_color, risk_label, PLOTLY_LAYOUT, SYNTHETIC_PROFILES,
                    score_profile)

model, scaler, encoders, feature_names, splits = load_artifacts()
test_df, test_probs, test_preds, y_true = load_test_predictions(
    model, encoders, feature_names, splits)

page_header("📡 Live Monitoring",
            "Real-time stream of user activities and security events")

# ── Live indicator + controls ──────────────────────────────────────────────────
hdr_l, hdr_r = st.columns([3, 1])
with hdr_l:
    st.markdown("""
    <div style="display:inline-flex; align-items:center; gap:10px;
                background:#111827; border:1px solid #1F2937; border-radius:10px;
                padding:8px 18px;">
        <div style="width:10px; height:10px; border-radius:50%; background:#22C55E;
                    box-shadow:0 0 8px #22C55E; animation:pulse 1.5s infinite;"></div>
        <span style="color:#22C55E; font-weight:700; letter-spacing:1px;">● LIVE</span>
    </div>
    <style>@keyframes pulse{0%,100%{opacity:1;}50%{opacity:0.4;}}</style>
    """, unsafe_allow_html=True)
with hdr_r:
    auto_refresh = st.toggle("Auto-refresh (5s)", value=False)

st.divider()

# ── KPI strip ──────────────────────────────────────────────────────────────────
# Simulate slight variation on each refresh
rng = np.random.default_rng(int(time.time()) // 5)
active_sessions = 342 + int(rng.integers(-15, 15))
events_per_min  = 24  + int(rng.integers(-4, 4))
live_alerts     = int((test_probs > 0.5).sum() % 12) + 3
high_risk_live  = int((test_probs > 0.80).sum() % 5) + 1

k1, k2, k3, k4 = st.columns(4)
k1.metric("Active Sessions",    f"{active_sessions:,}",  delta="8.2%")
k2.metric("Events Per Minute",  f"{events_per_min}",     delta="15.4%")
k3.metric("🚨 Alerts",          f"{live_alerts}",        delta=f"16.7%")
k4.metric("🔴 High Risk Events", f"{high_risk_live}",    delta="5.0%")

st.divider()

# ── Live event feed ────────────────────────────────────────────────────────────
feed_col, spark_col = st.columns([1.6, 1])

with feed_col:
    st.markdown("### 🔴 Live Event Feed")

    # Build events from real high-confidence predictions + synthetic profiles
    events = []
    top_mal = (test_df[test_df["Prediction"] == "🔴 MALICIOUS"]
               .nlargest(4, "Confidence (%)"))
    timestamps = ["10:24:31", "10:24:10", "09:23:47", "09:23:21",
                  "10:22:58", "10:22:35", "10:22:11"]
    event_templates = [
        ("downloaded {files} files to removable media",       "High Risk"),
        ("printed {pages} pages outside business hours",      "High Risk"),
        ("accessed files from another user's directory",      "Medium Risk"),
        ("multi-campus badge access detected",                "Medium Risk"),
        ("login from unusual location — abroad",              "High Risk"),
        ("privilege escalation attempt detected",             "High Risk"),
        ("large data transfer to external device",            "Medium Risk"),
    ]

    for i, (_, row) in enumerate(top_mal.iterrows()):
        tmpl, risk = event_templates[i % len(event_templates)]
        msg = tmpl.format(
            files=int(row["total_files_burned"]) if row["total_files_burned"] > 0 else 3,
            pages=int(row["num_printed_pages_off_hours"]) if row["num_printed_pages_off_hours"] > 0 else 12
        )
        events.append({
            "time": timestamps[i],
            "user": row["employee_id"],
            "event": f"User {row['employee_id']} {msg}",
            "risk": risk,
            "conf": row["Confidence (%)"],
        })

    # Add synthetic profile events
    for p in SYNTHETIC_PROFILES[:3]:
        pred, prob = score_profile(p, model, encoders, feature_names)
        if pred == 1:
            events.append({
                "time": timestamps[len(events) % len(timestamps)],
                "user": p["employee_id"],
                "event": f"User {p['name']} ({p['employee_id']}) flagged by model — {p['employee_department']}",
                "risk": risk_label(prob).replace("🔴 ","").replace("🟠 ","").replace("🟡 ","").replace("🟢 ","") + " Risk",
                "conf": round(prob * 100, 1),
            })

    for ev in events[:7]:
        rc = "#EF4444" if "High" in ev["risk"] else "#EAB308" if "Medium" in ev["risk"] else "#22C55E"
        st.markdown(f"""
        <div style="background:#111827; border-left:4px solid {rc};
                    border-radius:8px; padding:10px 16px; margin-bottom:8px;
                    display:flex; justify-content:space-between; align-items:center;">
            <div>
                <span style="color:#9CA3AF; font-size:11px; margin-right:12px;">{ev['time']}</span>
                <span style="color:#D1D5DB; font-size:13px;">{ev['event']}</span>
            </div>
            <span style="background:{rc}22; color:{rc}; padding:3px 12px;
                         border-radius:999px; font-size:12px; font-weight:700;
                         white-space:nowrap; margin-left:12px;">{ev['risk']}</span>
        </div>
        """, unsafe_allow_html=True)

    if st.button("View All Events →", use_container_width=False):
        st.info("Showing all flagged events from the test set below.")
        show = test_df[test_df["Prediction"] == "🔴 MALICIOUS"][
            ["employee_id","employee_department","employee_position",
             "Confidence (%)","Risk Level"]].head(30)
        st.dataframe(show, use_container_width=True)

with spark_col:
    st.markdown("### 📊 Real-time Risk Trend")

    # Simulate a rolling window of risk scores
    np.random.seed(int(time.time()) // 5)
    t_vals   = list(range(20))
    base     = test_probs[:20] * 100
    live_var = base + np.random.normal(0, 3, 20)
    live_var = np.clip(live_var, 0, 100)

    fig_spark = go.Figure()
    fig_spark.add_trace(go.Scatter(
        x=t_vals, y=live_var,
        mode="lines+markers",
        line=dict(color="#00D4FF", width=2),
        marker=dict(size=5, color="#00D4FF"),
        fill="tozeroy",
        fillcolor="rgba(0,212,255,0.08)",
        name="Risk Score"
    ))
    fig_spark.add_hline(y=80, line_dash="dot", line_color="#EF4444",
                        annotation_text="Critical", annotation_font_color="#EF4444")
    fig_spark.add_hline(y=50, line_dash="dot", line_color="#EAB308",
                        annotation_text="Threshold", annotation_font_color="#EAB308")
    fig_spark.update_layout(**PLOTLY_LAYOUT, height=200,
        xaxis=dict(title="Recent events", gridcolor="#1F2937", showticklabels=False),
        yaxis=dict(title="Risk %", gridcolor="#1F2937", range=[0,110]),
        showlegend=False)
    st.plotly_chart(fig_spark, use_container_width=True)

    st.markdown("### 📋 Session Summary")
    s1, s2 = st.columns(2)
    s1.metric("Normal Sessions",    f"{int((test_preds==0).sum()):,}")
    s2.metric("Flagged Sessions",   f"{int((test_preds==1).sum()):,}")
    s1.metric("Avg Confidence",     f"{test_probs.mean()*100:.1f}%")
    s2.metric("Max Confidence",     f"{test_probs.max()*100:.1f}%")

    st.markdown("### 🏢 Active Alerts by Dept")
    dept_alerts = (test_df.copy()
                   .assign(is_mal=test_preds)
                   .groupby("employee_department")["is_mal"].sum()
                   .sort_values(ascending=False).head(5))
    for dept, cnt in dept_alerts.items():
        st.markdown(f"""
        <div style="display:flex; justify-content:space-between;
                    padding:6px 0; border-bottom:1px solid #1F2937;">
            <span style="color:#D1D5DB; font-size:13px;">{dept}</span>
            <span style="color:#EF4444; font-weight:700;">{int(cnt)}</span>
        </div>
        """, unsafe_allow_html=True)

# ── Auto-refresh ───────────────────────────────────────────────────────────────
if auto_refresh:
    time.sleep(5)
    st.rerun()
