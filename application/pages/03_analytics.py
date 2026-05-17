import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sys, os

_banner_path = os.path.join(os.path.dirname(__file__), "../asset/", "batch.png")
if os.path.exists(_banner_path):
    st.image(_banner_path, use_container_width=True)
else: "not found"

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from shared import (load_artifacts, load_test_predictions, page_header, PLOTLY_LAYOUT)

model, scaler, encoders, feature_names, splits = load_artifacts()
test_df, test_probs, test_preds, y_true = load_test_predictions(
    model, encoders, feature_names, splits)


# ── Row 1: three summary charts ────────────────────────────────────────────────
r1a, r1b, r1c = st.columns(3)

with r1a:
    st.markdown("#### Threats by Department")
    dept_counts = (test_df.copy()
                   .assign(is_mal=test_preds)
                   .groupby("employee_department")["is_mal"].sum()
                   .reset_index()
                   .rename(columns={"employee_department": "Dept", "is_mal": "Threats"}))
    fig = px.pie(dept_counts, values="Threats", names="Dept", hole=0.45,
                 color_discrete_sequence=px.colors.sequential.Blues_r)
    fig.update_layout(**PLOTLY_LAYOUT, height=280,
                      legend=dict(bgcolor="#111827", font_size=11))
    st.plotly_chart(fig, use_container_width=True)

with r1b:
    st.markdown("#### Confidence Score Distribution")
    fig2 = px.histogram(x=test_probs, nbins=40,
                        color_discrete_sequence=["#3B82F6"],
                        labels={"x": "P(Malicious)"})
    fig2.add_vline(x=0.5, line_dash="dash", line_color="#EF4444")
    fig2.update_layout(**PLOTLY_LAYOUT, height=280,
        xaxis=dict(gridcolor="#1F2937"), yaxis=dict(gridcolor="#1F2937"))
    st.plotly_chart(fig2, use_container_width=True)

with r1c:
    st.markdown("#### Threats by Type (Risk Bucket)")
    risk_counts = pd.Series(
        ["🟢 Low" if p <= 0.30 else "🟡 Medium" if p <= 0.60
         else "🟠 High" if p <= 0.80 else "🔴 Critical"
         for p in test_probs]
    ).value_counts().reset_index()
    risk_counts.columns = ["Risk", "Count"]
    fig3 = px.pie(risk_counts, values="Count", names="Risk", hole=0.45,
                  color_discrete_sequence=["#22C55E","#EAB308","#F97316","#EF4444"])
    fig3.update_layout(**PLOTLY_LAYOUT, height=280,
                       legend=dict(bgcolor="#111827", font_size=11))
    st.plotly_chart(fig3, use_container_width=True)

st.divider()

# ── Row 2: heatmap activity + risk score dist ──────────────────────────────────
r2l, r2r = st.columns(2)

with r2l:
    st.markdown("#### Heatmap: Files Burned × Off-Hours Printing")
    sample = test_df.copy()
    sample["prob"] = test_probs
    sample["files_bin"] = pd.cut(sample["total_files_burned"],
                                  bins=[-1,0,5,15,50,200],
                                  labels=["0","1-5","6-15","16-50","50+"])
    sample["offprint_bin"] = pd.cut(sample["num_printed_pages_off_hours"],
                                     bins=[-1,0,5,20,100,300],
                                     labels=["0","1-5","6-20","21-100","100+"])
    heat = sample.groupby(["files_bin","offprint_bin"])["prob"].mean().reset_index()
    heat_pivot = heat.pivot(index="files_bin", columns="offprint_bin", values="prob").fillna(0)
    fig_heat = go.Figure(go.Heatmap(
        z=heat_pivot.values,
        x=list(heat_pivot.columns),
        y=list(heat_pivot.index),
        colorscale="Reds",
        hoverongaps=False,
        colorbar=dict(title="Avg P(Malicious)", tickcolor="#9CA3AF", title_font_color="#9CA3AF")
    ))
    fig_heat.update_layout(**PLOTLY_LAYOUT, height=300,
        xaxis=dict(title="Off-Hours Pages Printed", gridcolor="#1F2937"),
        yaxis=dict(title="Files Burned", gridcolor="#1F2937"))
    st.plotly_chart(fig_heat, use_container_width=True)

with r2r:
    st.markdown("#### Risk Score Distribution by Department")
    dept_prob = test_df.copy()
    dept_prob["prob"] = test_probs
    fig_box = px.box(
        dept_prob, x="prob", y="employee_department",
        color_discrete_sequence=["#3B82F6"],
        labels={"prob": "P(Malicious)", "employee_department": "Department"}
    )
    fig_box.update_layout(**PLOTLY_LAYOUT, height=300,
        xaxis=dict(gridcolor="#1F2937"), yaxis=dict(gridcolor="#1F2937"))
    st.plotly_chart(fig_box, use_container_width=True)

st.divider()

# ── Row 3: feature importance + campus analysis ────────────────────────────────
r3l, r3r = st.columns(2)

with r3l:
    st.markdown("#### Feature Importance")
    try:
        rf_component = dict(model.named_estimators_)["rf"]
        importances  = rf_component.feature_importances_
    except Exception:
        importances = getattr(model, "feature_importances_", None)

    if importances is not None:
        import pandas as pd
        fi = pd.Series(importances, index=feature_names).sort_values(ascending=True).tail(12)
        high_risk = {"total_files_burned", "burned_from_other", "num_printed_pages_off_hours"}
        colors_fi = ["#EF4444" if f in high_risk else "#3B82F6" for f in fi.index]
        fig_fi = go.Figure(go.Bar(
            x=fi.values, y=fi.index, orientation="h",
            marker_color=colors_fi,
            hovertemplate="%{y}: %{x:.4f}<extra></extra>"
        ))
        fig_fi.update_layout(**PLOTLY_LAYOUT, height=340,
            xaxis=dict(title="Importance Score", gridcolor="#1F2937"),
            yaxis=dict(gridcolor="#1F2937"))
        st.plotly_chart(fig_fi, use_container_width=True)

with r3r:
    st.markdown("#### Malicious Rate by Campus & Position")
    cp = test_df.copy()
    cp["is_mal"] = test_preds
    cp_grp = (cp.groupby(["employee_campus","employee_position"])["is_mal"]
               .mean().mul(100).reset_index()
               .rename(columns={"is_mal": "Malicious Rate (%)"}))
    fig_cp = px.scatter(
        cp_grp,
        x="employee_position", y="Malicious Rate (%)",
        color="employee_campus", size="Malicious Rate (%)",
        color_discrete_sequence=["#00D4FF","#F97316","#22C55E"],
        labels={"employee_position": "Position"}
    )
    fig_cp.update_layout(**PLOTLY_LAYOUT, height=340,
        xaxis=dict(tickangle=-30, gridcolor="#1F2937"),
        yaxis=dict(gridcolor="#1F2937"),
        legend=dict(bgcolor="#111827"))
    st.plotly_chart(fig_cp, use_container_width=True)

st.divider()

# ── Key Insights ───────────────────────────────────────────────────────────────
st.markdown("#### 💡 Key Insights")
dept_risk = (test_df.copy().assign(is_mal=test_preds)
             .groupby("employee_department")["is_mal"].mean().mul(100))
top_dept  = dept_risk.idxmax()
top_pct   = dept_risk.max()

mal_mask  = test_preds == 1
avg_files_mal  = test_df.loc[mal_mask, "total_files_burned"].mean()
avg_files_norm = test_df.loc[~mal_mask, "total_files_burned"].mean()

i1, i2, i3 = st.columns(3)
i1.metric("Highest Risk Department", top_dept, f"{top_pct:.1f}% malicious rate")
i2.metric("Avg Files Burned (Malicious)",  f"{avg_files_mal:.1f}")
i3.metric("Avg Files Burned (Normal)",     f"{avg_files_norm:.1f}")
