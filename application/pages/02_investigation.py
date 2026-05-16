import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from shared import (load_artifacts, SYNTHETIC_PROFILES, page_header,
                    risk_color, risk_label, encode_profile, score_profile,
                    PLOTLY_LAYOUT)

model, scaler, encoders, feature_names, splits = load_artifacts()

dept_options    = list(encoders["employee_department"].classes_)
campus_options  = list(encoders["employee_campus"].classes_)
pos_options     = list(encoders["employee_position"].classes_)
country_options = list(encoders["employee_origin_country"].classes_)

page_header("🕵️ Employee Investigation",
            "Detailed behavioural analysis and risk assessment per employee")

# ── Profile selector ───────────────────────────────────────────────────────────
st.markdown("### Select or Build a Profile")
mode = st.radio("Mode", ["📂 Load Synthetic Profile", "✏️ Manual Entry"],
                horizontal=True)

profile_names = [p["name"] for p in SYNTHETIC_PROFILES]

if mode == "📂 Load Synthetic Profile":
    selected_name = st.selectbox("Choose employee", profile_names)
    raw_input = next(p for p in SYNTHETIC_PROFILES if p["name"] == selected_name)
    pred, prob = score_profile(raw_input, model, encoders, feature_names)

    # ── Profile card ──────────────────────────────────────────────────────────
    color = risk_color(prob)
    rlabel = risk_label(prob)
    st.divider()
    hdr_l, hdr_r = st.columns([2, 1])
    with hdr_l:
        st.markdown(f"""
        <div class="ies-card" style="border-left:4px solid {color};">
            <div style="display:flex; align-items:center; gap:16px;">
                <div style="background:{color}22; border-radius:50%; width:56px; height:56px;
                            display:flex; align-items:center; justify-content:center;
                            font-size:28px;">👤</div>
                <div>
                    <h2 style="margin:0; color:white;">{raw_input['name']}</h2>
                    <p style="margin:0; color:#9CA3AF;">
                        {raw_input['employee_position']} &nbsp;·&nbsp;
                        {raw_input['employee_id']} &nbsp;·&nbsp;
                        Dept: {raw_input['employee_department']}
                    </p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with hdr_r:
        st.markdown(f"""
        <div class="ies-card" style="text-align:center; border-left:4px solid {color};">
            <p style="color:#9CA3AF; margin:0; font-size:13px;">Risk Score</p>
            <h1 style="color:{color}; margin:4px 0; font-size:3rem;">{prob*100:.0f}<span style="font-size:1.2rem;">/100</span></h1>
            <span style="background:{color}33; color:{color}; padding:3px 14px;
                         border-radius:999px; font-weight:700; font-size:13px;">{rlabel}</span>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # ── Sub-tabs ──────────────────────────────────────────────────────────────
    t1, t2, t3, t4 = st.tabs(["Overview", "Behavioural Analysis",
                               "Risk Factors", "Recommendations"])

    with t1:
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Seniority",       f"{raw_input['employee_seniority_years']} yrs")
        c2.metric("Files Burned",    raw_input['total_files_burned'])
        c3.metric("Off-hrs Printed", raw_input['num_printed_pages_off_hours'])
        c4.metric("Campus Entries",  raw_input['num_entries'])
        c5.metric("Campuses",        raw_input['num_unique_campus'])

        flags = []
        if raw_input["has_criminal_record"]:   flags.append("🔴 Criminal Record")
        if raw_input["has_foreign_citizenship"]:flags.append("🟡 Foreign Citizen")
        if raw_input["is_contractor"]:         flags.append("🟡 Contractor")
        if raw_input["is_abroad"]:             flags.append("🟠 Currently Abroad")
        if raw_input["entry_during_weekend"]:  flags.append("🟡 Weekend Access")
        if raw_input["hostility_country_level"] > 0:
            flags.append(f"🟠 Hostility Level {raw_input['hostility_country_level']}")
        if flags:
            st.markdown("**Active Flags:** " + " &nbsp; ".join(flags),
                        unsafe_allow_html=True)

    with t2:
        b1, b2 = st.columns(2)
        with b1:
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=round(prob * 100, 1),
                number={"suffix": "%", "font": {"color": color, "size": 36}},
                gauge={
                    "axis":    {"range": [0, 100], "tickcolor": "#9CA3AF"},
                    "bar":     {"color": color},
                    "bgcolor": "#111827",
                    "steps": [
                        {"range": [0,  30],  "color": "#064E3B"},
                        {"range": [30, 60],  "color": "#713F12"},
                        {"range": [60, 80],  "color": "#7C2D12"},
                        {"range": [80, 100], "color": "#450A0A"},
                    ],
                    "threshold": {"line": {"color": "white", "width": 3},
                                  "thickness": 0.75, "value": 50}
                },
                title={"text": "P(Malicious)", "font": {"color": "#9CA3AF"}}
            ))
            fig_gauge.update_layout(**PLOTLY_LAYOUT, height=280)
            st.plotly_chart(fig_gauge, use_container_width=True)

        with b2:
            activity_labels = ["Files Burned", "Off-hrs Print", "Entries", "Campuses", "Trip Days"]
            activity_vals   = [
                min(raw_input["total_files_burned"] / 50 * 100, 100),
                min(raw_input["num_printed_pages_off_hours"] / 100 * 100, 100),
                min(raw_input["num_entries"] / 4 * 100, 100),
                min(raw_input["num_unique_campus"] / 3 * 100, 100),
                min(raw_input["trip_day_number"] / 14 * 100, 100),
            ]
            fig_bar = go.Figure(go.Bar(
                x=activity_vals, y=activity_labels, orientation="h",
                marker_color=[color if v > 60 else "#3B82F6" for v in activity_vals],
                text=[f"{v:.0f}%" for v in activity_vals],
                textposition="outside"
            ))
            fig_bar.update_layout(**PLOTLY_LAYOUT, height=280,
                xaxis=dict(range=[0, 120], gridcolor="#1F2937"),
                yaxis=dict(gridcolor="#1F2937"),
                title="Activity Levels (normalised %)")
            st.plotly_chart(fig_bar, use_container_width=True)

    with t3:
        risk_factors = {
            "Files Burned (Removable Media)": min(raw_input["total_files_burned"] / 50, 1.0),
            "Off-Hours Printing":             min(raw_input["num_printed_pages_off_hours"] / 100, 1.0),
            "Files from Other Users":         float(raw_input["burned_from_other"]),
            "Weekend Building Access":        float(raw_input["entry_during_weekend"]),
            "Multi-Campus Activity":          min((raw_input["num_unique_campus"] - 1) / 2, 1.0),
            "Travel Risk (Hostility)":        raw_input["hostility_country_level"] / 3,
            "Currently Abroad":               float(raw_input["is_abroad"]),
            "Criminal Record":                float(raw_input["has_criminal_record"]),
        }
        for factor, score in sorted(risk_factors.items(), key=lambda x: -x[1]):
            pct = score * 100
            fc  = "#EF4444" if pct > 70 else "#F97316" if pct > 40 else "#EAB308" if pct > 10 else "#22C55E"
            st.markdown(f"""
            <div style="margin-bottom:10px;">
                <div style="display:flex; justify-content:space-between; margin-bottom:3px;">
                    <span style="color:#D1D5DB; font-size:13px;">{factor}</span>
                    <span style="color:{fc}; font-weight:700;">{pct:.0f}%</span>
                </div>
                <div style="background:#1F2937; border-radius:4px; height:8px;">
                    <div style="background:{fc}; width:{pct}%; height:8px; border-radius:4px;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    with t4:
        if pred == 1 and prob > 0.80:
            st.error("🔴 **CRITICAL:** Escalate immediately to the Security Operations Centre. Suspend access and preserve all logs under chain-of-custody procedures.")
            st.markdown("""
            - ☐ Lock user account immediately
            - ☐ Preserve all access and file logs
            - ☐ Notify SOC and legal team
            - ☐ Conduct forensic review of removable media activity
            - ☐ Interview employee with HR present
            """)
        elif pred == 1:
            st.warning("🟠 **HIGH RISK:** Flag for urgent security team review.")
            st.markdown("""
            - ☐ Flag for security team review
            - ☐ Monitor all subsequent activity
            - ☐ Cross-reference physical access logs
            - ☐ Review data transfer logs
            """)
        elif prob > 0.30:
            st.info("🟡 **MEDIUM RISK:** Schedule routine review and continue passive monitoring.")
            st.markdown("""
            - ☐ Add to enhanced monitoring list
            - ☐ Schedule quarterly review
            - ☐ No immediate action required
            """)
        else:
            st.success("🟢 **LOW RISK:** Behaviour consistent with normal activity. No immediate action required.")

else:
    # ── Manual Entry form ──────────────────────────────────────────────────────
    st.divider()
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**👤 Employee Profile**")
        department = st.selectbox("Department", dept_options)
        campus     = st.selectbox("Campus", campus_options)
        position   = st.selectbox("Position", pos_options)
        origin     = st.selectbox("Origin Country", country_options)
        seniority  = st.slider("Seniority (years)", 0, 40, 5)
        classif    = st.selectbox("Privilege Classification", [1, 2, 3, 4])
        contractor = st.checkbox("Is Contractor")

    with col2:
        st.markdown("**📁 Behavioural Activity**")
        total_printed   = st.number_input("Total Pages Printed",            min_value=0, value=0)
        off_hrs_printed = st.number_input("Pages Printed Off-Hours",        min_value=0, value=0)
        files_burned    = st.number_input("Files Burned (Removable Media)", min_value=0, value=0)
        burned_other    = st.number_input("Files Burned (from Others)",     min_value=0, value=0)
        num_entries     = st.number_input("Building Entries",                min_value=0, value=1)
        num_campus      = st.number_input("Unique Campuses Visited",         min_value=1, value=1)
        weekend_entry   = st.checkbox("Weekend Building Access")

    with col3:
        st.markdown("**⚠️ Risk Flags & Travel**")
        foreign_citizen = st.checkbox("Has Foreign Citizenship")
        criminal_record = st.checkbox("Has Criminal Record")
        medical_history = st.checkbox("Has Medical History on File")
        abroad          = st.checkbox("Currently Abroad")
        trip_day        = st.number_input("Trip Day Number", min_value=0, value=0)
        hostility       = st.selectbox("Hostility Country Level", [0, 1, 2, 3])

    raw_input = {
        "name": "Manual Entry", "employee_id": "N/A",
        "employee_department": department, "employee_campus": campus,
        "employee_position": position, "employee_origin_country": origin,
        "employee_seniority_years": seniority, "is_contractor": int(contractor),
        "employee_classification": classif,
        "has_foreign_citizenship": int(foreign_citizen),
        "has_criminal_record": int(criminal_record),
        "has_medical_history": int(medical_history),
        "total_printed_pages": total_printed,
        "num_printed_pages_off_hours": off_hrs_printed,
        "total_files_burned": files_burned, "burned_from_other": burned_other,
        "is_abroad": int(abroad), "trip_day_number": trip_day,
        "hostility_country_level": hostility,
        "num_entries": num_entries, "num_unique_campus": num_campus,
        "entry_during_weekend": int(weekend_entry),
    }

    if st.button("🔍 Analyse Employee", use_container_width=True, type="primary"):
        pred, prob = score_profile(raw_input, model, encoders, feature_names)
        color  = risk_color(prob)
        rlabel = risk_label(prob)
        label  = "🔴 MALICIOUS INSIDER ACTIVITY DETECTED" if pred == 1 else "🟢 NORMAL / BENIGN BEHAVIOUR"

        st.divider()
        st.markdown(f"""
        <div style="background:{'#1C0A0A' if pred==1 else '#0A1C0A'};
                    border:2px solid {color}; border-radius:14px;
                    padding:20px; margin-bottom:16px;">
            <h2 style="color:{color}; margin:0;">{label}</h2>
        </div>
        """, unsafe_allow_html=True)

        m1, m2, m3 = st.columns(3)
        m1.metric("Classification",  "MALICIOUS" if pred == 1 else "NORMAL")
        m2.metric("Confidence",      f"{prob*100:.1f}%")
        m3.metric("Risk Level",      rlabel)
