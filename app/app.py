import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
import os
import base64

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="InsiderEyeShield",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.stApp { background-color: #0B1120; }


[data-testid="metric-container"] {
    background-color: #111827;
    border: 1px solid #1F2937;
    padding: 15px;
    border-radius: 15px;
    box-shadow: 0 0 12px rgba(0,212,255,0.08);
}

section[data-testid="stSidebar"] {
    background-color: #07101E;
    border-right: 1px solid #1F2937;
}

.stButton button {
    background: linear-gradient(90deg, #00D4FF, #2563EB);
    color: white;
    border-radius: 10px;
    border: none;
    padding: 12px;
    font-weight: bold;
}

h1, h2, h3 { color: #E5E7EB; }
[data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; }
.stAlert { border-radius: 14px; }

.stTabs [data-baseweb="tab-list"] {
    background-color: #111827;
    border-radius: 12px;
    padding: 4px;
    display: flex;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    color: #9CA3AF;
    font-weight: 800;
    min-height: 75px;
    padding-top: 8px;
    flex: 1;               
    justify-content: center;
}
.stTabs [aria-selected="true"] {
    background-color: #1F2937;
    color: #00D4FF;
}
</style>
""", unsafe_allow_html=True)

# ── Paths ──────────────────────────────────────────────────────────────────────
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


# ── Load model artifacts ───────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    with open(f"{MODELS_DIR}/recall_model.pkl",   "rb") as f: model    = pickle.load(f)
    with open(f"{MODELS_DIR}/scaler.pkl",         "rb") as f: scaler   = pickle.load(f)
    with open(f"{MODELS_DIR}/encoders.pkl",       "rb") as f: encoders = pickle.load(f)
    with open(f"{MODELS_DIR}/feature_names.pkl",  "rb") as f: features = pickle.load(f)
    with open(f"{MODELS_DIR}/splits.pkl",         "rb") as f: splits = pickle.load(f)
    return model, scaler, encoders, features, splits

model, scaler, encoders, feature_names , splits = load_artifacts()
TEST_DATA  = splits['X_test']

dept_options    = list(encoders["employee_department"].classes_)
campus_options  = list(encoders["employee_campus"].classes_)
pos_options     = list(encoders["employee_position"].classes_)
country_options = list(encoders["employee_origin_country"].classes_)

# ── Load & score the real test set ────────────────────────────────────────────
@st.cache_data
def load_test_predictions():
    if isinstance(TEST_DATA, pd.DataFrame):
        raw = TEST_DATA.copy()
    else:
        raw = pd.read_csv(TEST_DATA)

    y_true = None
    if "is_malicious" in raw.columns:
        y_true = raw.pop("is_malicious")

    X = raw[feature_names]

    probs = model.predict_proba(X)[:, 1]
    preds = model.predict(X)

    # Decode categoricals for human-readable display
    display_df = raw.copy()
    for col, le in encoders.items():
        if col in display_df.columns:
            display_df[col] = le.inverse_transform(display_df[col].astype(int))

    display_df["employee_id"]    = [f"u{str(i+10000001).zfill(8)}" for i in range(len(display_df))]
    display_df["Confidence (%)"] = (probs * 100).round(1)
    display_df["Prediction"]     = ["🔴 MALICIOUS" if p == 1 else "🟢 NORMAL" for p in preds]
    display_df["Risk Level"]     = pd.cut(
        probs,
        bins=[0, .3, .6, .8, 1.01],
        labels=["🟢 Low", "🟡 Medium", "🟠 High", "🔴 Critical"]
    )
    if y_true is not None:
        display_df["Ground Truth"] = ["Malicious" if v == 1 else "Normal" for v in y_true]

    return display_df, probs, preds, y_true

test_df, test_probs, test_preds, y_true = load_test_predictions()

# ── Dashboard KPIs (all real) ─────────────────────────────────────────────────
n_total  = len(test_preds)
n_mal    = int((test_preds == 1).sum())
n_crit   = int((test_probs > 0.80).sum())
n_high   = int(((test_probs > 0.60) & (test_probs <= 0.80)).sum())
n_medium = int(((test_probs > 0.30) & (test_probs <= 0.60)).sum())
n_low    = int((test_probs <= 0.30).sum())
det_rate = n_mal / n_total * 100

# ── Helpers ────────────────────────────────────────────────────────────────────
def encode_input(raw):
    df = pd.DataFrame([raw])
    for col, le in encoders.items():
        if col in df.columns:
            df[col] = le.transform(df[col].astype(str))
    return df[feature_names]


def risk_color(prob):
    if prob > 0.80: return "#EF4444"
    if prob > 0.60: return "#F97316"
    if prob > 0.30: return "#EAB308"
    return "#22C55E"


def build_explanation(raw):
    lines = []
    if raw["total_files_burned"] > 5:
        lines.append(f"🔴 **High file burning activity** — {raw['total_files_burned']} files copied to removable media, significantly above the normal range.")
    elif raw["total_files_burned"] > 0:
        lines.append(f"🟡 **File burning detected** — {raw['total_files_burned']} file(s) burned to removable media.")
    if raw["num_printed_pages_off_hours"] > 0:
        lines.append(f"🔴 **Off-hours printing** — {raw['num_printed_pages_off_hours']} pages printed outside business hours.")
    if raw["burned_from_other"] > 0:
        lines.append(f"🔴 **Files burned from other users** — {raw['burned_from_other']} file(s) accessed from other employees and copied.")
    if raw["entry_during_weekend"] == 1:
        lines.append("🟡 **Weekend building access** — physical presence outside normal working days.")
    if raw["num_unique_campus"] > 1:
        lines.append(f"🟡 **Multi-campus access** — {raw['num_unique_campus']} campuses visited, may indicate reconnaissance.")
    if raw["hostility_country_level"] > 0:
        lines.append(f"🟡 **Travel risk flag** — hostility country level {raw['hostility_country_level']}.")
    if raw["is_abroad"] == 1:
        lines.append("🟡 **Currently abroad** — remote access from overseas increases exposure risk.")
    if raw["has_criminal_record"] == 1:
        lines.append("🟠 **Criminal record on file** — pre-employment screening flag active.")
    if not lines:
        lines.append("✅ No individual high-risk indicators flagged. Classification based on combined low-level signal pattern.")
    return lines


def get_feature_importances():
    try:
        rf_component = dict(model.named_estimators_)["rf"]
        importances  = rf_component.feature_importances_
    except Exception:
        importances = getattr(model, "feature_importances_", None)
        if importances is None:
            return None
    return pd.Series(importances, index=feature_names).sort_values(ascending=True).tail(12)


def render_result(pred, prob, raw_input):
    color = "#EF4444" if pred == 1 else "#22C55E"
    label = "🔴  MALICIOUS INSIDER ACTIVITY DETECTED" if pred == 1 else "🟢  NORMAL / BENIGN BEHAVIOUR"
    st.markdown(f"""
    <div style="background:{'#1C0A0A' if pred==1 else '#0A1C0A'};
                border:2px solid {color}; border-radius:14px;
                padding:20px; margin-bottom:16px;">
        <h2 style="color:{color}; margin:0;">{label}</h2>
    </div>
    """, unsafe_allow_html=True)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Classification",         "MALICIOUS" if pred == 1 else "NORMAL")
    k2.metric("Confidence (Malicious)", f"{prob * 100:.1f}%")
    risk_label = ("🔴 Critical" if prob > 0.80 else "🟠 High" if prob > 0.60
                  else "🟡 Medium" if prob > 0.30 else "🟢 Low")
    k3.metric("Risk Level", risk_label)
    k4.metric("Model", "RF + LR Ensemble")

    st.divider()
    left, right = st.columns(2)

    with left:
        st.markdown("#### 📊 Confidence Score")
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=round(prob * 100, 1),
            number={"suffix": "%", "font": {"color": risk_color(prob), "size": 36}},
            gauge={
                "axis":    {"range": [0, 100], "tickcolor": "#9CA3AF"},
                "bar":     {"color": risk_color(prob)},
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
        fig.update_layout(
            paper_bgcolor="#0B1120", font_color="#E5E7EB",
            height=280, margin=dict(t=40, b=0, l=20, r=20)
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.markdown("#### 🔍 Feature Importance")
        fi = get_feature_importances()
        if fi is not None:
            high_risk = {"total_files_burned", "burned_from_other", "num_printed_pages_off_hours"}
            colors_fi = ["#EF4444" if f in high_risk else "#3B82F6" for f in fi.index]
            fig2 = go.Figure(go.Bar(
                x=fi.values, y=fi.index, orientation="h",
                marker_color=colors_fi,
                hovertemplate="%{y}: %{x:.4f}<extra></extra>"
            ))
            fig2.update_layout(
                paper_bgcolor="#0B1120", plot_bgcolor="#111827",
                font_color="#E5E7EB", height=320,
                margin=dict(t=10, b=10, l=10, r=10),
                xaxis=dict(title="Importance Score", gridcolor="#1F2937"),
                yaxis=dict(gridcolor="#1F2937"),
            )
            st.plotly_chart(fig2, use_container_width=True)

    st.divider()

    with st.container():
        st.markdown("#### 🧠 Risk Indicator Analysis")
        st.caption("Plain-language explanation — suitable for non-technical managers:")
        for line in build_explanation(raw_input):
            st.markdown(f"- {line}")

    st.divider()
    st.markdown("#### 📋 Recommended Action")
    if pred == 1 and prob > 0.80:
        st.error("**CRITICAL:** Escalate immediately to the Security Operations Centre. Suspend access and preserve all logs under chain-of-custody procedures.")
    elif pred == 1:
        st.warning("**HIGH RISK:** Flag for urgent security team review. Monitor all subsequent activity and cross-reference physical access logs.")
    elif prob > 0.30:
        st.info("**MEDIUM RISK:** Anomalous signals below the malicious threshold. Schedule routine review and continue passive monitoring.")
    else:
        st.success("**LOW RISK:** Behaviour consistent with normal activity. No immediate action required.")


# ══════════════════════════════════════════════════════════════════════════════
# Sidebar
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    # ── Logo ───────────────────────────────────────────────────────────────────
    _logo_path = os.path.join(os.path.dirname(__file__), "logo.png")
    if os.path.exists(_logo_path):
        st.image(_logo_path, use_container_width=True)
    else:
        st.markdown("""
        <div style="text-align:center; padding:10px 0;">
            <h2 style="color:#00D4FF; margin:0;">🛡️ InsiderEyeShield </h2>
            <p style="color:#9CA3AF; font-size:12px; margin:4px 0;">
            COS720 · University of Pretoria · 2026</p>
        </div>
        """, unsafe_allow_html=True)
    st.divider()

    st.markdown("**🎯 Risk Thresholds**")
    st.markdown("- 🟢 **Low** — P < 30%")
    st.markdown("- 🟡 **Medium** — P 30–60%")
    st.markdown("- 🟠 **High** — P 60–80%")
    st.markdown("- 🔴 **Critical** — P > 80%")

# ══════════════════════════════════════════════════════════════════════════════
# Header banner
# ══════════════════════════════════════════════════════════════════════════════
_banner_path = os.path.join(os.path.dirname(__file__), "banner.png")
if os.path.exists(_banner_path):
        st.image(_banner_path, use_container_width=True)
else:
        st.markdown("""
        <div style="text-align:center; padding:10px 0;">
            <h2 style="color:#00D4FF; margin:0;">🛡️ InsiderEyeShield </h2>
            <p style="color:#9CA3AF; font-size:12px; margin:4px 0;">
            COS720 · University of Pretoria · 2026</p>
        </div>
        """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# Tabs
# ══════════════════════════════════════════════════════════════════════════════
import base64

def load_tab_icon(filename):
    icon_path = os.path.join(os.path.dirname(__file__), filename)
    if os.path.exists(icon_path):
        with open(icon_path, "rb") as f:
            data = base64.b64encode(f.read()).decode()
        ext = filename.split(".")[-1].replace("jpg", "jpeg")
        return f"data:image/{ext};base64,{data}"
    return None

icon_css = ""
tab_files = [
    "tab_dashboard.png",
    "tab_investigation.png",
    "tab_batch.png",
    "tab_model.png",
]

for i, filename in enumerate(tab_files):
    icon_path = os.path.join(os.path.dirname(__file__), filename)
    if os.path.exists(icon_path):
        with open(icon_path, "rb") as f:
            data = base64.b64encode(f.read()).decode()
        icon_css += f"""
        .stTabs [data-baseweb="tab"]:nth-child({i + 1}) {{
            padding-left: 75px;
            background-image: url("data:image/png;base64,{data}");
            background-repeat: no-repeat;
            background-size: 64px 64px;
            background-position: 8px center;
        }}
        """

st.markdown(f"<style>{icon_css}</style>", unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs([
    "  Dashboard",
    "  Investigation",
    "  Batch Analytics",
    "  Model Performance",
])
# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Dashboard (all real data)
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    # ── Real KPIs ──────────────────────────────────────────────────────────────
    t1, t2, t3, t4 = st.columns(4)
    t1.metric("👤 Records Analysed",  f"{n_total:,}")
    t2.metric("🚨 Threats Detected",  f"{n_mal:,}",
              delta=f"{det_rate:.1f}% of total")
    t3.metric("🔴 Critical Alerts",   f"{n_crit:,}")
    t4.metric("🟠 High Risk",          f"{n_high:,}")

    st.divider()
    left_col, right_col = st.columns([1.2, 1])

    # ── Real risk distribution donut ───────────────────────────────────────────
    with left_col:
        st.markdown("### 📊 Risk Distribution — Test Set")
        fig_donut = px.pie(
            values=[n_low, n_medium, n_high, n_crit],
            names=["🟢 Low", "🟡 Medium", "🟠 High", "🔴 Critical"],
            hole=0.55,
            color_discrete_sequence=["#22C55E", "#EAB308", "#F97316", "#EF4444"]
        )
        fig_donut.update_traces(textinfo="percent+value")
        fig_donut.update_layout(
            paper_bgcolor="#0B1120", font_color="#E5E7EB",
            legend=dict(bgcolor="#111827", bordercolor="#1F2937"),
            margin=dict(t=10, b=10, l=10, r=10), height=320
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    # ── Real live event feed — top flagged records ─────────────────────────────
    with right_col:
        st.markdown("### 🚨 Top Flagged Employees")
        top_threats = (test_df[test_df["Prediction"] == "🔴 MALICIOUS"]
                       .nlargest(6, "Confidence (%)"))

        for _, row in top_threats.iterrows():
            lvl   = str(row["Risk Level"])
            conf  = row["Confidence (%)"]
            eid   = row["employee_id"]
            pos   = row["employee_position"]
            dept  = row["employee_department"]
            files = row["total_files_burned"]
            off_p = row["num_printed_pages_off_hours"]

            color = ("#EF4444" if "Critical" in lvl else
                     "#F97316" if "High"     in lvl else
                     "#EAB308" if "Medium"   in lvl else "#22C55E")

            # Build a short reason
            reason = []
            if files > 0:   reason.append(f"{int(files)} files burned")
            if off_p > 0:   reason.append(f"{int(off_p)} off-hrs pages")
            if row["entry_during_weekend"] == 1: reason.append("weekend access")
            reason_str = " · ".join(reason) if reason else "combined signals"

            st.markdown(f"""
            <div style="background:#111827; border-left:4px solid {color};
                        border-radius:8px; padding:10px 14px; margin-bottom:8px;">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <span style="color:{color}; font-weight:bold; font-size:13px;">
                        {lvl} &nbsp; {eid}</span>
                    <span style="color:#9CA3AF; font-size:12px;">{conf:.1f}% confidence</span>
                </div>
                <div style="color:#D1D5DB; font-size:12px; margin-top:3px;">
                    {pos} · {dept}
                </div>
                <div style="color:#6B7280; font-size:11px; margin-top:2px;">
                    ⚠ {reason_str}
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.divider()

    # ── Confidence score distribution across all test records ──────────────────
    st.markdown("### 📈 Confidence Score Distribution — Full Test Set")
    fig_dist = px.histogram(
        x=test_probs, nbins=50,
        labels={"x": "P(Malicious)", "y": "Count"},
        color_discrete_sequence=["#3B82F6"]
    )
    fig_dist.add_vline(x=0.5, line_dash="dash", line_color="#EF4444",
                       annotation_text="Decision boundary (0.5)",
                       annotation_font_color="#EF4444")
    fig_dist.add_vline(x=0.8, line_dash="dot", line_color="#F97316",
                       annotation_text="Critical threshold (0.8)",
                       annotation_font_color="#F97316")
    fig_dist.update_layout(
        paper_bgcolor="#0B1120", plot_bgcolor="#111827",
        font_color="#E5E7EB", height=280,
        xaxis=dict(gridcolor="#1F2937"),
        yaxis=dict(gridcolor="#1F2937"),
        margin=dict(t=20, b=10, l=10, r=10)
    )
    st.plotly_chart(fig_dist, use_container_width=True)

    # ── Malicious rate by department ───────────────────────────────────────────
    st.markdown("### 🏢 Malicious Rate by Department")
    dept_risk = (test_df.copy()
                 .assign(is_mal=test_preds)
                 .groupby("employee_department")["is_mal"]
                 .mean()
                 .mul(100)
                 .reset_index()
                 .rename(columns={"employee_department": "Department",
                                  "is_mal": "Malicious Rate (%)"}))
    dept_risk = dept_risk.sort_values("Malicious Rate (%)", ascending=True)

    fig_dept = px.bar(
        dept_risk, x="Malicious Rate (%)", y="Department",
        orientation="h",
        color="Malicious Rate (%)",
        color_continuous_scale="Reds",
        text=dept_risk["Malicious Rate (%)"].apply(lambda x: f"{x:.1f}%")
    )
    fig_dept.update_layout(
        paper_bgcolor="#0B1120", plot_bgcolor="#111827",
        font_color="#E5E7EB", height=380, coloraxis_showscale=False,
        margin=dict(t=10, b=10, l=10, r=10),
        xaxis=dict(gridcolor="#1F2937"),
        yaxis=dict(gridcolor="#1F2937"),
    )
    st.plotly_chart(fig_dept, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Investigation (Manual Entry)
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### 🕵️ Employee Behaviour Investigation")
    st.caption("Enter a behavioural profile to classify as Normal or Malicious.")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**👤 Employee Profile**")
        department = st.selectbox("Department", dept_options,
                                  index=dept_options.index("Engineering Department")
                                        if "Engineering Department" in dept_options else 0)
        campus     = st.selectbox("Campus", campus_options, index=0)
        position   = st.selectbox("Position", pos_options,
                                  index=pos_options.index("Systems Engineer")
                                        if "Systems Engineer" in pos_options else 0)
        origin     = st.selectbox("Origin Country", country_options,
                                  index=country_options.index("Israel")
                                        if "Israel" in country_options else 0)
        seniority  = st.slider("Seniority (years)", 0, 40, 5)
        classif    = st.selectbox("Privilege Classification", [1, 2, 3, 4], index=1)
        contractor = st.checkbox("Is Contractor")

    with col2:
        st.markdown("**📁 Behavioural Activity**")
        total_printed   = st.number_input("Total Pages Printed",            min_value=0, value=0, step=1)
        off_hrs_printed = st.number_input("Pages Printed Off-Hours",        min_value=0, value=0, step=1)
        files_burned    = st.number_input("Files Burned (Removable Media)", min_value=0, value=0, step=1)
        burned_other    = st.number_input("Files Burned (from Others)",     min_value=0, value=0, step=1)
        num_entries     = st.number_input("Building Entries (count)",        min_value=0, value=1, step=1)
        num_campus      = st.number_input("Unique Campuses Visited",         min_value=1, value=1, step=1)
        weekend_entry   = st.checkbox("Weekend Building Access")

    with col3:
        st.markdown("**⚠️ Risk Flags & Travel**")
        foreign_citizen = st.checkbox("Has Foreign Citizenship")
        criminal_record = st.checkbox("Has Criminal Record")
        medical_history = st.checkbox("Has Medical History on File")
        abroad          = st.checkbox("Currently Abroad")
        trip_day        = st.number_input("Trip Day Number",       min_value=0, value=0, step=1)
        hostility       = st.selectbox("Hostility Country Level", [0, 1, 2, 3, 4], index=0)

    raw_input = {
        "employee_department":         department,
        "employee_campus":             campus,
        "employee_position":           position,
        "employee_seniority_years":    seniority,
        "is_contractor":               int(contractor),
        "employee_classification":     classif,
        "has_foreign_citizenship":     int(foreign_citizen),
        "has_criminal_record":         int(criminal_record),
        "has_medical_history":         int(medical_history),
        "employee_origin_country":     origin,
        "total_printed_pages":         total_printed,
        "num_printed_pages_off_hours": off_hrs_printed,
        "total_files_burned":          files_burned,
        "burned_from_other":           burned_other,
        "is_abroad":                   int(abroad),
        "trip_day_number":             trip_day,
        "hostility_country_level":     hostility,
        "num_entries":                 num_entries,
        "num_unique_campus":           num_campus,
        "entry_during_weekend":        int(weekend_entry),
    }

    if st.button("🔍  Analyse Employee", use_container_width=True, type="primary"):
        X_enc = encode_input(raw_input)
        pred  = int(model.predict(X_enc)[0])
        prob  = float(model.predict_proba(X_enc)[0][1])
        st.divider()
        render_result(pred, prob, raw_input)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Batch Analytics (CSV Upload)
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### 📈 Batch Threat Analysis")
    st.info("Upload a CSV with the same columns as the Kaggle insider threat dataset. "
            "`is_malicious` and `late_exit_flag` are ignored if present.")
    uploaded = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded:
        try:
            raw_df = pd.read_csv(uploaded, sep=None, engine="python")
            st.write(f"Loaded **{len(raw_df):,} records** × {raw_df.shape[1]} columns.")

            with st.expander("Preview raw data"):
                st.dataframe(raw_df.head(5), use_container_width=True)

            for drop_col in ["is_malicious", "late_exit_flag"]:
                if drop_col in raw_df.columns:
                    raw_df = raw_df.drop(columns=[drop_col])

            enc_df = raw_df.copy()
            for col, le in encoders.items():
                if col in enc_df.columns:
                    enc_df[col] = le.transform(enc_df[col].astype(str))

            b_preds = model.predict(enc_df[feature_names])
            b_probs = model.predict_proba(enc_df[feature_names])[:, 1]

            out_df = raw_df.copy()
            out_df["Prediction"]     = ["🔴 MALICIOUS" if p == 1 else "🟢 NORMAL" for p in b_preds]
            out_df["Confidence (%)"] = (b_probs * 100).round(1)
            out_df["Risk Level"]     = pd.cut(b_probs,
                bins=[0, .3, .6, .8, 1.01],
                labels=["🟢 Low", "🟡 Medium", "🟠 High", "🔴 Critical"])

            st.divider()
            bn_mal  = int((b_preds == 1).sum())
            bn_crit = int((b_probs > 0.80).sum())
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Records",        f"{len(b_preds):,}")
            c2.metric("🔴 Malicious Flagged", f"{bn_mal:,}")
            c3.metric("🔴 Critical Risk",      f"{bn_crit:,}")
            c4.metric("Detection Rate",        f"{bn_mal / len(b_preds) * 100:.1f}%")

            lb, rb = st.columns(2)
            with lb:
                fig_bar = px.bar(
                    x=["🟢 Normal", "🔴 Malicious"],
                    y=[(b_preds == 0).sum(), (b_preds == 1).sum()],
                    color=["Normal", "Malicious"],
                    color_discrete_map={"Normal": "#22C55E", "Malicious": "#EF4444"},
                    title="Prediction Distribution"
                )
                fig_bar.update_layout(
                    paper_bgcolor="#0B1120", plot_bgcolor="#111827",
                    font_color="#E5E7EB", showlegend=False,
                    margin=dict(t=40, b=10, l=10, r=10), height=300
                )
                st.plotly_chart(fig_bar, use_container_width=True)

            with rb:
                fig_hist = px.histogram(
                    x=b_probs, nbins=30,
                    title="Confidence Score Distribution",
                    labels={"x": "P(Malicious)"},
                    color_discrete_sequence=["#3B82F6"]
                )
                fig_hist.add_vline(x=0.5, line_dash="dash",
                                   line_color="red", annotation_text="Decision boundary")
                fig_hist.update_layout(
                    paper_bgcolor="#0B1120", plot_bgcolor="#111827",
                    font_color="#E5E7EB",
                    margin=dict(t=40, b=10, l=10, r=10), height=300
                )
                st.plotly_chart(fig_hist, use_container_width=True)

            with st.expander("View Full Results Table"):
                display_cols = ["Prediction", "Confidence (%)", "Risk Level"] + list(raw_df.columns)
                st.dataframe(out_df[display_cols], use_container_width=True)

            st.download_button("⬇️  Download Results CSV",
                               out_df.to_csv(index=False),
                               file_name="threat_predictions.csv", mime="text/csv")

        except Exception as e:
            st.error(f"Error processing file: {e}")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Model Performance
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("### ⚙️ Model Performance Summary")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Model",    "Random Forests","Logistic Regression")
    m2.metric("Recall",   "80.0%",  help="Proportion of real threats detected")
    m3.metric("F1-Score", "71.1%",  help="Harmonic mean of precision and recall")
    m4.metric("AUC",      "96.26%", help="Area under the ROC curve")

    st.divider()
    left_m, right_m = st.columns(2)

    with left_m:
        st.markdown("**📌 Selection Rationale**")
        st.markdown("""
        The model was selected based on **Recall** rather than F1-Score because in insider
        threat detection the cost of a **False Negative** (missing a real threat) is far
        greater than the cost of a **False Positive** (a false alarm analysts can clear).

        The model combines the pattern-recognition strength of Random Forest
        with the linear decision boundary of Logistic Regression, producing a more robust
        classifier than either model alone.
        """)
        st.markdown("**📋 Training Setup**")
        st.markdown("""
        - **Dataset:** 118,614 records (no duplicates found)
        - **Features:** 20 (after dropping `late_exit_flag` — zero variance)
        - **Class imbalance:** 94.62% Normal / 5.38% Malicious before SMOTE
        - **Imbalance handling:** SMOTE on training set only
        - **Validation:** 10 shuffled runs, averaged metrics
        - **Test split:** 20% stratified hold-out (23,722 records)
        - **Encoding:** Label Encoding (4 categorical features)
        """)

        # Real confusion matrix numbers from test set
        if y_true is not None:
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(y_true, test_preds)
            tn, fp, fn, tp = cm.ravel()
            st.markdown("**📊 Confusion Matrix — Test Set**")
            cm_data = pd.DataFrame(
                {"Predicted Normal": [tn, fn],
                 "Predicted Malicious": [fp, tp]},
                index=["Actual Normal", "Actual Malicious"]
            )
            st.dataframe(cm_data, use_container_width=True)
            st.caption(f"TP={tp:,}  FP={fp:,}  FN={fn:,}  TN={tn:,}")

    with right_m:
        categories = ["Recall", "Precision", "F1", "AUC×100", "Accuracy"]
        values     = [80.0, 63.2, 71.1, 96.26, 94.8]
        fig_radar = go.Figure(go.Scatterpolar(
            r=values + [values[0]],
            theta=categories + [categories[0]],
            fill="toself",
            fillcolor="rgba(0,212,255,0.15)",
            line=dict(color="#00D4FF", width=2),
        ))
        fig_radar.update_layout(
            polar=dict(
                bgcolor="#111827",
                radialaxis=dict(visible=True, range=[0, 100],
                                gridcolor="#1F2937", tickcolor="#9CA3AF", color="#9CA3AF"),
                angularaxis=dict(gridcolor="#1F2937", color="#9CA3AF")
            ),
            paper_bgcolor="#0B1120", font_color="#E5E7EB",
            showlegend=False, height=380,
            margin=dict(t=20, b=20, l=20, r=20)
        )
        st.plotly_chart(fig_radar, use_container_width=True)