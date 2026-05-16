import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import base64

# ── Paths ──────────────────────────────────────────────────────────────────────
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
APP_DIR    = os.path.dirname(__file__)

# ── CSS ────────────────────────────────────────────────────────────────────────
GLOBAL_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;600;700&display=swap');

* { 
    font-family: 'Rajdhani', sans-serif !important; 
    letter-spacing: 0.5px;
    color: white;
}

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
    font-weight: 600;
    flex: 1;
    justify-content: center;
    min-height: 48px;
}
.stTabs [aria-selected="true"] {
    background-color: #1F2937;
    color: #00D4FF;
}

/* card style */
.ies-card {
    background: #111827;
    border: 1px solid #1F2937;
    border-radius: 14px;
    padding: 18px 20px;
    margin-bottom: 12px;
}
.ies-card-red   { border-left: 4px solid #EF4444; }
.ies-card-orange{ border-left: 4px solid #F97316; }
.ies-card-yellow{ border-left: 4px solid #EAB308; }
.ies-card-green { border-left: 4px solid #22C55E; }
.ies-card-blue  { border-left: 4px solid #3B82F6; }

/* nav pill badge */
.risk-badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 999px;
    font-size: 12px;
    font-weight: 700;
}
</style>
"""

PLOTLY_LAYOUT = dict(
    paper_bgcolor="#0B1120",
    plot_bgcolor="#111827",
    font_color="#E5E7EB",
    font_family="Rajdhani",
    margin=dict(t=30, b=10, l=10, r=10),
)

# ── Artifact loader ────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    with open(f"{MODELS_DIR}/recall_model.pkl",  "rb") as f: model    = pickle.load(f)
    with open(f"{MODELS_DIR}/scaler.pkl",        "rb") as f: scaler   = pickle.load(f)
    with open(f"{MODELS_DIR}/encoders.pkl",      "rb") as f: encoders = pickle.load(f)
    with open(f"{MODELS_DIR}/feature_names.pkl", "rb") as f: features = pickle.load(f)
    with open(f"{MODELS_DIR}/splits.pkl",        "rb") as f: splits   = pickle.load(f)
    return model, scaler, encoders, features, splits

@st.cache_data
def load_test_predictions(_model, _encoders, feature_names, _splits):
    TEST_DATA = _splits["X_test"]
    raw = TEST_DATA.copy() if isinstance(TEST_DATA, pd.DataFrame) else pd.read_csv(TEST_DATA)

    y_true = None
    y_test = _splits.get("y_test")
    if y_test is not None:
        y_true = y_test.reset_index(drop=True)

    X = raw[feature_names]
    probs = _model.predict_proba(X)[:, 1]
    preds = _model.predict(X)

    display_df = raw.copy().reset_index(drop=True)
    for col, le in _encoders.items():
        if col in display_df.columns:
            display_df[col] = le.inverse_transform(display_df[col].astype(int))

    display_df["employee_id"]    = [f"EMP{str(i+10001).zfill(6)}" for i in range(len(display_df))]
    display_df["Confidence (%)"] = (probs * 100).round(1)
    display_df["Prediction"]     = ["🔴 MALICIOUS" if p == 1 else "🟢 NORMAL" for p in preds]
    display_df["Risk Level"]     = pd.cut(
        probs, bins=[0, .3, .6, .8, 1.01],
        labels=["🟢 Low", "🟡 Medium", "🟠 High", "🔴 Critical"]
    )
    if y_true is not None:
        display_df["Ground Truth"] = ["Malicious" if v == 1 else "Normal" for v in y_true]

    return display_df, probs, preds, y_true

# ── Synthetic employee profiles ────────────────────────────────────────────────
SYNTHETIC_PROFILES = [
    {
        "name": "John Doe",         "employee_id": "EMP001024",
        "employee_department":      "Information Technology",
        "employee_campus":          "Campus A",
        "employee_position":        "Systems Engineer",
        "employee_seniority_years": 7,
        "is_contractor":            0,
        "employee_classification":  2,
        "has_foreign_citizenship":  0,
        "has_criminal_record":      0,
        "has_medical_history":      0,
        "employee_origin_country":  "India",
        "total_printed_pages":      12,
        "num_printed_pages_off_hours": 42,
        "total_files_burned":       18,
        "burned_from_other":        1,
        "is_abroad":                0,
        "trip_day_number":          0,
        "hostility_country_level":  0,
        "num_entries":              2,
        "num_unique_campus":        3,
        "entry_during_weekend":     1,
    },
    {
        "name": "Sarah Lee",        "employee_id": "EMP002317",
        "employee_department":      "Finance",
        "employee_campus":          "Campus B",
        "employee_position":        "Accountant",
        "employee_seniority_years": 3,
        "is_contractor":            0,
        "employee_classification":  1,
        "has_foreign_citizenship":  0,
        "has_criminal_record":      0,
        "has_medical_history":      1,
        "employee_origin_country":  "Germany",
        "total_printed_pages":      5,
        "num_printed_pages_off_hours": 0,
        "total_files_burned":       0,
        "burned_from_other":        0,
        "is_abroad":                0,
        "trip_day_number":          0,
        "hostility_country_level":  0,
        "num_entries":              1,
        "num_unique_campus":        1,
        "entry_during_weekend":     0,
    },
    {
        "name": "Mike Smith",       "employee_id": "EMP003881",
        "employee_department":      "R&D Department",
        "employee_campus":          "Campus C",
        "employee_position":        "Head of R&D",
        "employee_seniority_years": 14,
        "is_contractor":            0,
        "employee_classification":  4,
        "has_foreign_citizenship":  1,
        "has_criminal_record":      1,
        "has_medical_history":      0,
        "employee_origin_country":  "Russia",
        "total_printed_pages":      87,
        "num_printed_pages_off_hours": 63,
        "total_files_burned":       45,
        "burned_from_other":        1,
        "is_abroad":                1,
        "trip_day_number":          6,
        "hostility_country_level":  3,
        "num_entries":              3,
        "num_unique_campus":        3,
        "entry_during_weekend":     1,
    },
    {
        "name": "Aisha Patel",      "employee_id": "EMP004552",
        "employee_department":      "Human Resources",
        "employee_campus":          "Campus A",
        "employee_position":        "Training Coordinator",
        "employee_seniority_years": 5,
        "is_contractor":            1,
        "employee_classification":  1,
        "has_foreign_citizenship":  1,
        "has_criminal_record":      0,
        "has_medical_history":      0,
        "employee_origin_country":  "India",
        "total_printed_pages":      3,
        "num_printed_pages_off_hours": 0,
        "total_files_burned":       0,
        "burned_from_other":        0,
        "is_abroad":                0,
        "trip_day_number":          0,
        "hostility_country_level":  0,
        "num_entries":              1,
        "num_unique_campus":        1,
        "entry_during_weekend":     0,
    },
    {
        "name": "Omar Hassan",      "employee_id": "EMP005739",
        "employee_department":      "Security and Information Security",
        "employee_campus":          "Campus B",
        "employee_position":        "Design Engineer",
        "employee_seniority_years": 9,
        "is_contractor":            0,
        "employee_classification":  3,
        "has_foreign_citizenship":  1,
        "has_criminal_record":      0,
        "has_medical_history":      0,
        "employee_origin_country":  "Sudan",
        "total_printed_pages":      22,
        "num_printed_pages_off_hours": 18,
        "total_files_burned":       9,
        "burned_from_other":        0,
        "is_abroad":                1,
        "trip_day_number":          3,
        "hostility_country_level":  2,
        "num_entries":              2,
        "num_unique_campus":        2,
        "entry_during_weekend":     0,
    },
    {
        "name": "Elena Volkov",     "employee_id": "EMP006104",
        "employee_department":      "Engineering Department",
        "employee_campus":          "Campus A",
        "employee_position":        "Development Engineer (Hardware / Software / Mechanical)",
        "employee_seniority_years": 2,
        "is_contractor":            1,
        "employee_classification":  2,
        "has_foreign_citizenship":  1,
        "has_criminal_record":      0,
        "has_medical_history":      0,
        "employee_origin_country":  "Ukraine",
        "total_printed_pages":      0,
        "num_printed_pages_off_hours": 0,
        "total_files_burned":       1,
        "burned_from_other":        0,
        "is_abroad":                0,
        "trip_day_number":          0,
        "hostility_country_level":  1,
        "num_entries":              1,
        "num_unique_campus":        1,
        "entry_during_weekend":     0,
    },
]

# ── Helpers ────────────────────────────────────────────────────────────────────
def risk_color(prob):
    if prob > 0.80: return "#EF4444"
    if prob > 0.60: return "#F97316"
    if prob > 0.30: return "#EAB308"
    return "#22C55E"

def risk_label(prob):
    if prob > 0.80: return "🔴 Critical"
    if prob > 0.60: return "🟠 High"
    if prob > 0.30: return "🟡 Medium"
    return "🟢 Low"

def encode_profile(raw, encoders, feature_names):
    import pandas as pd
    df = pd.DataFrame([raw])
    for col, le in encoders.items():
        if col in df.columns:
            df[col] = le.transform(df[col].astype(str))
    return df[feature_names]

def score_profile(profile, model, encoders, feature_names):
    X = encode_profile(profile, encoders, feature_names)
    prob = float(model.predict_proba(X)[0][1])
    pred = int(model.predict(X)[0])
    return pred, prob

def sidebar_logo():
    logo_path = os.path.join(APP_DIR, "logo.png")
    if os.path.exists(logo_path):
        st.sidebar.image(logo_path, use_container_width=True)
    else:
        st.sidebar.markdown("""
        <div style="text-align:center; padding:10px 0;">
            <h2 style="color:#00D4FF; margin:0; font-family:Rajdhani;">
            <span style="color:white;">Insider</span><span style="color:#00D4FF;">Eye</span><span style="color:white;">Shield</span></h2>
            <p style="color:#9CA3AF; font-size:11px; letter-spacing:2px;">SEE BEYOND. PROTECT WITH INTELLIGENCE.</p>
        </div>""", unsafe_allow_html=True)
    st.sidebar.divider()

def load_tab_icon(filename):
    icon_path = os.path.join(APP_DIR, filename)
    if os.path.exists(icon_path):
        with open(icon_path, "rb") as f:
            data = base64.b64encode(f.read()).decode()
        return f"data:image/png;base64,{data}"
    return None

def inject_tab_icons(tab_files):
    icon_css = ""
    for i, filename in enumerate(tab_files):
        icon_path = os.path.join(APP_DIR, filename)
        if os.path.exists(icon_path):
            with open(icon_path, "rb") as f:
                data = base64.b64encode(f.read()).decode()
            icon_css += f"""
            .stTabs [data-baseweb="tab"]:nth-child({i+1}) {{
                padding-left: 50px;
                background-image: url("data:image/png;base64,{data}");
                background-repeat: no-repeat;
                background-size: 32px 32px;
                background-position: 10px center;
            }}"""
    if icon_css:
        st.markdown(f"<style>{icon_css}</style>", unsafe_allow_html=True)

def page_header(title, subtitle=""):
    st.markdown(f"""
    <div style="padding:20px 24px; border-radius:14px;
                background:linear-gradient(90deg,#0F172A,#111827);
                border:1px solid #1F2937; margin-bottom:20px;">
        <h1 style="color:white; margin:0; font-family:Rajdhani; font-size:2rem;">{title}</h1>
        {"<p style='color:#9CA3AF; margin:4px 0 0 0;'>"+subtitle+"</p>" if subtitle else ""}
    </div>
    """, unsafe_allow_html=True)
