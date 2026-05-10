import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

st.set_page_config(
    page_title="InsiderGuard – AI Threat Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

@st.cache_resource
def load_artifacts():
    with open(f"{MODELS_DIR}/rf_model.pkl",     "rb") as f: model    = pickle.load(f)
    with open(f"{MODELS_DIR}/scaler.pkl",        "rb") as f: scaler   = pickle.load(f)
    with open(f"{MODELS_DIR}/encoders.pkl",      "rb") as f: encoders = pickle.load(f)
    with open(f"{MODELS_DIR}/feature_names.pkl", "rb") as f: features = pickle.load(f)
    return model, scaler, encoders, features

model, scaler, encoders, feature_names = load_artifacts()

dept_options    = list(encoders["employee_department"].classes_)
campus_options  = list(encoders["employee_campus"].classes_)
pos_options     = list(encoders["employee_position"].classes_)
country_options = list(encoders["employee_origin_country"].classes_)


def encode_input(raw):
    df = pd.DataFrame([raw])
    for col, le in encoders.items():
        if col in df.columns:
            df[col] = le.transform(df[col].astype(str))
    return df[feature_names]


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


def render_result(pred, prob, raw_input):
    if pred == 1:
        st.error("## 🔴  MALICIOUS INSIDER ACTIVITY DETECTED")
    else:
        st.success("## 🟢  NORMAL / BENIGN BEHAVIOUR")

    k1, k2, k3 = st.columns(3)
    k1.metric("Classification",         "MALICIOUS" if pred == 1 else "NORMAL")
    k2.metric("Confidence (Malicious)", f"{prob * 100:.1f}%")
    risk = ("🔴 Critical" if prob > 0.80 else
            "🟠 High"     if prob > 0.60 else
            "🟡 Medium"   if prob > 0.30 else "🟢 Low")
    k3.metric("Risk Level", risk)

    st.divider()
    left, right = st.columns(2)

    with left:
        st.markdown("#### Confidence Score")
        fig, ax = plt.subplots(figsize=(5, 2.5))
        bar_color = "#DD8452" if pred == 1 else "#4C72B0"
        ax.barh(["P(Malicious)", "P(Normal)"], [prob, 1 - prob],
                color=[bar_color, "#AAAAAA"], edgecolor="white", height=0.45)
        ax.axvline(0.5, color="red", linestyle="--", linewidth=1.2, label="Decision boundary (0.5)")
        ax.set_xlim(0, 1)
        ax.set_xlabel("Probability")
        ax.legend(fontsize=8, loc="lower right")
        ax.text(min(prob + 0.03, 0.88), 0, f"{prob*100:.1f}%", va="center", fontsize=11, fontweight="bold")
        ax.text(min(1-prob + 0.03, 0.88), 1, f"{(1-prob)*100:.1f}%", va="center", fontsize=11)
        ax.set_title("Model Confidence", fontweight="bold")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

    with right:
        st.markdown("#### Feature Importance (Global – Random Forest)")
        fi = (pd.Series(model.feature_importances_, index=feature_names)
                .sort_values(ascending=True).tail(12))
        high_risk = {"total_files_burned", "burned_from_other", "num_printed_pages_off_hours"}
        colors_fi = ["#DD8452" if f in high_risk else "#4C72B0" for f in fi.index]
        fig2, ax2 = plt.subplots(figsize=(5, 4))
        ax2.barh(fi.index, fi.values, color=colors_fi, edgecolor="white")
        ax2.set_xlabel("Importance Score")
        ax2.set_title("Top Features Driving Classification", fontweight="bold")
        ax2.legend(handles=[
            mpatches.Patch(color="#DD8452", label="High-risk indicator"),
            mpatches.Patch(color="#4C72B0", label="Profile feature"),
        ], fontsize=8, loc="lower right")
        fig2.tight_layout()
        st.pyplot(fig2)
        plt.close()

    st.divider()
    st.markdown("#### 🧠 Risk Indicator Analysis")
    st.caption("Plain-language explanation — suitable for non-technical managers:")
    for line in build_explanation(raw_input):
        st.markdown(f"- {line}")

    st.divider()
    st.markdown("#### 📋 Recommended Action")
    if pred == 1 and prob > 0.80:
        st.warning("**CRITICAL:** Escalate immediately to the Security Operations Centre. Suspend access and preserve all logs under chain-of-custody procedures.")
    elif pred == 1:
        st.warning("**HIGH RISK:** Flag for urgent security team review. Monitor all subsequent activity and cross-reference physical access logs.")
    elif prob > 0.30:
        st.info("**MEDIUM RISK:** Anomalous signals below the malicious threshold. Schedule routine review and continue passive monitoring.")
    else:
        st.success("**LOW RISK:** Behaviour consistent with normal activity. No immediate action required.")


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🛡️ InsiderGuard")
    st.caption("COS720 – AI Insider Threat Detection\nUniversity of Pretoria, 2026")
    st.divider()
    mode = st.radio("Input mode", ["📋 Manual Entry", "📁 Upload CSV"])
    st.divider()
    st.markdown("**Model:** Random Forest (100 trees)")
    st.markdown("**Training records:** 118,614")
    st.markdown("**Imbalance:** SMOTE applied")
    st.markdown("**Test Recall (malicious):** 82%")
    st.markdown("**Test F1 (malicious):** 75%")
    st.markdown("**AUC:** 0.97")

# ── Header ─────────────────────────────────────────────────────────────────────
st.title("🛡️ InsiderGuard — AI-Powered Insider Threat Detection")
st.markdown("Classify employee behaviour as **Normal** or **Malicious Insider**, with a confidence score and plain-language risk explanation.")
st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# MANUAL ENTRY
# ══════════════════════════════════════════════════════════════════════════════
if mode == "📋 Manual Entry":
    st.subheader("Employee Behaviour Profile")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Employee Profile**")
        department = st.selectbox("Department",     dept_options,
                                  index=dept_options.index("Engineering Department")
                                        if "Engineering Department" in dept_options else 0)
        campus     = st.selectbox("Campus",         campus_options, index=0)
        position   = st.selectbox("Position",       pos_options,
                                  index=pos_options.index("Systems Engineer")
                                        if "Systems Engineer" in pos_options else 0)
        origin     = st.selectbox("Origin Country", country_options,
                                  index=country_options.index("Israel")
                                        if "Israel" in country_options else 0)
        seniority  = st.slider("Seniority (years)", 0, 40, 5)
        classif    = st.selectbox("Privilege Classification", [1, 2, 3, 4], index=1)
        contractor = st.checkbox("Is Contractor")

    with col2:
        st.markdown("**Behavioural Activity**")
        total_printed   = st.number_input("Total Pages Printed",            min_value=0, value=0, step=1)
        off_hrs_printed = st.number_input("Pages Printed Off-Hours",        min_value=0, value=0, step=1)
        files_burned    = st.number_input("Files Burned (Removable Media)", min_value=0, value=0, step=1)
        burned_other    = st.number_input("Files Burned (from Others)",     min_value=0, value=0, step=1)
        num_entries     = st.number_input("Building Entries (count)",        min_value=0, value=1, step=1)
        num_campus      = st.number_input("Unique Campuses Visited",         min_value=1, value=1, step=1)
        weekend_entry   = st.checkbox("Weekend Building Access")

    with col3:
        st.markdown("**Risk Flags & Travel**")
        foreign_citizen = st.checkbox("Has Foreign Citizenship")
        criminal_record = st.checkbox("Has Criminal Record")
        medical_history = st.checkbox("Has Medical History on File")
        abroad          = st.checkbox("Currently Abroad")
        trip_day        = st.number_input("Trip Day Number",         min_value=0, value=0, step=1)
        hostility       = st.selectbox("Hostility Country Level", [0, 1, 2, 3, 4], index=0)

    raw_input = {
        "employee_department":        department,
        "employee_campus":            campus,
        "employee_position":          position,
        "employee_seniority_years":   seniority,
        "is_contractor":              int(contractor),
        "employee_classification":    classif,
        "has_foreign_citizenship":    int(foreign_citizen),
        "has_criminal_record":        int(criminal_record),
        "has_medical_history":        int(medical_history),
        "employee_origin_country":    origin,
        "total_printed_pages":        total_printed,
        "num_printed_pages_off_hours": off_hrs_printed,
        "total_files_burned":         files_burned,
        "burned_from_other":          burned_other,
        "is_abroad":                  int(abroad),
        "trip_day_number":            trip_day,
        "hostility_country_level":    hostility,
        "num_entries":                num_entries,
        "num_unique_campus":          num_campus,
        "entry_during_weekend":       int(weekend_entry),
    }

    if st.button("🔍  Analyse Employee", use_container_width=True, type="primary"):
        X_enc = encode_input(raw_input)
        pred  = int(model.predict(X_enc)[0])
        prob  = float(model.predict_proba(X_enc)[0][1])
        st.divider()
        render_result(pred, prob, raw_input)

# ══════════════════════════════════════════════════════════════════════════════
# CSV UPLOAD
# ══════════════════════════════════════════════════════════════════════════════
else:
    st.subheader("Batch Analysis — Upload CSV")
    st.info("Upload a CSV with the same columns as the Kaggle insider threat dataset. `is_malicious` and `late_exit_flag` are ignored if present.")
    uploaded = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded:
        try:
            raw_df = pd.read_csv(uploaded, sep=None, engine="python")
            st.write(f"Loaded **{len(raw_df):,} records** × {raw_df.shape[1]} columns.")
            st.dataframe(raw_df.head(5), use_container_width=True)

            for drop_col in ["is_malicious", "late_exit_flag"]:
                if drop_col in raw_df.columns:
                    raw_df = raw_df.drop(columns=[drop_col])

            enc_df = raw_df.copy()
            for col, le in encoders.items():
                if col in enc_df.columns:
                    enc_df[col] = le.transform(enc_df[col].astype(str))

            X_batch = enc_df[feature_names]
            preds   = model.predict(X_batch)
            probs   = model.predict_proba(X_batch)[:, 1]

            out_df = raw_df.copy()
            out_df["Prediction"]     = ["🔴 MALICIOUS" if p == 1 else "🟢 NORMAL" for p in preds]
            out_df["Confidence (%)"] = (probs * 100).round(1)
            out_df["Risk Level"]     = pd.cut(probs,
                bins=[0, .3, .6, .8, 1.01],
                labels=["🟢 Low", "🟡 Medium", "🟠 High", "🔴 Critical"])

            st.divider()
            n_mal = int((preds == 1).sum())
            c1, c2, c3 = st.columns(3)
            c1.metric("Total Records",        f"{len(preds):,}")
            c2.metric("🔴 Malicious Flagged", f"{n_mal:,}")
            c3.metric("Detection Rate",        f"{n_mal / len(preds) * 100:.1f}%")

            display_cols = ["Prediction", "Confidence (%)", "Risk Level"] + list(raw_df.columns)
            st.dataframe(out_df[display_cols], use_container_width=True)

            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            axes[0].bar(["Normal", "Malicious"], [(preds==0).sum(), (preds==1).sum()],
                        color=["#4C72B0","#DD8452"], edgecolor="white")
            axes[0].set_title("Prediction Distribution", fontweight="bold")
            axes[0].set_ylabel("Count")
            axes[1].hist(probs, bins=30, color="#55A868", edgecolor="white", alpha=0.85)
            axes[1].axvline(0.5, color="red", linestyle="--", label="Decision boundary")
            axes[1].set_title("Confidence Score Distribution", fontweight="bold")
            axes[1].set_xlabel("P(Malicious)")
            axes[1].set_ylabel("Count")
            axes[1].legend()
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            st.download_button("⬇️  Download Results CSV",
                               out_df.to_csv(index=False),
                               file_name="threat_predictions.csv", mime="text/csv")

        except Exception as e:
            st.error(f"Error processing file: {e}")
