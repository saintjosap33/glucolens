import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np
import joblib
import qrcode
import os
from datetime import datetime
from PIL import Image
import base64
import plotly.graph_objects as go

# ---------------- CONFIG ----------------
DATA_PATH   = "https://rxvrjhwmhjtipcqoxbyg.supabase.co/storage/v1/object/public/probation/diabetes_data.csv"
MODEL_PATH  = "diabetes_rf.pkl"
SCALER_PATH = "scaler.pkl"

QR_FOLDER = "/tmp/qrcodes"
os.makedirs(QR_FOLDER, exist_ok=True)

DEFAULT_FEATURES = [
    "Age","Gender","Ethnicity","SocioeconomicStatus","EducationLevel",
    "BMI","Smoking","AlcoholConsumption","PhysicalActivity","DietQuality",
    "SleepQuality","FamilyHistoryDiabetes","GestationalDiabetes","PolycysticOvarySyndrome",
    "PreviousPreDiabetes","Hypertension","SystolicBP","DiastolicBP","FastingBloodSugar",
    "HbA1c","SerumCreatinine","BUNLevels","CholesterolTotal","CholesterolLDL",
    "CholesterolHDL","CholesterolTriglycerides","AntihypertensiveMedications",
    "Statins","AntidiabeticMedications","FrequentUrination","ExcessiveThirst",
    "UnexplainedWeightLoss","FatigueLevels","BlurredVision","SlowHealingSores",
    "TinglingHandsFeet","QualityOfLifeScore","HeavyMetalsExposure",
    "OccupationalExposureChemicals","WaterQuality","MedicalCheckupsFrequency",
    "MedicationAdherence","HealthLiteracy"
]

# ---------------- CREDENTIALS ----------------
CREDENTIALS = {
    "admin":   {"password": "admin123", "role": "Admin"},
    "doctor":  {"password": "doc123",   "role": "Doctor"},
    "patient": {"password": "pat123",   "role": "Patient"},
}

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="GLUCO-LENS", layout="wide", page_icon="🩺")

st.markdown("""
<style>
    .main-header {
        text-align: center;
        font-size: 38px;
        font-weight: bold;
        color: white;
        background: linear-gradient(135deg, #0ea5e9, #7c3aed);
        padding: 22px;
        border-radius: 15px;
        margin-bottom: 24px;
        letter-spacing: 2px;
    }
    .metric-card {
        background: linear-gradient(135deg, #f0f9ff, #e0e7ff);
        border-radius: 12px;
        padding: 18px;
        text-align: center;
        margin-bottom: 10px;
    }
    .risk-high { color: #dc2626; font-size: 28px; font-weight: bold; }
    .risk-mid  { color: #d97706; font-size: 28px; font-weight: bold; }
    .risk-low  { color: #16a34a; font-size: 28px; font-weight: bold; }
    .login-box {
        max-width: 420px;
        margin: 60px auto;
        padding: 36px;
        border-radius: 18px;
        background: #f8fafc;
        box-shadow: 0 4px 32px rgba(0,0,0,0.10);
    }
    div[data-testid="stSidebar"] { background: #1e1b4b; }
    div[data-testid="stSidebar"] * { color: #e0e7ff !important; }
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='main-header'>🩺 GLUCO-LENS</div>", unsafe_allow_html=True)

# ---------------- SESSION STATE ----------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "role" not in st.session_state:
    st.session_state.role = None
if "username" not in st.session_state:
    st.session_state.username = None

# ---------------- MODEL ----------------
@st.cache_resource
def load_model_and_scaler():
    try:
        model  = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
    except Exception:
        model, scaler = None, None
    features = list(getattr(model, "feature_names_in_", DEFAULT_FEATURES)) if model else DEFAULT_FEATURES
    return model, scaler, features

MODEL, SCALER, FEATURES = load_model_and_scaler()

# ---------------- FALLBACK PATIENT DATA ----------------
DEFAULT_PATIENTS = pd.DataFrame([
    {
        "PatientID": "P001", "Name": "Arun Kumar",    "Age": 52, "Gender": 1,
        "BMI": 29.4, "HbA1c": 7.2, "FastingBloodSugar": 136, "CholesterolTotal": 210,
        "SystolicBP": 138, "DiastolicBP": 88, "Hypertension": 1,
        "FamilyHistoryDiabetes": 1, "Smoking": 0, "PhysicalActivity": 2,
    },
    {
        "PatientID": "P002", "Name": "Meena Selvam",  "Age": 45, "Gender": 0,
        "BMI": 27.1, "HbA1c": 6.1, "FastingBloodSugar": 112, "CholesterolTotal": 188,
        "SystolicBP": 122, "DiastolicBP": 80, "Hypertension": 0,
        "FamilyHistoryDiabetes": 0, "Smoking": 0, "PhysicalActivity": 4,
    },
    {
        "PatientID": "P003", "Name": "Ravi Shankar",  "Age": 61, "Gender": 1,
        "BMI": 33.2, "HbA1c": 8.4, "FastingBloodSugar": 162, "CholesterolTotal": 235,
        "SystolicBP": 148, "DiastolicBP": 94, "Hypertension": 1,
        "FamilyHistoryDiabetes": 1, "Smoking": 1, "PhysicalActivity": 1,
    },
    {
        "PatientID": "P004", "Name": "Priya Nair",    "Age": 38, "Gender": 0,
        "BMI": 24.8, "HbA1c": 5.6, "FastingBloodSugar": 95,  "CholesterolTotal": 172,
        "SystolicBP": 116, "DiastolicBP": 76, "Hypertension": 0,
        "FamilyHistoryDiabetes": 0, "Smoking": 0, "PhysicalActivity": 5,
    },
    {
        "PatientID": "P005", "Name": "Suresh Babu",   "Age": 57, "Gender": 1,
        "BMI": 31.0, "HbA1c": 7.8, "FastingBloodSugar": 148, "CholesterolTotal": 220,
        "SystolicBP": 142, "DiastolicBP": 90, "Hypertension": 1,
        "FamilyHistoryDiabetes": 1, "Smoking": 1, "PhysicalActivity": 2,
    },
])

# ---------------- DATA LOADER ----------------
@st.cache_data
def load_patient_db():
    try:
        df = pd.read_csv(DATA_PATH, dtype=str).fillna("")
        if df.empty:
            raise ValueError("Empty CSV")
        return df
    except Exception:
        return DEFAULT_PATIENTS.astype(str)

# ---------------- UTILS ----------------
def safe_float(x):
    try:
        return float(x)
    except Exception:
        return 0.0

def generate_qr(pid):
    path = os.path.join(QR_FOLDER, f"{pid}.png")
    if not os.path.exists(path):
        qr = qrcode.make(pid)
        qr.save(path)
    return path

def predict(record):
    if MODEL is None or SCALER is None:
        # Heuristic fallback using HbA1c + FastingBloodSugar
        hba1c = safe_float(record.get("HbA1c", 5.5))
        fbs   = safe_float(record.get("FastingBloodSugar", 90))
        score = min(1.0, max(0.0, (hba1c - 4.5) / 6.0 * 0.6 + (fbs - 70) / 200.0 * 0.4))
        return round(score, 4)
    x = [safe_float(record.get(f, 0)) for f in FEATURES]
    try:
        x = SCALER.transform([x])
        return float(MODEL.predict_proba(x)[0][1])
    except Exception:
        return 0.0

def risk_label(prob):
    if prob >= 0.65:
        return "🔴 High Risk",   "risk-high"
    elif prob >= 0.35:
        return "🟡 Moderate Risk", "risk-mid"
    else:
        return "🟢 Low Risk",    "risk-low"

# ================================================================
#  LOGIN SCREEN
# ================================================================
def show_login():
    st.markdown("<div class='login-box'>", unsafe_allow_html=True)
    st.markdown("### 🔐 Login to GLUCO-LENS")
    st.markdown("---")

    username = st.text_input("Username", placeholder="admin / doctor / patient")
    password = st.text_input("Password", type="password", placeholder="Enter password")

    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Login", use_container_width=True, type="primary"):
            if username in CREDENTIALS and CREDENTIALS[username]["password"] == password:
                st.session_state.logged_in = True
                st.session_state.role      = CREDENTIALS[username]["role"]
                st.session_state.username  = username
                st.rerun()
            else:
                st.error("❌ Invalid username or password.")

    st.markdown("---")
    st.caption("Demo credentials — Admin: `admin/admin123` | Doctor: `doctor/doc123` | Patient: `patient/pat123`")
    st.markdown("</div>", unsafe_allow_html=True)

# ================================================================
#  SIDEBAR (post-login)
# ================================================================
def show_sidebar():
    with st.sidebar:
        st.markdown(f"### 👤 {st.session_state.username}")
        st.markdown(f"**Role:** {st.session_state.role}")
        st.markdown("---")
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.logged_in = False
            st.session_state.role      = None
            st.session_state.username  = None
            st.rerun()

# ================================================================
#  ADMIN DASHBOARD
# ================================================================
def admin_dashboard(df):
    st.title("🛡️ Admin Dashboard")
    st.markdown("---")

    total   = len(df)
    has_hba = "HbA1c" in df.columns

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("👥 Total Patients", total)
    with col2:
        if has_hba:
            avg_hba1c = pd.to_numeric(df["HbA1c"], errors="coerce").mean()
            st.metric("📊 Avg HbA1c", f"{avg_hba1c:.2f}" if not np.isnan(avg_hba1c) else "N/A")
    with col3:
        if "Age" in df.columns:
            avg_age = pd.to_numeric(df["Age"], errors="coerce").mean()
            st.metric("🎂 Avg Age", f"{avg_age:.0f}" if not np.isnan(avg_age) else "N/A")

    st.markdown("#### 📋 All Patient Records")
    st.dataframe(df, use_container_width=True)

    # Quick chart if relevant columns exist
    if has_hba and "Age" in df.columns:
        st.markdown("#### 📈 HbA1c Distribution")
        hba_vals = pd.to_numeric(df["HbA1c"], errors="coerce").dropna()
        if not hba_vals.empty:
            fig = px.histogram(hba_vals, nbins=15, labels={"value": "HbA1c"},
                               color_discrete_sequence=["#7c3aed"])
            fig.update_layout(showlegend=False, margin=dict(t=20, b=20))
            st.plotly_chart(fig, use_container_width=True)

# ================================================================
#  DOCTOR DASHBOARD
# ================================================================
def doctor_dashboard(df):
    st.title("👨‍⚕️ Doctor Dashboard")
    st.markdown("---")

    pid = st.text_input("🔍 Enter Patient ID", placeholder="e.g. P001")

    if st.button("Fetch Patient", type="primary"):
        if not pid.strip():
            st.warning("Please enter a Patient ID.")
            return

        patient = df[df["PatientID"].astype(str).str.strip() == pid.strip()]

        if not patient.empty:
            rec  = patient.iloc[0].to_dict()
            prob = predict(rec)
            label, css_class = risk_label(prob)

            st.success("✅ Patient found!")

            col1, col2 = st.columns([2, 1])

            with col1:
                st.markdown("#### 📄 Patient Record")
                st.dataframe(patient.T.rename(columns={patient.index[0]: "Value"}),
                             use_container_width=True)

            with col2:
                st.markdown("#### 🧬 Diabetes Risk")
                st.markdown(f"<div class='metric-card'>"
                            f"<div class='{css_class}'>{prob*100:.1f}%</div>"
                            f"<div style='font-size:16px;margin-top:6px;'>{label}</div>"
                            f"</div>", unsafe_allow_html=True)

                st.markdown("#### 📱 Patient QR Code")
                try:
                    qr_path = generate_qr(pid.strip())
                    st.image(qr_path, width=160)
                except Exception:
                    st.info("QR generation unavailable.")

            # Risk gauge chart
            st.markdown("#### 📊 Risk Gauge")
            fig = go.Figure(go.Indicator(
                mode   = "gauge+number",
                value  = round(prob * 100, 1),
                number = {"suffix": "%"},
                gauge  = {
                    "axis": {"range": [0, 100]},
                    "bar":  {"color": "#7c3aed"},
                    "steps": [
                        {"range": [0,  35], "color": "#bbf7d0"},
                        {"range": [35, 65], "color": "#fef08a"},
                        {"range": [65,100], "color": "#fecaca"},
                    ],
                    "threshold": {"line": {"color": "red", "width": 4}, "value": 65}
                },
                title  = {"text": "Diabetes Risk Score"}
            ))
            fig.update_layout(height=280, margin=dict(t=40, b=10))
            st.plotly_chart(fig, use_container_width=True)

        else:
            st.error(f"❌ No patient found with ID: {pid}")

# ================================================================
#  PATIENT PORTAL
# ================================================================
def patient_portal(df):
    st.title("🧑‍💼 Patient Portal")
    st.markdown("---")

    pid = st.text_input("🪪 Enter Your Patient ID", placeholder="e.g. P001")

    if st.button("View My Record", type="primary"):
        if not pid.strip():
            st.warning("Please enter your Patient ID.")
            return

        patient = df[df["PatientID"].astype(str).str.strip() == pid.strip()]

        if not patient.empty:
            rec  = patient.iloc[0].to_dict()
            prob = predict(rec)
            label, css_class = risk_label(prob)

            name = rec.get("Name", pid)
            st.success(f"✅ Welcome, {name}!")

            col1, col2 = st.columns([2, 1])

            with col1:
                st.markdown("#### 📋 Your Health Record")
                display_cols = [c for c in ["PatientID","Name","Age","BMI","HbA1c",
                                             "FastingBloodSugar","CholesterolTotal",
                                             "SystolicBP","DiastolicBP"] if c in patient.columns]
                st.dataframe(patient[display_cols].T.rename(
                    columns={patient.index[0]: "Your Value"}),
                    use_container_width=True)

            with col2:
                st.markdown("#### 🩺 Your Diabetes Risk")
                st.markdown(f"<div class='metric-card'>"
                            f"<div class='{css_class}'>{prob*100:.1f}%</div>"
                            f"<div style='font-size:16px;margin-top:6px;'>{label}</div>"
                            f"</div>", unsafe_allow_html=True)

                if prob >= 0.65:
                    st.warning("⚠️ Please consult your doctor soon.")
                elif prob >= 0.35:
                    st.info("ℹ️ Monitor your health regularly.")
                else:
                    st.success("✅ Keep up the healthy lifestyle!")

            # Sparkline for key metrics
            key_metrics = {k: safe_float(rec.get(k, 0))
                           for k in ["BMI","HbA1c","FastingBloodSugar","CholesterolTotal"]
                           if k in rec}
            if key_metrics:
                st.markdown("#### 📊 Key Metrics")
                m_df = pd.DataFrame({"Metric": list(key_metrics.keys()),
                                     "Value":  list(key_metrics.values())})
                fig = px.bar(m_df, x="Metric", y="Value",
                             color="Metric",
                             color_discrete_sequence=["#0ea5e9","#7c3aed","#10b981","#f59e0b"])
                fig.update_layout(showlegend=False, margin=dict(t=20, b=10), height=260)
                st.plotly_chart(fig, use_container_width=True)

        else:
            st.error(f"❌ No record found for Patient ID: {pid}")

# ================================================================
#  MAIN ROUTER
# ================================================================
if not st.session_state.logged_in:
    show_login()
else:
    show_sidebar()
    df = load_patient_db()

    role = st.session_state.role

    if role == "Admin":
        admin_dashboard(df)
    elif role == "Doctor":
        doctor_dashboard(df)
    elif role == "Patient":
        patient_portal(df)
    else:
        st.error("Unknown role. Please log out and try again.")