# app.py

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

# ---------------- UI ----------------
st.set_page_config(page_title="GLUCO-LENS", layout="wide")

st.markdown("""
<div style='text-align:center; font-size:36px; font-weight:bold; color:white;
background:linear-gradient(135deg,#0ea5e9,#7c3aed);
padding:20px; border-radius:15px; margin-bottom:20px;'>
GLUCO-LENS
</div>
""", unsafe_allow_html=True)

# ---------------- MODEL ----------------
@st.cache_resource
def load_model_and_scaler():
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
    except:
        model, scaler = None, None

    features = getattr(model, "feature_names_in_", DEFAULT_FEATURES) if model else DEFAULT_FEATURES
    return model, scaler, list(features)

MODEL, SCALER, FEATURES = load_model_and_scaler()

# ---------------- DATA ----------------
@st.cache_data
def load_patient_db():
    try:
        return pd.read_csv(DATA_PATH, dtype=str).fillna("")
    except:
        return pd.DataFrame(columns=["PatientID"] + FEATURES)

# ---------------- UTILS ----------------
def safe_float(x):
    try:
        return float(x)
    except:
        return 0.0

def generate_qr(pid):
    path = os.path.join(QR_FOLDER, f"{pid}.png")
    if not os.path.exists(path):
        qr = qrcode.make(pid)
        qr.save(path)
    return path

def predict(record):
    if MODEL is None or SCALER is None:
        return 0.0
    x = [safe_float(record.get(f,0)) for f in FEATURES]
    try:
        x = SCALER.transform([x])
        return MODEL.predict_proba(x)[0][1]
    except:
        return 0.0

# ---------------- APP ----------------
st.sidebar.title("Select Role")
role = st.sidebar.radio("", ["Admin","Doctor","Patient"])

df = load_patient_db()

# ---------------- ADMIN ----------------
if role == "Admin":
    st.title("Admin Dashboard")
    st.write("Total Patients:", len(df))
    st.dataframe(df)

# ---------------- DOCTOR ----------------
elif role == "Doctor":
    st.title("Doctor Dashboard")

    uploaded = st.file_uploader("Upload QR Code (optional)", type=["png","jpg"])
    if uploaded:
        st.image(uploaded)
        st.info("QR auto-scan disabled. Enter Patient ID below.")

    pid = st.text_input("Enter Patient ID")

    if st.button("Fetch"):
        patient = df[df["PatientID"] == pid]
        if not patient.empty:
            rec = patient.iloc[0].to_dict()
            prob = predict(rec)

            st.success(f"Risk: {prob*100:.2f}%")

            qr = generate_qr(pid)
            st.image(qr, width=150)

            st.dataframe(patient)
        else:
            st.error("Patient not found")

# ---------------- PATIENT ----------------
else:
    st.title("Patient Portal")

    uploaded = st.file_uploader("Upload QR Code", type=["png","jpg"])
    if uploaded:
        st.image(uploaded)
        st.info("QR scanning disabled. Enter Patient ID manually.")

    pid = st.text_input("Enter Patient ID")

    if st.button("View Record"):
        patient = df[df["PatientID"] == pid]

        if not patient.empty:
            rec = patient.iloc[0].to_dict()
            prob = predict(rec)

            st.metric("Diabetes Risk", f"{prob*100:.2f}%")
            st.dataframe(patient)
        else:
            st.error("Patient not found")