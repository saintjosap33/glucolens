"""
GLUCO-LENS - Smart EMR
Cloud-safe Streamlit app (no pyzbar, no cv2, no webcam)
Requires: diabetes_rf.pkl, scaler.pkl (optional - heuristic fallback if missing)
"""

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
QR_FOLDER   = "/tmp/qrcodes"
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

CREDENTIALS = {
    "admin":   {"password": "admin123", "role": "Admin"},
    "doctor":  {"password": "doc123",   "role": "Doctor"},
    "patient": {"password": "pat123",   "role": "Patient"},
}

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="GLUCO-LENS", page_icon="🔍", layout="wide")

st.markdown("""
<style>
body {
    background: #0f1724;
    color: #e6eef8;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}
.main-header {
    background: linear-gradient(135deg, #0ea5e9, #7c3aed);
    padding: 22px 18px;
    border-radius: 18px;
    color: white;
    text-align: center;
    margin-bottom: 24px;
    font-size: 36px;
    font-weight: 800;
    box-shadow: 0 10px 24px rgba(0,0,0,0.5);
    letter-spacing: 2px;
}
.card {
    background: linear-gradient(180deg, rgba(255,255,255,0.04), rgba(255,255,255,0.01));
    border-radius: 16px;
    padding: 20px;
    box-shadow: 0 10px 24px rgba(2,6,23,0.7);
    margin-bottom: 18px;
}
.login-box {
    max-width: 420px;
    margin: 40px auto;
    padding: 36px;
    border-radius: 18px;
    background: rgba(255,255,255,0.04);
    box-shadow: 0 4px 32px rgba(0,0,0,0.4);
}
.risk-high { color: #ff3b30; font-size: 30px; font-weight: bold; }
.risk-mid  { color: #ff9f0a; font-size: 30px; font-weight: bold; }
.risk-low  { color: #34c759; font-size: 30px; font-weight: bold; }
.stButton>button {
    background: linear-gradient(135deg, #7c3aed, #0ea5e9);
    color: white;
    border-radius: 12px;
    padding: 12px 24px;
    font-size: 15px;
    font-weight: 700;
    border: none;
    transition: 0.2s;
}
.stButton>button:hover { opacity: 0.88; transform: translateY(-2px); }
div[data-testid="stSidebar"] { background: #1e1b4b; }
div[data-testid="stSidebar"] * { color: #e0e7ff !important; }
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='main-header'>🔍 GLUCO-LENS</div>", unsafe_allow_html=True)

# ---------------- SESSION STATE ----------------
for key, val in [("logged_in", False), ("role", None), ("username", None),
                 ("current_record_doc", None), ("current_pid_doc", None),
                 ("current_record", None), ("current_pid", None)]:
    if key not in st.session_state:
        st.session_state[key] = val

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

# ---------------- FALLBACK PATIENTS ----------------
DEFAULT_PATIENTS = pd.DataFrame([
    {"PatientID":"P001","Name":"Arun Kumar",   "Age":52,"Gender":1,"BMI":29.4,
     "HbA1c":7.2,"FastingBloodSugar":136,"CholesterolTotal":210,
     "SystolicBP":138,"DiastolicBP":88,"Hypertension":1,
     "FamilyHistoryDiabetes":1,"Smoking":0,"PhysicalActivity":2,
     "DietQuality":2,"SleepQuality":2,"Diagnosis":"Pre-diabetic","DoctorRemarks":""},
    {"PatientID":"P002","Name":"Meena Selvam", "Age":45,"Gender":0,"BMI":27.1,
     "HbA1c":6.1,"FastingBloodSugar":112,"CholesterolTotal":188,
     "SystolicBP":122,"DiastolicBP":80,"Hypertension":0,
     "FamilyHistoryDiabetes":0,"Smoking":0,"PhysicalActivity":4,
     "DietQuality":3,"SleepQuality":3,"Diagnosis":"Normal","DoctorRemarks":""},
    {"PatientID":"P003","Name":"Ravi Shankar", "Age":61,"Gender":1,"BMI":33.2,
     "HbA1c":8.4,"FastingBloodSugar":162,"CholesterolTotal":235,
     "SystolicBP":148,"DiastolicBP":94,"Hypertension":1,
     "FamilyHistoryDiabetes":1,"Smoking":1,"PhysicalActivity":1,
     "DietQuality":1,"SleepQuality":1,"Diagnosis":"Diabetic","DoctorRemarks":"On Metformin"},
    {"PatientID":"P004","Name":"Priya Nair",   "Age":38,"Gender":0,"BMI":24.8,
     "HbA1c":5.6,"FastingBloodSugar":95, "CholesterolTotal":172,
     "SystolicBP":116,"DiastolicBP":76,"Hypertension":0,
     "FamilyHistoryDiabetes":0,"Smoking":0,"PhysicalActivity":5,
     "DietQuality":4,"SleepQuality":4,"Diagnosis":"Normal","DoctorRemarks":""},
    {"PatientID":"P005","Name":"Suresh Babu",  "Age":57,"Gender":1,"BMI":31.0,
     "HbA1c":7.8,"FastingBloodSugar":148,"CholesterolTotal":220,
     "SystolicBP":142,"DiastolicBP":90,"Hypertension":1,
     "FamilyHistoryDiabetes":1,"Smoking":1,"PhysicalActivity":2,
     "DietQuality":2,"SleepQuality":2,"Diagnosis":"Diabetic","DoctorRemarks":"Follow-up needed"},
])

# ---------------- DATA ----------------
@st.cache_data
def load_patient_db():
    try:
        df = pd.read_csv(DATA_PATH, dtype=str).fillna("")
        if df.empty:
            raise ValueError("empty")
        for col in ["Name","Diagnosis","DoctorRemarks","created_at"]:
            if col not in df.columns:
                df[col] = ""
        return df
    except Exception:
        return DEFAULT_PATIENTS.astype(str)

# ---------------- UTILS ----------------
def safe_float(x, default=0.0):
    try:
        if x is None or x == "" or (isinstance(x, float) and np.isnan(x)):
            return float(default)
        return float(x)
    except Exception:
        return float(default)

def generate_qr(pid):
    path = os.path.join(QR_FOLDER, f"{pid}.png")
    if not os.path.exists(path):
        qr = qrcode.QRCode(version=1, box_size=6, border=2)
        qr.add_data(pid)
        qr.make(fit=True)
        img = qr.make_image(fill_color="black", back_color="white")
        img.save(path)
    return path

def get_patient_by_id(pid, df=None):
    if df is None:
        df = load_patient_db()
    rows = df[df["PatientID"].astype(str).str.strip() == str(pid).strip()]
    return rows.iloc[0].to_dict() if not rows.empty else None

def predict_record_prob(record):
    if MODEL is None or SCALER is None:
        hba1c = safe_float(record.get("HbA1c", 5.5))
        fbs   = safe_float(record.get("FastingBloodSugar", 90))
        prob  = min(1.0, max(0.0, (hba1c - 4.5) / 6.0 * 0.6 + (fbs - 70) / 200.0 * 0.4))
        return round(prob, 4), "heuristic"
    x = [safe_float(record.get(f, 0)) for f in FEATURES]
    try:
        arr  = SCALER.transform([x])
        prob = float(MODEL.predict_proba(arr)[0][1])
        return prob, "model"
    except Exception:
        hba1c = safe_float(record.get("HbA1c", 5.5))
        prob  = 0.9 if hba1c >= 6.5 else (0.5 if hba1c >= 5.7 else 0.1)
        return prob, "heuristic"

def risk_label(prob):
    if prob >= 0.65:
        return "🔴 High Risk",     "risk-high"
    elif prob >= 0.35:
        return "🟡 Moderate Risk", "risk-mid"
    else:
        return "🟢 Low Risk",      "risk-low"

def simulate_future_risk(record, years=5):
    projections = []
    for y in range(1, years + 1):
        mod = dict(record)
        mod["BMI"]              = str(safe_float(mod.get("BMI", 25)) + 0.3 * y)
        mod["FastingBloodSugar"]= str(safe_float(mod.get("FastingBloodSugar", 100)) + 2 * y)
        pred, _ = predict_record_prob(mod)
        projections.append(round(pred * 100, 2))
    return projections

def human_ai_summary(record, prob):
    hba1c = safe_float(record.get("HbA1c", 5.5))
    bmi   = safe_float(record.get("BMI", 25))
    hba1c_status = "normal" if hba1c < 5.7 else ("pre-diabetes range" if hba1c < 6.5 else "diabetes range")
    bmi_status   = "healthy weight" if bmi < 25 else ("overweight" if bmi < 30 else "obese")
    if prob < 0.35:
        tone = (f"Low risk. HbA1c {hba1c:.1f}% ({hba1c_status}), BMI {bmi:.1f} ({bmi_status}). "
                "Recommend maintaining current lifestyle with yearly check-ups.")
    elif prob < 0.65:
        tone = (f"Moderate risk. HbA1c {hba1c:.1f}% ({hba1c_status}), BMI {bmi:.1f} ({bmi_status}). "
                "Advise dietary improvements, increased activity, follow-up in 3–6 months.")
    else:
        tone = (f"High risk. HbA1c {hba1c:.1f}% ({hba1c_status}), BMI {bmi:.1f} ({bmi_status}). "
                "Recommend urgent clinical follow-up, medication review, and targeted interventions.")
    proj      = simulate_future_risk(record)
    proj_text = "Projected 5-year risk: " + " → ".join([f"{p}%" for p in proj])
    return tone + " " + proj_text

def format_val(v):
    if v is None or v == "":
        return "N/A"
    try:
        return f"{float(v):.2f}"
    except Exception:
        return str(v)

def download_csv_link(df, filename="data.csv"):
    b64 = base64.b64encode(df.to_csv(index=False).encode()).decode()
    return f'<a href="data:file/csv;base64,{b64}" download="{filename}">⬇️ Download CSV</a>'

def risk_gauge(prob):
    fig = go.Figure(go.Indicator(
        mode  = "gauge+number",
        value = round(prob * 100, 1),
        number= {"suffix": "%", "font": {"color": "white"}},
        gauge = {
            "axis": {"range": [0, 100], "tickcolor": "white"},
            "bar":  {"color": "#7c3aed"},
            "bgcolor": "#1e1b4b",
            "steps": [
                {"range": [0,  35], "color": "#064e3b"},
                {"range": [35, 65], "color": "#78350f"},
                {"range": [65, 100],"color": "#7f1d1d"},
            ],
            "threshold": {"line": {"color": "white", "width": 3}, "value": prob * 100}
        },
        title = {"text": "Diabetes Risk Score", "font": {"color": "white"}}
    ))
    fig.update_layout(height=280, margin=dict(t=40, b=10),
                      paper_bgcolor="#0f1724", font=dict(color="white"))
    return fig

# ================================================================
#  LOGIN
# ================================================================
def show_login():
    st.markdown("<div class='login-box'>", unsafe_allow_html=True)
    st.markdown("### 🔐 Login to GLUCO-LENS")
    st.markdown("---")
    username = st.text_input("Username", placeholder="admin / doctor / patient")
    password = st.text_input("Password", type="password")
    if st.button("Login", use_container_width=True, type="primary"):
        u = username.lower().strip()
        if u in CREDENTIALS and CREDENTIALS[u]["password"] == password:
            st.session_state.logged_in = True
            st.session_state.role      = CREDENTIALS[u]["role"]
            st.session_state.username  = u
            st.rerun()
        else:
            st.error("❌ Invalid username or password.")
    st.markdown("---")
    st.caption("Demo — `admin / admin123` · `doctor / doc123` · `patient / pat123`")
    st.markdown("</div>", unsafe_allow_html=True)

# ================================================================
#  SIDEBAR
# ================================================================
def show_sidebar():
    with st.sidebar:
        st.markdown(f"### 👤 {st.session_state.username}")
        st.markdown(f"**Role:** `{st.session_state.role}`")
        st.markdown("---")
        if st.button("🚪 Logout", use_container_width=True):
            for k in ["logged_in","role","username",
                      "current_record_doc","current_pid_doc",
                      "current_record","current_pid"]:
                st.session_state[k] = False if k == "logged_in" else None
            st.rerun()

# ================================================================
#  ADMIN DASHBOARD
# ================================================================
def admin_dashboard():
    st.title("👑 Admin Dashboard")
    df = load_patient_db()

    c1, c2, c3 = st.columns(3)
    c1.metric("👥 Total Patients", len(df))
    hba = pd.to_numeric(df.get("HbA1c", pd.Series(dtype=float)), errors="coerce").mean()
    c2.metric("📊 Avg HbA1c", f"{hba:.2f}" if not np.isnan(hba) else "N/A")
    age = pd.to_numeric(df.get("Age",  pd.Series(dtype=float)), errors="coerce").mean()
    c3.metric("🎂 Avg Age",   f"{age:.0f}" if not np.isnan(age) else "N/A")

    tab1, tab2, tab3 = st.tabs(["📋 View & Export", "➕ Add Patient", "🔧 Bulk Utilities"])

    with tab1:
        st.markdown(download_csv_link(df, "patients_export.csv"), unsafe_allow_html=True)
        st.dataframe(df, use_container_width=True)
        if "HbA1c" in df.columns:
            hba_vals = pd.to_numeric(df["HbA1c"], errors="coerce").dropna()
            if not hba_vals.empty:
                st.markdown("#### 📈 HbA1c Distribution")
                fig = px.histogram(hba_vals, nbins=15, labels={"value":"HbA1c"},
                                   color_discrete_sequence=["#7c3aed"])
                fig.update_layout(showlegend=False, margin=dict(t=20,b=10),
                                  paper_bgcolor="#0f1724", plot_bgcolor="#0f1724",
                                  font=dict(color="white"))
                st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("Add New Patient")
        with st.form("add_patient_form"):
            pid   = st.text_input("PatientID", value=f"P{len(df)+1:03d}")
            name  = st.text_input("Name")
            age_v = st.number_input("Age",               min_value=1,   max_value=120, value=40)
            bmi_v = st.number_input("BMI",               min_value=10.0,max_value=60.0,value=25.0)
            hba_v = st.number_input("HbA1c",             min_value=3.0, max_value=15.0,value=5.5)
            fbs_v = st.number_input("FastingBloodSugar", min_value=50,  max_value=400, value=100)
            cho_v = st.number_input("CholesterolTotal",  min_value=50,  max_value=500, value=180)
            sbp_v = st.number_input("SystolicBP",        min_value=60,  max_value=250, value=120)
            dbp_v = st.number_input("DiastolicBP",       min_value=40,  max_value=150, value=80)
            diag  = st.text_input("Diagnosis", value="")
            if st.form_submit_button("✅ Create Patient"):
                if df[df["PatientID"].astype(str) == str(pid)].empty:
                    row = {"PatientID":pid,"Name":name,"Age":age_v,"BMI":bmi_v,
                           "HbA1c":hba_v,"FastingBloodSugar":fbs_v,
                           "CholesterolTotal":cho_v,"SystolicBP":sbp_v,"DiastolicBP":dbp_v,
                           "Diagnosis":diag,"DoctorRemarks":"",
                           "created_at":datetime.utcnow().isoformat()}
                    try:
                        qr_p = generate_qr(pid)
                        st.image(qr_p, width=150, caption=f"QR for {pid}")
                    except Exception:
                        pass
                    st.success(f"Patient {pid} created!")
                else:
                    st.error("PatientID already exists.")

    with tab3:
        uploaded = st.file_uploader("Upload CSV to append patients", type=["csv"])
        if uploaded:
            try:
                add_df = pd.read_csv(uploaded, dtype=str).fillna("")
                st.success(f"Preview — {len(add_df)} rows loaded.")
                st.dataframe(add_df.head())
            except Exception as e:
                st.error(f"Failed: {e}")
        if st.button("🔄 Regenerate All QR Codes"):
            df2 = load_patient_db()
            n = 0
            for p in df2["PatientID"].astype(str):
                try:
                    generate_qr(p); n += 1
                except Exception:
                    pass
            st.success(f"Regenerated {n} QR codes.")

# ================================================================
#  DOCTOR DASHBOARD
# ================================================================
def doctor_dashboard():
    st.title("👩‍⚕️ Doctor Dashboard")
    df = load_patient_db()

    pid_input = st.text_input("🔍 Enter Patient ID", placeholder="e.g. P001")
    if st.button("Fetch Patient", type="primary"):
        if pid_input.strip():
            rec = get_patient_by_id(pid_input.strip(), df)
            if rec:
                st.session_state.current_record_doc = rec
                st.session_state.current_pid_doc    = pid_input.strip()
            else:
                st.error(f"No patient found: {pid_input.strip()}")

    rec = st.session_state.get("current_record_doc")
    pid = st.session_state.get("current_pid_doc")

    if rec:
        prob, source = predict_record_prob(rec)
        label, css   = risk_label(prob)
        st.success(f"✅ **{rec.get('Name', pid)}** | ID: `{pid}`")
        if source == "heuristic":
            st.caption("⚠️ ML model not loaded — using heuristic estimate.")

        vitals = ["FastingBloodSugar","HbA1c","CholesterolTotal","BMI","SystolicBP","DiastolicBP"]
        vcols  = st.columns(len(vitals))
        for i, v in enumerate(vitals):
            vcols[i].metric(v, format_val(rec.get(v,"")))

        col_l, col_r = st.columns([2, 1])
        with col_l:
            proj     = simulate_future_risk(rec)
            fig_proj = go.Figure()
            fig_proj.add_trace(go.Scatter(
                x=list(range(1,6)), y=proj, mode="lines+markers",
                line=dict(color="#7c3aed",width=3), marker=dict(size=8,color="#0ea5e9")))
            fig_proj.update_layout(
                title="📈 5-Year Risk Projection",
                xaxis_title="Years from now", yaxis=dict(range=[0,100],title="Risk %"),
                paper_bgcolor="#0f1724", plot_bgcolor="#0f1724",
                font=dict(color="white"), height=280, margin=dict(t=40,b=20))
            st.plotly_chart(fig_proj, use_container_width=True)

            st.markdown("#### 📋 Full Patient Record")
            tbl = [(k, format_val(v)) for k, v in rec.items() if k != "created_at"]
            st.table(pd.DataFrame(tbl, columns=["Field","Value"]))

        with col_r:
            st.plotly_chart(risk_gauge(prob), use_container_width=True)
            st.markdown(f"<div style='text-align:center'><span class='{css}'>{label}</span></div>",
                        unsafe_allow_html=True)
            st.markdown("#### 📱 Patient QR Code")
            try:
                st.image(generate_qr(pid), width=160)
            except Exception:
                st.info("QR unavailable.")

        st.markdown("---")
        st.markdown("#### 🤖 AI Clinical Summary")
        st.info(human_ai_summary(rec, prob))

        note = st.text_area("📝 Doctor Remarks", value=rec.get("DoctorRemarks",""))
        if st.button("💾 Save Remarks"):
            st.session_state.current_record_doc["DoctorRemarks"] = note
            st.success("Remarks saved to session ✅")
    else:
        st.info("Enter a Patient ID above and click **Fetch Patient**.")

# ================================================================
#  PATIENT PORTAL
# ================================================================
def patient_portal():
    st.title("🫀 Patient Portal")
    df = load_patient_db()

    pid_input = st.text_input("🪪 Enter Your Patient ID", placeholder="e.g. P001")
    if st.button("View My Record", type="primary"):
        if pid_input.strip():
            rec = get_patient_by_id(pid_input.strip(), df)
            if rec:
                st.session_state.current_record = rec
                st.session_state.current_pid    = pid_input.strip()
            else:
                st.error(f"No record found for: {pid_input.strip()}")

    rec = st.session_state.get("current_record")
    pid = st.session_state.get("current_pid")

    if rec:
        prob, source = predict_record_prob(rec)
        label, css   = risk_label(prob)
        st.success(f"✅ Welcome, **{rec.get('Name', pid)}**!")

        vitals = ["FastingBloodSugar","HbA1c","BMI","CholesterolTotal","SystolicBP","DiastolicBP"]
        vcols  = st.columns(len(vitals))
        for i, v in enumerate(vitals):
            vcols[i].metric(v, format_val(rec.get(v,"")))

        col_l, col_r = st.columns([2, 1])
        with col_l:
            key_m = {k: safe_float(rec.get(k,0))
                     for k in ["BMI","HbA1c","FastingBloodSugar","CholesterolTotal"] if k in rec}
            fig_bar = px.bar(
                pd.DataFrame({"Metric":list(key_m.keys()),"Value":list(key_m.values())}),
                x="Metric", y="Value", color="Metric",
                color_discrete_sequence=["#0ea5e9","#7c3aed","#10b981","#f59e0b"])
            fig_bar.update_layout(showlegend=False, height=260, margin=dict(t=20,b=10),
                                  paper_bgcolor="#0f1724", plot_bgcolor="#0f1724",
                                  font=dict(color="white"))
            st.plotly_chart(fig_bar, use_container_width=True)

            proj     = simulate_future_risk(rec)
            fig_proj = go.Figure()
            fig_proj.add_trace(go.Scatter(
                x=list(range(1,6)), y=proj, mode="lines+markers",
                line=dict(color="#0ea5e9",width=3), marker=dict(size=8,color="#7c3aed")))
            fig_proj.update_layout(
                title="📈 5-Year Risk Projection",
                yaxis=dict(range=[0,100],title="Risk %"), xaxis_title="Years from now",
                paper_bgcolor="#0f1724", plot_bgcolor="#0f1724",
                font=dict(color="white"), height=260, margin=dict(t=40,b=10))
            st.plotly_chart(fig_proj, use_container_width=True)

        with col_r:
            st.plotly_chart(risk_gauge(prob), use_container_width=True)
            st.markdown(f"<div style='text-align:center'><span class='{css}'>{label}</span></div>",
                        unsafe_allow_html=True)
            if prob >= 0.65:
                st.error("⚠️ Please consult your doctor soon.")
            elif prob >= 0.35:
                st.warning("💡 Monitor your health regularly.")
            else:
                st.success("✅ Keep up the healthy lifestyle!")

        st.markdown("---")
        st.markdown("#### 🤖 AI Health Summary")
        st.info(human_ai_summary(rec, prob))

        remarks = rec.get("DoctorRemarks","")
        if remarks:
            st.markdown("#### 📝 Doctor's Remarks")
            st.text_area("", value=remarks, height=100, disabled=True)

        st.download_button("⬇️ Download My Record as CSV",
                           data=pd.DataFrame([rec]).to_csv(index=False),
                           file_name=f"{pid}_record.csv", mime="text/csv")

        # ---- Lifestyle Questionnaire ----
        st.markdown("---")
        st.markdown("#### 📝 Lifestyle Risk Estimator")
        st.caption("Answer to see how your daily habits influence your diabetes risk.")

        with st.form("lifestyle_form"):
            q1 = st.select_slider("Exercise (mins/day)?",       ["0","10-30","30-60","60+"],          value="30-60")
            q2 = st.select_slider("Added sugar (tsp/day)?",     ["0-5","6-10","11-20","20+"],          value="6-10")
            q3 = st.select_slider("Veg/fruit servings/day?",    ["0-1","2-3","4-5","5+"],              value="2-3")
            q4 = st.select_slider("Sleep (hrs/night)?",         ["<5","5-6","6-7","7+"],               value="6-7")
            q5 = st.select_slider("Fast food frequency?",       ["Daily","Few/week","Weekly","Rarely"],value="Few/week")
            q6 = st.select_slider("Stress level (0-10 scale)?", ["0-2","3-5","6-8","9-10"],            value="3-5")
            submitted = st.form_submit_button("Calculate Lifestyle-Adjusted Risk")

        if submitted:
            sm = {"0":0,"10-30":1,"30-60":2,"60+":3,
                  "0-5":3,"6-10":2,"11-20":1,"20+":0,
                  "0-1":0,"2-3":1,"4-5":2,"5+":3,
                  "<5":0,"5-6":1,"6-7":2,"7+":3,
                  "Daily":0,"Few/week":1,"Weekly":2,"Rarely":3,
                  "0-2":3,"3-5":2,"6-8":1,"9-10":0}
            total    = sum(sm.get(q,0) for q in [q1,q2,q3,q4,q5,q6])
            mod      = (total - 9) / 18.0
            new_prob = float(np.clip(prob - mod * 0.25, 0, 1))
            diff     = new_prob - prob
            hba_base = safe_float(rec.get("HbA1c", 5.5))
            hba_new  = hba_base + diff * 2

            ca, cb = st.columns(2)
            ca.metric("Original Risk",           f"{prob*100:.1f}%")
            cb.metric("Lifestyle-Adjusted Risk", f"{new_prob*100:.1f}%", delta=f"{diff*100:+.1f}%")
            st.metric("Estimated HbA1c", f"{hba_new:.2f}%", delta=f"{(hba_new-hba_base):+.2f}")

            lm = {"Exercise":sm[q1],"Sugar":sm[q2],"Veg/Fruit":sm[q3],
                  "Sleep":sm[q4],"Fast Food":sm[q5],"Stress":sm[q6]}
            fig_heat = px.bar(
                pd.DataFrame({"Factor":list(lm.keys()),"Score":list(lm.values())}),
                x="Factor", y="Score", color="Score",
                color_continuous_scale="RdYlGn",
                title="Your Lifestyle Score Breakdown (0=worst, 3=best)")
            fig_heat.update_layout(paper_bgcolor="#0f1724", plot_bgcolor="#0f1724",
                                   font=dict(color="white"), height=280, margin=dict(t=40,b=10))
            st.plotly_chart(fig_heat, use_container_width=True)

    else:
        st.info("Enter your Patient ID above to view your health record.")

# ================================================================
#  MAIN
# ================================================================
if not st.session_state.logged_in:
    show_login()
else:
    show_sidebar()
    role = st.session_state.role
    if role == "Admin":
        admin_dashboard()
    elif role == "Doctor":
        doctor_dashboard()
    elif role == "Patient":
        patient_portal()
    else:
        st.error("Unknown role — please log out and try again.")
