# app.py
"""
Smart EMR - Unified Streamlit app
- Single-file merged app for Admin / Doctor / Patient
- Uses pretrained model: diabetes_rf.pkl and scaler.pkl
- Make sure: diabetes_data.csv, diabetes_rf.pkl, scaler.pkl exist in same folder
- Webcam QR scanning only (no file upload)
"""

import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np
import joblib
import qrcode
import os
from datetime import datetime
from pyzbar import pyzbar
from PIL import Image
import base64
import plotly.graph_objects as go

# try importing cv2; gracefully handle if not available
try:
    import cv2
except Exception:
    cv2 = None
# ---------------- CONFIG ----------------
DATA_PATH   = "https://rxvrjhwmhjtipcqoxbyg.supabase.co/storage/v1/object/public/probation/diabetes_data.csv"
MODEL_PATH  = "diabetes_rf.pkl"
SCALER_PATH = "scaler.pkl"
QR_FOLDER   = "qrcodes"
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
# ---------------- UI STYLES -----------------
# ---------------- UI STYLES -----------------
st.set_page_config(page_title="GLUCO-LENS", page_icon="🔍➕", layout="wide")
st.markdown("""
<style>
/* Body & Fonts */
body { 
    background: #0f1724; 
    color: #e6eef8; 
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}

/* Main Header */
.main-header {
    background: linear-gradient(135deg,#0ea5e9,#7c3aed);
    padding: 22px 18px;
    border-radius: 18px;
    color: white;
    text-align: center;
    margin-bottom: 24px;
    font-size: 36px;
    font-weight: 800;
    box-shadow: 0 10px 24px rgba(0,0,0,0.5);
    display: flex;
    justify-content: center;
    align-items: center;
    gap: 16px;
}

/* Role Cards (hover effect) */
.role-card {
    background: rgba(255,255,255,0.03); 
    padding: 20px; 
    border-radius: 16px; 
    transition: 0.3s; 
    cursor: pointer;
    box-shadow: 0 6px 16px rgba(0,0,0,0.3);
    text-align: center;
}
.role-card:hover {
    background: rgba(255,255,255,0.12);
    transform: translateY(-6px);
    box-shadow: 0 10px 22px rgba(0,0,0,0.5);
}

/* Small text */
.small-muted {
    color:#9aa8c5; 
    font-size:13px;
}

/* Tables */
.futuristic-table th {
    background:linear-gradient(90deg,#16ffe5,#00BFFF); 
    color:white; 
    padding:14px; 
    font-weight:700;
    font-size:14px;
}
.futuristic-table td {
    color:white; 
    padding:12px;
}

/* Cards */
.card {
    background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01)); 
    border-radius:16px; 
    padding:18px; 
    box-shadow: 0 10px 24px rgba(2,6,23,0.7);
    transition: 0.25s;
}
.card:hover {
    box-shadow: 0 14px 32px rgba(2,6,23,0.9);
    transform: translateY(-3px);
}

/* Buttons - bigger & bold */
.stButton>button {
    background: linear-gradient(135deg,#7c3aed,#0ea5e9); 
    color:white; 
    border-radius:14px; 
    padding:14px 28px; 
    font-size:16px;
    font-weight:700;
    transition: 0.25s;
}
.stButton>button:hover {
    opacity:0.9;
    transform: translateY(-2px);
}

/* Title Icon */
.header-icon {
    width:48px;
    height:48px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER WITH ICON -----------------
st.markdown("""
<div class="main-header">
    <img class="header-icon" src="https://static.thenounproject.com/png/7938532-512.png" alt="gluco-lens-icon"/>
    GLUCO-LENS
</div>
""", unsafe_allow_html=True)



# ---------------- MODEL LOADING -----------------
@st.cache_resource
def load_model_and_scaler():
    try:
        model = joblib.load(MODEL_PATH)
    except Exception as e:
        st.warning(f"Could not load model: {e}")
        model = None
    try:
        scaler = joblib.load(SCALER_PATH) if model is not None else None
    except Exception as e:
        st.warning(f"Could not load scaler: {e}")
        scaler = None

    features = getattr(model, "feature_names_in_", DEFAULT_FEATURES)
    features = list(features) if features is not None else DEFAULT_FEATURES
    return model, scaler, features

MODEL, SCALER, FEATURES = load_model_and_scaler()

# ---------------- DATA HELPERS -----------------
@st.cache_data
def load_patient_db(path=DATA_PATH):
    try:
        df = pd.read_csv(path, dtype=str).fillna("")
    except Exception as e:
        st.warning(f"Failed to load patient DB: {e}")
        df = pd.DataFrame(columns=["PatientID"] + FEATURES + ["Name", "Diagnosis", "created_at", "DoctorRemarks"])
    
    # Ensure required columns exist
    for col in ["PatientID", "created_at", "DoctorRemarks", "Name"]:
        if col not in df.columns:
            df[col] = ""
    for f in FEATURES:
        if f not in df.columns:
            df[f] = ""
    return df

def save_patient_db(df, path=DATA_PATH):
    df.to_csv(path, index=False)

# Load the DB
patient_df = load_patient_db()

# ---------------- UTILS -----------------
def safe_float(x, default=0.0):
    try:
        if x is None or x == "" or pd.isna(x):
            return float(default)
        return float(x)
    except:
        return float(default)

def generate_qr(patient_id):
    out = os.path.join(QR_FOLDER, f"{patient_id}.png")
    if not os.path.exists(out):
        qr = qrcode.QRCode(version=1, box_size=6, border=2)
        qr.add_data(patient_id)
        qr.make(fit=True)
        img = qr.make_image(fill_color="black", back_color="white")
        img.save(out)
    return out

def decode_qr_imagefile(pil_image):
    try:
        arr = np.array(pil_image.convert("RGB"))
        decoded = pyzbar.decode(arr)
        if decoded:
            return decoded[0].data.decode("utf-8")
    except:
        return None
    return None

def scan_qr_live():
    """Live webcam QR scanner. Returns decoded string or None.
       Requires cv2 + pyzbar. Works for local deployments with a camera.
    """
    if cv2 is None:
        st.error("OpenCV (cv2) is not installed. Live scan unavailable.")
        return None

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Could not open webcam. Check camera permissions/connection.")
        return None

    st.info("📷 Scanning... show QR to the camera. Press 'q' in the camera window to cancel.")
    qr_data = None
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # convert to grayscale for faster decode (pyzbar handles color too)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            decoded_objs = pyzbar.decode(gray)
            for obj in decoded_objs:
                qr_data = obj.data.decode("utf-8")
                break
            # show frame window so user can align QR (this requires GUI environment)
            cv2.imshow("SmartEMR - QR Scanner (press q to cancel)", frame)
            if qr_data:
                break
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
    return qr_data

def get_patient_by_id(pid):
    df = load_patient_db()
    recs = df[df["PatientID"].astype(str) == str(pid)]
    if not recs.empty:
        return recs.iloc[0].to_dict()
    return None

def update_doctor_remarks(pid, remarks):
    df = load_patient_db()
    if pid in df["PatientID"].values:
        df.loc[df["PatientID"]==pid,"DoctorRemarks"] = remarks
        save_patient_db(df)

def predict_record_prob(record):
    """
    Returns (probability_float, status_string)
    status_string in {"ok","model_missing","heuristic_hba1c","failed"}
    """
    if MODEL is None or SCALER is None:
        return 0.0, "model_missing"

    x = []
    for f in FEATURES:
        raw = record.get(f, "")
        x.append(safe_float(raw, default=np.nan))

    df_db = load_patient_db()
    medians = []
    for i,f in enumerate(FEATURES):
        try:
            colvals = pd.to_numeric(df_db[f], errors="coerce")
            med = float(colvals.median()) if not colvals.dropna().empty else 0.0
        except:
            med = 0.0
        medians.append(med)
    for i,val in enumerate(x):
        if np.isnan(val):
            x[i] = medians[i]

    arr = np.array(x).reshape(1,-1)
    try:
        arr_scaled = SCALER.transform(arr)
        prob = float(MODEL.predict_proba(arr_scaled)[0][1])
        return prob, "ok"
    except:
        # fallback heuristic using HbA1c if available
        h = safe_float(record.get("HbA1c", np.nan))
        if not np.isnan(h):
            prob = 0.9 if h>=6.5 else (0.5 if h>=5.7 else 0.05)
            return prob, "heuristic_hba1c"
        return 0.0, "failed"

def simulate_future_risk(record, years=5):
    base_prob, _ = predict_record_prob(record)
    baseline = np.array([safe_float(record.get(f,0)) for f in FEATURES], dtype=float)
    projections = []
    for y in range(1, years+1):
        mod = baseline.copy()
        if "BMI" in FEATURES:
            mod[FEATURES.index("BMI")] += 0.3*y
        if "PhysicalActivity" in FEATURES:
            idx = FEATURES.index("PhysicalActivity")
            mod[idx] = max(0, mod[idx]-0.5*y)
        if "FastingBloodSugar" in FEATURES:
            idx = FEATURES.index("FastingBloodSugar")
            mod[idx] += 2*y
        try:
            pred = float(MODEL.predict_proba(SCALER.transform(mod.reshape(1,-1)))[0][1])
        except:
            pred = min(0.99, base_prob + 0.03*y)
        projections.append(round(pred*100,2))
    return projections

def human_ai_summary(record, prob):
    """
    Return a human-friendly AI summary string based on key metrics and probability.
    """
    age = safe_float(record.get("Age","N/A"))
    hbA1c = safe_float(record.get("HbA1c","N/A"))
    bmi = safe_float(record.get("BMI","N/A"))
    sbp = safe_float(record.get("SystolicBP","N/A"))
    dbp = safe_float(record.get("DiastolicBP","N/A"))
    chol_total = safe_float(record.get("CholesterolTotal","N/A"))

    hbA1c_status = "normal" if hbA1c<5.7 else ("prediabetes" if hbA1c<6.5 else "diabetes-range")
    bmi_status = "healthy" if bmi<25 else ("overweight" if bmi<30 else "obese")
    bp_status = "normal" if sbp<120 and dbp<80 else ("elevated" if sbp<130 and dbp<80 else "hypertensive")

    if prob < 0.35:
        tone = ("Low risk. Patient's numbers look largely okay — "
                f"HbA1c {hbA1c:.1f}% ({hbA1c_status}), BMI {bmi:.1f} ({bmi_status}). "
                "Recommend maintaining current lifestyle, yearly check-ups.")
    elif prob < 0.65:
        tone = ("Moderate risk. There are areas to watch — "
                f"HbA1c {hbA1c:.1f}% ({hbA1c_status}), BMI {bmi:.1f}. "
                "Advise lifestyle tweaks (diet, activity), closer monitoring, follow-up in 3-6 months.")
    else:
        tone = ("High risk. Findings suggest elevated risk — "
                f"HbA1c {hbA1c:.1f}% ({hbA1c_status}), BMI {bmi:.1f}. "
                "Recommend urgent clinical follow-up, medication review and targeted interventions.")

    proj = simulate_future_risk(record)
    proj_text = f"Projected risk (%) over 5 years: {', '.join([str(p)+'%' for p in proj])}."

    return tone + " " + proj_text

def format_val(v):
    if v is None or v=="":
        return "N/A"
    try:
        return f"{float(v):.2f}"
    except:
        return str(v)

def download_link_df(df, name="data.csv"):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{name}">⬇️ Download CSV</a>'
    return href

# ---------------- DASHBOARDS -----------------
def admin_dashboard():
    st.markdown('<div class="main-header"><h1>👑 Admin Dashboard</h1></div>', unsafe_allow_html=True)
    st.markdown('<div class="card">', unsafe_allow_html=True)
    df = load_patient_db()
    tab1, tab2, tab3 = st.tabs(["View & Export","Add Patient","Bulk/Utilities"])
    with tab1:
        st.write("Total records:", len(df))
        st.markdown(download_link_df(df, "patients_export.csv"), unsafe_allow_html=True)
        st.dataframe(df,use_container_width=True)
    with tab2:
        st.subheader("Add New Patient")
        with st.form("add_patient_form"):
            new = {}
            new["PatientID"] = st.text_input("PatientID", value=f"P{len(df)+1:05d}")
            new["Name"] = st.text_input("Name", value="")
            for f in FEATURES:
                new[f] = st.number_input(f,value=0.0,key=f"add_{f}")
            new["Diagnosis"] = st.text_input("Diagnosis")
            if st.form_submit_button("Create Patient"):
                if df[df["PatientID"]==new["PatientID"]].empty:
                    new["created_at"] = datetime.utcnow().isoformat()
                    new["DoctorRemarks"] = ""
                    df = pd.concat([df,pd.DataFrame([new])],ignore_index=True)
                    save_patient_db(df)
                    qr = generate_qr(new["PatientID"])
                    st.success(f"Created patient {new['PatientID']}")
                    st.image(qr,width=160)
                else:
                    st.error("PatientID exists.")
    with tab3:
        st.subheader("Bulk utilities")
        uploaded = st.file_uploader("Upload CSV to append", type=["csv"])
        if uploaded:
            try:
                add_df = pd.read_csv(uploaded,dtype=str).fillna("")
                df = pd.concat([df, add_df], ignore_index=True)
                save_patient_db(df)
                st.success("Appended CSV")
            except Exception as e:
                st.error(f"Failed: {e}")
        if st.button("Regenerate all QR codes"):
            for pid in df["PatientID"]:
                generate_qr(pid)
            st.success("QR codes regenerated")
    st.markdown('</div>', unsafe_allow_html=True)

def doctor_dashboard():
    st.markdown('<div class="main-header"><h1>👩‍⚕️ Doctor Dashboard</h1></div>', unsafe_allow_html=True)
    st.markdown('<div class="card">', unsafe_allow_html=True)
    if st.session_state.get("role") != "Doctor":
        st.info("Please log in as a Doctor to view patient profiles.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    st.markdown("### Options")
    col1, col2 = st.columns(2)
    with col1:
        pid_input = st.text_input("Enter Patient ID (fallback)")
        if st.button("Fetch Patient by ID"):
            rec = get_patient_by_id(pid_input.strip())
            if rec:
                st.session_state["current_record_doc"] = rec
                st.session_state["current_pid_doc"] = pid_input.strip()
            else:
                st.error("Not found")
    with col2:
        if st.button("📷 Scan Patient QR (Webcam)"):
            pid = scan_qr_live()
            if pid:
                st.success(f"Scanned QR: {pid}")
                rec = get_patient_by_id(pid)
                if rec:
                    st.session_state["current_record_doc"] = rec
                    st.session_state["current_pid_doc"] = pid
                else:
                    st.error("No record found for this patient.")
            else:
                st.error("No QR detected or webcam not available.")

    pid = st.session_state.get("current_pid_doc")
    patient = st.session_state.get("current_record_doc")
    if patient:
        st.markdown(f"<h2 style='color:white'>Patient: {patient.get('Name', pid)} — {pid}</h2>", unsafe_allow_html=True)

        # AI summary + risk
        prob, status = predict_record_prob(patient)
        risk_text = "Low ✅" if prob<0.35 else ("Moderate ⚠️" if prob<0.65 else "High ❌")
        st.markdown(f"**AI Risk:** {risk_text} (prob: {prob*100:.1f}%)")
        st.info(human_ai_summary(patient, prob))

        # Key vitals
        vitals = {f: patient.get(f,"") for f in ["FastingBloodSugar","HbA1c","CholesterolTotal","BMI","SystolicBP","DiastolicBP"] if f in patient}
        cols = st.columns(len(vitals) if vitals else 1)
        for i,(k,v) in enumerate(vitals.items()):
            cols[i].metric(label=k,value=format_val(v))

        # Ring chart
        ring = go.Figure(go.Pie(values=[prob*100,100-prob*100], hole=0.7,
                                marker_colors=["#ff3b30" if prob>0.6 else ("#ff9f0a" if prob>0.3 else "#34c759"), "#111218"],textinfo="none"))
        ring.update_layout(annotations=[dict(text=f"<b>{prob*100:.1f}%</b>", showarrow=False,font=dict(color="white"))],
                           paper_bgcolor="#0f1724", plot_bgcolor="#0f1724")
        st.plotly_chart(ring,use_container_width=True)

        proj = simulate_future_risk(patient)
        proj_fig = go.Figure()
        proj_fig.add_trace(go.Scatter(x=list(range(1,6)),y=proj,mode="lines+markers",name="Risk %"))
        proj_fig.update_layout(title="5-year projection (%)",yaxis=dict(range=[0,100]),
                               plot_bgcolor="#0f1724",paper_bgcolor="#0f1724",font=dict(color="white"))
        st.plotly_chart(proj_fig,use_container_width=True)

        st.markdown("### Full Record")
        table_rows = [(k,format_val(v)) for k,v in patient.items() if k!="created_at"]
        st.table(pd.DataFrame(table_rows, columns=["Feature","Value"]))

        # Doctor can edit/save remarks
        note = st.text_area("Doctor Remarks", value=patient.get("DoctorRemarks",""))
        if st.button("Save Note"):
            update_doctor_remarks(pid,note)
            st.success("Saved ✅")
    else:
        st.info("No patient loaded. Use Scan (Webcam) or fetch by ID above.")
    st.markdown('</div>', unsafe_allow_html=True)

def patient_dashboard():
    st.markdown('<div class="main-header"><h1>🫀 Patient Portal — Futuristic View</h1></div>', unsafe_allow_html=True)
    st.markdown('<div class="card">', unsafe_allow_html=True)
    
    if "current_record" not in st.session_state:
        st.session_state["current_record"] = None
        st.session_state["current_pid"] = None

    st.subheader("📷 Scan QR Code via Webcam (no uploads)")
    if st.button("Start Live QR Scan"):
        pid = scan_qr_live()
        if pid:
            st.success(f"Scanned QR: {pid}")
            st.session_state["current_pid"] = pid
            st.session_state["current_record"] = get_patient_by_id(pid)
            if st.session_state["current_record"] is None:
                st.error("Patient not found")
        else:
            st.error("No QR detected or webcam unavailable. Try again.")

    record = st.session_state.get("current_record")
    pid = st.session_state.get("current_pid")
    
    if record:
        st.markdown(f"<h2 style='color:white'>Patient: {pid}</h2>", unsafe_allow_html=True)

        # --- Key Vitals ---
        vitals = {f: record.get(f,"") for f in ["FastingBloodSugar","HbA1c","CholesterolTotal","BMI","SystolicBP","DiastolicBP"] if f in record}
        cols = st.columns(len(vitals) if vitals else 1)
        for i,(k,v) in enumerate(vitals.items()):
            cols[i].metric(label=k,value=format_val(v))

        # --- AI Risk ---
        prob,_ = predict_record_prob(record)
        ring = go.Figure(go.Pie(values=[prob*100,100-prob*100], hole=0.7,
                                marker_colors=["#ff3b30" if prob>0.6 else ("#ff9f0a" if prob>0.3 else "#34c759"), "#111218"],textinfo="none"))
        ring.update_layout(annotations=[dict(text=f"<b>{prob*100:.1f}%</b>", showarrow=False,font=dict(color="white"))],
                        paper_bgcolor="#0f1724", plot_bgcolor="#0f1724")
        st.plotly_chart(ring,use_container_width=True)

        st.markdown("### 🤖 AI Summary — Quick Overview")
        st.info(human_ai_summary(record, prob))

        proj = simulate_future_risk(record)
        proj_fig = go.Figure()
        proj_fig.add_trace(go.Scatter(x=list(range(1,6)),y=proj,mode="lines+markers"))
        proj_fig.update_layout(title="5-year projection (%)",yaxis=dict(range=[0,100]),
                               plot_bgcolor="#0f1724",paper_bgcolor="#0f1724",font=dict(color="white"))
        st.plotly_chart(proj_fig,use_container_width=True)

        # --- Full Table ---
        table_rows = [(k,format_val(v)) for k,v in record.items() if k!="created_at"]
        st.table(pd.DataFrame(table_rows, columns=["Feature","Value"]))

        # --- Doctor's Remarks ---
        remarks = record.get("DoctorRemarks","")
        if remarks:
            st.markdown("### 📝 Doctor's Remarks")
            st.text_area("Remarks", value=remarks, height=120, disabled=True)

        if st.button("Download Patient CSV"):
            single_df = pd.DataFrame([record])
            st.download_button("Download CSV", single_df.to_csv(index=False), file_name=f"{pid}.csv", mime="text/csv")

        # ================= LIFESTYLE QUESTIONNAIRE =================
        st.markdown("### 📝 Lifestyle Questionnaire")
        st.info("Answer the following questions to see how lifestyle changes affect your diabetes risk.")

        with st.form("lifestyle_form"):
            q1 = st.select_slider("1. Average minutes of moderate to vigorous exercise per day?", ["0","10-30","30-60","60+"], value="30-60")
            q2 = st.select_slider("2. Daily intake of added sugar (teaspoons)?", ["0-5","6-10","11-20","20+"], value="6-10")
            q3 = st.select_slider("3. Average daily servings of vegetables/fruits?", ["0-1","2-3","4-5","5+"], value="2-3")
            q4 = st.select_slider("4. Average hours of sleep per night?", ["<5","5-6","6-7","7+"], value="6-7")
            q5 = st.select_slider("5. Frequency of processed/fast food consumption?", ["Daily","Few times/week","Once a week","Rarely/Never"], value="Few times/week")
            q6 = st.select_slider("6. Average daily stress level (0-10)?", ["0-2","3-5","6-8","9-10"], value="3-5")

            submitted = st.form_submit_button("Calculate Lifestyle-Adjusted Risk")

        if submitted and record:
            # Map answers to numeric score
            scores = {
                "0":0,"10-30":1,"30-60":2,"60+":3,
                "0-5":3,"6-10":2,"11-20":1,"20+":0,
                "0-1":0,"2-3":1,"4-5":2,"5+":3,
                "<5":0,"5-6":1,"6-7":2,"7+":3,
                "Daily":0,"Few times/week":1,"Once a week":2,"Rarely/Never":3,
                "0-2":3,"3-5":2,"6-8":1,"9-10":0
            }

            total_score = scores[q1] + scores[q2] + scores[q3] + scores[q4] + scores[q5] + scores[q6]
            max_score = 3*6
            lifestyle_modifier = (total_score - max_score/2) / max_score  # range -0.5 to +0.5

            prob_new = min(max(prob - lifestyle_modifier*0.2, 0),1)
            diff = prob_new - prob
            trend_text = "declining" if diff<0 else ("rising" if diff>0 else "stable")
            st.markdown(f"**Previous AI Risk:** {prob*100:.1f}% → **Lifestyle-adjusted Risk:** {prob_new*100:.1f}% ({trend_text}, Δ={diff*100:.1f}%)")

            hbA1c_base = safe_float(record.get("HbA1c",5.5))
            hbA1c_new = hbA1c_base + diff*2
            st.markdown(f"**Estimated HbA1c:** {hbA1c_new:.2f}% (previous: {hbA1c_base:.2f}%)")

            # ================= HbA1c vs Lifestyle Heatmap =================
            lifestyle_metrics = {
                "Exercise": scores[q1],
                "Added Sugar": scores[q2],
                "Veg/Fruit Intake": scores[q3],
                "Sleep": scores[q4],
                "Fast Food": scores[q5],
                "Stress": scores[q6]
            }

            df_heatmap = pd.DataFrame({
                "Metric": list(lifestyle_metrics.keys()) + ["HbA1c"],
                "Value": list(lifestyle_metrics.values()) + [hbA1c_new]
            }).set_index("Metric").T

            fig_hb = px.imshow(df_heatmap, text_auto=True, color_continuous_scale='Blues')
            fig_hb.update_layout(
                title="HbA1c vs Lifestyle Metrics",
                plot_bgcolor="#0f1724",
                paper_bgcolor="#0f1724",
                font=dict(color="white")
            )
            st.plotly_chart(fig_hb, use_container_width=True)

    else:
        st.info("Start the live QR scan to open your record.")

    st.markdown('</div>', unsafe_allow_html=True)


# ---------------- AUTH / LOGIN -----------------
def login_form_for_role(selected_role):
    st.markdown(f"### Login as {selected_role}")
    username = st.text_input("Username", key=f"user_{selected_role}")
    password = st.text_input("Password", type="password", key=f"pass_{selected_role}")
    st.markdown("**Demo credentials:** admin/admin123, doctor/doc123")
    if st.button("Login", key=f"login_{selected_role}"):
        creds = {"admin":"admin123","doctor":"doc123"}
        roles = {"admin":"Admin","doctor":"Doctor"}
        if username.lower() in creds and password==creds[username.lower()]:
            st.session_state["role"] = roles[username.lower()]
            st.success(f"Logged in as {roles[username.lower()]} ✅")
            if "selected_role" in st.session_state:
                del st.session_state["selected_role"]
            st.rerun()
        else:
            st.error("Invalid credentials")

# ---------------- MAIN -----------------
def main():
    # session keys setup
    if "role" not in st.session_state:
        st.session_state["role"] = None
    if "selected_role" not in st.session_state:
        st.session_state["selected_role"] = None

    # role selection screen (minimal — patient first-run experience simplified)
    if st.session_state["role"] is None and st.session_state["selected_role"] is None:
        st.markdown("<h2 style='color:white'>Select your role:</h2>", unsafe_allow_html=True)
        col1,col2,col3 = st.columns(3)
        with col1:
            if st.button("👑 Admin"):
                st.session_state["selected_role"] = "Admin"
        with col2:
            if st.button("👩‍⚕️ Doctor"):
                st.session_state["selected_role"] = "Doctor"
        with col3:
            if st.button("🫀 Patient"):
                st.session_state["selected_role"] = "Patient"
        st.markdown("<br><hr><div style='color:#9aa8c5'>Demo: admin/admin123, doctor/doc123</div>", unsafe_allow_html=True)
        return

    # if a role was selected but not yet logged in
    if st.session_state["role"] is None and st.session_state["selected_role"] is not None:
        sel = st.session_state["selected_role"]
        if sel in ["Admin","Doctor"]:
            login_form_for_role(sel)
            st.markdown("<hr>")
            st.info("Need to login to proceed. If you already logged in, refresh may be required.")
            return
        elif sel == "Patient":
            # patient doesn't need login
            patient_dashboard()
            return

    # now role is set (logged in) OR patient selected
    st.sidebar.markdown(f"### Role: {st.session_state['role'] if st.session_state['role'] else st.session_state['selected_role']}")
    if st.sidebar.button("Logout"):
        st.session_state["role"] = None
        st.session_state["selected_role"] = None
        st.rerun()

    role = st.session_state.get("role")
    if role == "Admin":
        admin_dashboard()
    elif role == "Doctor":
        doctor_dashboard()
    else:
        # fallback: patient view if selected_role was patient
        if st.session_state.get("selected_role") == "Patient":
            patient_dashboard()
        else:
            st.error("Unknown role. Logging out.")
            st.session_state["role"] = None
            st.session_state["selected_role"] = None

if __name__=="__main__":
    main()