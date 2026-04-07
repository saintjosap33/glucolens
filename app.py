"""
GLUCO-LENS v3 — Smart Diabetes EMR
- Supabase DB for persistent patient storage
- Auto-trains RandomForest if .pkl missing
- Premium dark glassmorphism UI
- Roles: Admin / Doctor / Patient
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import joblib
import qrcode
import os
import io
import base64
from datetime import datetime
from supabase import create_client, Client


# ── Supabase ──────────────────────────────────────────────────────────────────
try:
    from supabase import create_client, Client
    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False

# ── ML ────────────────────────────────────────────────────────────────────────
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ═══════════════════════════════════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════════════════════════════════
MODEL_PATH  = "diabetes_rf.pkl"
SCALER_PATH = "scaler.pkl"
QR_FOLDER   = "/tmp/qrcodes"
os.makedirs(QR_FOLDER, exist_ok=True)

# Load secrets safely
SUPABASE_URL = ""
SUPABASE_KEY = ""

try:
    SUPABASE_URL = st.secrets["SUPABASE_URL"]
    SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
except Exception:
    pass  # fallback to empty if not set

TABLE_NAME = "patients"

@st.cache_resource
def get_supabase() -> "Client | None":
    """Create and cache Supabase client"""
    if not SUPABASE_URL or not SUPABASE_KEY:
        return None
    try:
        client = create_client(SUPABASE_URL, SUPABASE_KEY)
        return client
    except Exception as e:
        st.sidebar.error(f"Supabase connection failed: {e}")
        return None
CREDENTIALS = {
    "admin":   {"password": "admin123", "role": "Admin"},
    "doctor":  {"password": "doc123",   "role": "Doctor"},
    "patient": {"password": "pat123",   "role": "Patient"},
}

ML_FEATURES = [
    "Age","Gender","BMI","Smoking","AlcoholConsumption","PhysicalActivity",
    "DietQuality","SleepQuality","FamilyHistoryDiabetes","GestationalDiabetes",
    "PolycysticOvarySyndrome","PreviousPreDiabetes","Hypertension",
    "SystolicBP","DiastolicBP","FastingBloodSugar","HbA1c",
    "SerumCreatinine","BUNLevels","CholesterolTotal","CholesterolLDL",
    "CholesterolHDL","CholesterolTriglycerides","AntihypertensiveMedications",
    "Statins","AntidiabeticMedications","FrequentUrination","ExcessiveThirst",
    "UnexplainedWeightLoss","FatigueLevels","BlurredVision","SlowHealingSores",
    "TinglingHandsFeet","QualityOfLifeScore","MedicalCheckupsFrequency",
    "MedicationAdherence","HealthLiteracy"
]

# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE CONFIG & CSS
# ═══════════════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="GLUCO-LENS", page_icon="🩺", layout="wide",
                   initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; }

html, body, [data-testid="stAppViewContainer"] {
    background: #060912 !important;
    color: #e2e8f5 !important;
    font-family: 'Outfit', sans-serif !important;
}

[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(ellipse 80% 50% at 20% -10%, rgba(99,102,241,0.18) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 80% 110%, rgba(14,165,233,0.14) 0%, transparent 60%),
        #060912 !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: rgba(10,12,28,0.95) !important;
    border-right: 1px solid rgba(99,102,241,0.2) !important;
    backdrop-filter: blur(20px) !important;
}
[data-testid="stSidebar"] * { color: #c7d2fe !important; font-family:'Outfit',sans-serif !important; }

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stDecoration"] { display:none; }

/* ── Glass card ── */
.glass {
    background: rgba(255,255,255,0.035);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 20px;
    padding: 24px;
    backdrop-filter: blur(12px);
    margin-bottom: 20px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.06);
}

/* ── Top header bar ── */
.topbar {
    background: linear-gradient(135deg, rgba(99,102,241,0.25), rgba(14,165,233,0.2));
    border: 1px solid rgba(99,102,241,0.3);
    border-radius: 20px;
    padding: 20px 32px;
    margin-bottom: 28px;
    display: flex;
    align-items: center;
    gap: 16px;
    box-shadow: 0 0 60px rgba(99,102,241,0.15), inset 0 1px 0 rgba(255,255,255,0.1);
}
.topbar-title {
    font-size: 28px;
    font-weight: 800;
    letter-spacing: 3px;
    background: linear-gradient(135deg, #a5b4fc, #38bdf8, #a5b4fc);
    background-size: 200% auto;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: shimmer 3s linear infinite;
}
.topbar-sub { font-size: 13px; color: #94a3b8; letter-spacing: 1px; margin-top:2px; }

@keyframes shimmer { 0%{background-position:0%} 100%{background-position:200%} }

/* ── Stat card ── */
.stat-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 16px;
    padding: 20px;
    text-align: center;
    transition: transform 0.2s, box-shadow 0.2s;
}
.stat-card:hover { transform: translateY(-3px); box-shadow: 0 12px 40px rgba(0,0,0,0.5); }
.stat-number { font-size: 32px; font-weight: 800; line-height: 1; }
.stat-label  { font-size: 12px; color: #64748b; text-transform: uppercase; letter-spacing: 1.5px; margin-top: 6px; }

/* ── Risk badge ── */
.risk-high { color: #f87171; }
.risk-mid  { color: #fbbf24; }
.risk-low  { color: #34d399; }

/* ── Inputs ── */
.stTextInput input, .stNumberInput input, .stTextArea textarea, .stSelectbox select {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(99,102,241,0.25) !important;
    border-radius: 12px !important;
    color: #e2e8f5 !important;
    font-family: 'Outfit', sans-serif !important;
    font-size: 14px !important;
    transition: border-color 0.2s !important;
}
.stTextInput input:focus, .stNumberInput input:focus {
    border-color: rgba(99,102,241,0.6) !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,0.12) !important;
}
.stTextInput label, .stNumberInput label, .stTextArea label, .stSelectbox label,
.stSlider label, .stRadio label {
    color: #94a3b8 !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    letter-spacing: 0.5px !important;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #6366f1, #0ea5e9) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 12px 28px !important;
    font-family: 'Outfit', sans-serif !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    letter-spacing: 0.5px !important;
    transition: opacity 0.2s, transform 0.2s !important;
    box-shadow: 0 4px 20px rgba(99,102,241,0.35) !important;
}
.stButton > button:hover { opacity: 0.88 !important; transform: translateY(-1px) !important; }
.stButton > button:active { transform: translateY(0) !important; }

/* ── Download button ── */
.stDownloadButton > button {
    background: rgba(99,102,241,0.15) !important;
    border: 1px solid rgba(99,102,241,0.35) !important;
    color: #a5b4fc !important;
    border-radius: 12px !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(255,255,255,0.03) !important;
    border-radius: 12px !important;
    padding: 4px !important;
    gap: 4px !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px !important;
    color: #64748b !important;
    font-weight: 500 !important;
    font-family: 'Outfit', sans-serif !important;
}
.stTabs [aria-selected="true"] {
    background: rgba(99,102,241,0.2) !important;
    color: #a5b4fc !important;
}

/* ── Metrics ── */
[data-testid="metric-container"] {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 14px !important;
    padding: 16px !important;
}
[data-testid="metric-container"] label { color: #64748b !important; font-size:12px !important; }
[data-testid="metric-container"] [data-testid="stMetricValue"] { color: #e2e8f5 !important; font-weight:700 !important; }

/* ── Tables ── */
.stDataFrame { border-radius: 14px !important; overflow: hidden !important; }
[data-testid="stDataFrame"] { background: rgba(255,255,255,0.02) !important; border-radius: 14px !important; }

/* ── Alerts ── */
.stSuccess, .stInfo, .stWarning, .stError {
    border-radius: 12px !important;
    font-family: 'Outfit', sans-serif !important;
}

/* ── Divider ── */
hr { border-color: rgba(255,255,255,0.06) !important; }

/* ── Login ── */
.login-wrap {
    min-height: 80vh;
    display: flex;
    align-items: center;
    justify-content: center;
}
.login-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(99,102,241,0.25);
    border-radius: 24px;
    padding: 48px 44px;
    width: 100%;
    max-width: 440px;
    box-shadow: 0 24px 80px rgba(0,0,0,0.6), 0 0 80px rgba(99,102,241,0.08);
}
.login-logo {
    font-size: 42px;
    font-weight: 800;
    letter-spacing: 4px;
    background: linear-gradient(135deg, #a5b4fc 0%, #38bdf8 50%, #a5b4fc 100%);
    background-size: 200% auto;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: shimmer 3s linear infinite;
    text-align: center;
    margin-bottom: 4px;
}
.login-sub {
    text-align: center;
    color: #475569;
    font-size: 13px;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 32px;
}
.cred-badge {
    background: rgba(99,102,241,0.1);
    border: 1px solid rgba(99,102,241,0.2);
    border-radius: 10px;
    padding: 12px 16px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    color: #94a3b8;
    margin-top: 20px;
    line-height: 1.8;
}

/* ── Section heading ── */
.sec-head {
    font-size: 18px;
    font-weight: 700;
    color: #c7d2fe;
    letter-spacing: 0.5px;
    margin-bottom: 16px;
    padding-bottom: 10px;
    border-bottom: 1px solid rgba(255,255,255,0.07);
}

/* ── AI summary box ── */
.ai-box {
    background: linear-gradient(135deg, rgba(99,102,241,0.08), rgba(14,165,233,0.06));
    border: 1px solid rgba(99,102,241,0.2);
    border-left: 3px solid #6366f1;
    border-radius: 14px;
    padding: 18px 20px;
    font-size: 14px;
    line-height: 1.7;
    color: #cbd5e1;
}

/* ── Sidebar nav ── */
.sb-role {
    background: rgba(99,102,241,0.12);
    border: 1px solid rgba(99,102,241,0.2);
    border-radius: 10px;
    padding: 12px 16px;
    margin-bottom: 12px;
}
.sb-role .role-name { font-size:18px; font-weight:700; color:#a5b4fc; }
.sb-role .role-user { font-size:12px; color:#64748b; margin-top:2px; }

/* ── Progress bar ── */
.prog-wrap { margin:8px 0; }
.prog-label { font-size:12px; color:#64748b; margin-bottom:4px; display:flex; justify-content:space-between; }
.prog-bar { height:6px; background:rgba(255,255,255,0.06); border-radius:99px; overflow:hidden; }
.prog-fill { height:100%; border-radius:99px; transition: width 0.6s cubic-bezier(.4,0,.2,1); }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
#  SESSION STATE
# ═══════════════════════════════════════════════════════════════════════════════
_defaults = dict(logged_in=False, role=None, username=None,
                 doc_record=None, doc_pid=None,
                 pat_record=None, pat_pid=None)
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ═══════════════════════════════════════════════════════════════════════════════
#  SUPABASE CLIENT
# ═══════════════════════════════════════════════════════════════════════════════
@st.cache_resource
def get_supabase() -> "Client | None":
    if not SUPABASE_AVAILABLE or not SUPABASE_URL or not SUPABASE_KEY:
        return None
    try:
        return create_client(SUPABASE_URL, SUPABASE_KEY)
    except Exception:
        return None

def db_fetch_all() -> pd.DataFrame:
    """Fetch all patients from Supabase. Falls back to default demo data."""
    sb = get_supabase()
    if sb:
        try:
            res = sb.table(TABLE_NAME).select("*").execute()
            if res.data:
                return pd.DataFrame(res.data)
        except Exception as e:
            st.sidebar.caption(f"⚠️ DB fetch error: {e}")
    return _demo_df()

def db_upsert(row: dict) -> bool:
    """Insert or update a patient row. Returns True on success."""
    sb = get_supabase()
    if sb:
        try:
            sb.table(TABLE_NAME).upsert(row, on_conflict="patient_id").execute()
            return True
        except Exception as e:
            st.error(f"DB write error: {e}")
    return False

def db_delete(pid: str) -> bool:
    sb = get_supabase()
    if sb:
        try:
            sb.table(TABLE_NAME).delete().eq("patient_id", pid).execute()
            return True
        except Exception as e:
            st.error(f"DB delete error: {e}")
    return False

def _demo_df() -> pd.DataFrame:
    return pd.DataFrame([
        {"patient_id":"P001","name":"Arun Kumar",   "age":52,"gender":1,"bmi":29.4,
         "hba1c":7.2,"fasting_blood_sugar":136,"cholesterol_total":210,
         "systolic_bp":138,"diastolic_bp":88,"hypertension":1,
         "family_history_diabetes":1,"smoking":0,"physical_activity":2,
         "diet_quality":2,"sleep_quality":2,"diagnosis":"Pre-diabetic","doctor_remarks":"",
         "created_at":datetime.utcnow().isoformat()},
        {"patient_id":"P002","name":"Meena Selvam", "age":45,"gender":0,"bmi":27.1,
         "hba1c":6.1,"fasting_blood_sugar":112,"cholesterol_total":188,
         "systolic_bp":122,"diastolic_bp":80,"hypertension":0,
         "family_history_diabetes":0,"smoking":0,"physical_activity":4,
         "diet_quality":3,"sleep_quality":3,"diagnosis":"Normal","doctor_remarks":"",
         "created_at":datetime.utcnow().isoformat()},
        {"patient_id":"P003","name":"Ravi Shankar", "age":61,"gender":1,"bmi":33.2,
         "hba1c":8.4,"fasting_blood_sugar":162,"cholesterol_total":235,
         "systolic_bp":148,"diastolic_bp":94,"hypertension":1,
         "family_history_diabetes":1,"smoking":1,"physical_activity":1,
         "diet_quality":1,"sleep_quality":1,"diagnosis":"Diabetic",
         "doctor_remarks":"On Metformin 500mg","created_at":datetime.utcnow().isoformat()},
        {"patient_id":"P004","name":"Priya Nair",   "age":38,"gender":0,"bmi":24.8,
         "hba1c":5.6,"fasting_blood_sugar":95, "cholesterol_total":172,
         "systolic_bp":116,"diastolic_bp":76,"hypertension":0,
         "family_history_diabetes":0,"smoking":0,"physical_activity":5,
         "diet_quality":4,"sleep_quality":4,"diagnosis":"Normal","doctor_remarks":"",
         "created_at":datetime.utcnow().isoformat()},
        {"patient_id":"P005","name":"Suresh Babu",  "age":57,"gender":1,"bmi":31.0,
         "hba1c":7.8,"fasting_blood_sugar":148,"cholesterol_total":220,
         "systolic_bp":142,"diastolic_bp":90,"hypertension":1,
         "family_history_diabetes":1,"smoking":1,"physical_activity":2,
         "diet_quality":2,"sleep_quality":2,"diagnosis":"Diabetic",
         "doctor_remarks":"Follow-up in 3 months","created_at":datetime.utcnow().isoformat()},
    ])

# ═══════════════════════════════════════════════════════════════════════════════
#  ML — AUTO-TRAIN IF MISSING
# ═══════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner="🧠 Loading / training ML model…")
def load_or_train_model():
    """
    1. Try loading from .pkl files
    2. If missing, synthesise training data and train a RandomForest
    Returns (model, scaler, features, accuracy, source)
    """
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        try:
            model  = joblib.load(MODEL_PATH)
            scaler = joblib.load(SCALER_PATH)
            feats  = list(getattr(model, "feature_names_in_", ML_FEATURES))
            return model, scaler, feats, None, "loaded"
        except Exception:
            pass

    # ── synthesise realistic training data ──────────────────────────────────
    np.random.seed(42)
    N = 8000
    age = np.random.normal(50, 15, N).clip(18, 90)
    bmi = np.random.normal(27, 6,  N).clip(15, 55)
    hba = np.random.normal(5.8, 1.2, N).clip(4, 14)
    fbs = np.random.normal(110, 35, N).clip(60, 400)
    sbp = np.random.normal(128, 20, N).clip(80, 220)
    dbp = np.random.normal(82,  12, N).clip(50, 130)
    chol= np.random.normal(195, 40, N).clip(100, 400)
    ldl = chol * np.random.uniform(0.45,0.65, N)
    hdl = np.random.normal(50, 12, N).clip(20, 100)
    tg  = np.random.normal(150, 60, N).clip(50, 500)
    creat=np.random.normal(0.95,0.25,N).clip(0.4, 3)
    bun = np.random.normal(16, 5, N).clip(5, 60)
    smoke  = np.random.binomial(1, 0.22, N)
    alc    = np.random.uniform(0,5,N)
    pa     = np.random.uniform(0,5,N)
    diet   = np.random.uniform(0,5,N)
    sleep  = np.random.uniform(4,9,N)
    fam    = np.random.binomial(1, 0.30, N)
    gest   = np.random.binomial(1, 0.08, N)
    pcos   = np.random.binomial(1, 0.07, N)
    prev   = np.random.binomial(1, 0.18, N)
    hyp    = np.random.binomial(1, 0.30, N)
    antihyp= np.random.binomial(1, 0.25, N)
    statins= np.random.binomial(1, 0.20, N)
    antidiab=np.random.binomial(1, 0.15, N)
    freq_ur= np.random.binomial(1, 0.20, N)
    ex_thr = np.random.binomial(1, 0.15, N)
    wt_loss= np.random.binomial(1, 0.12, N)
    fatigue= np.random.uniform(0,5,N)
    blur   = np.random.binomial(1, 0.12, N)
    sores  = np.random.binomial(1, 0.10, N)
    tingle = np.random.binomial(1, 0.14, N)
    qol    = np.random.uniform(0,100,N)
    mcu    = np.random.uniform(0,5,N)
    madh   = np.random.uniform(0,5,N)
    hlitt  = np.random.uniform(0,5,N)
    gender = np.random.binomial(1, 0.50, N)

    # risk score → label
    score = (
        (hba  - 4.5) * 0.40 +
        (fbs  - 70)  * 0.0025 +
        (bmi  - 18)  * 0.035 +
        (age  - 18)  * 0.010 +
        (sbp  - 90)  * 0.008 +
        fam          * 0.55 +
        smoke        * 0.30 +
        prev         * 0.50 +
        hyp          * 0.25 +
        gest         * 0.45 +
        pcos         * 0.40 -
        pa           * 0.08 -
        diet         * 0.06
    )
    prob_true = 1 / (1 + np.exp(-(score - 3.5)))
    label = (np.random.uniform(0,1,N) < prob_true).astype(int)

    X = np.column_stack([
        age, gender, bmi, smoke, alc, pa, diet, sleep,
        fam, gest, pcos, prev, hyp, sbp, dbp, fbs, hba,
        creat, bun, chol, ldl, hdl, tg,
        antihyp, statins, antidiab,
        freq_ur, ex_thr, wt_loss, fatigue, blur, sores, tingle,
        qol, mcu, madh, hlitt
    ])

    X_tr, X_te, y_tr, y_te = train_test_split(X, label, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    model = RandomForestClassifier(n_estimators=300, max_depth=12,
                                   min_samples_leaf=5, n_jobs=-1, random_state=42)
    model.fit(X_tr_s, y_tr)
    acc = accuracy_score(y_te, model.predict(X_te_s))

    model.feature_names_in_ = np.array(ML_FEATURES)

    try:
        joblib.dump(model,  MODEL_PATH)
        joblib.dump(scaler, SCALER_PATH)
    except Exception:
        pass

    return model, scaler, ML_FEATURES, round(acc * 100, 1), "trained"

MODEL, SCALER, FEATURES, MODEL_ACC, MODEL_SRC = load_or_train_model()

# ═══════════════════════════════════════════════════════════════════════════════
#  UTILS
# ═══════════════════════════════════════════════════════════════════════════════
def sf(x, d=0.0):
    try:
        if x is None or x == "" or (isinstance(x, float) and np.isnan(x)):
            return float(d)
        return float(x)
    except Exception:
        return float(d)

def col(df, name, fallback=""):
    """Case-insensitive column lookup."""
    for c in df.columns:
        if c.lower() == name.lower():
            return c
    return None

def patient_to_feature_vec(rec: dict) -> np.ndarray:
    """Map a patient record dict → ML feature vector."""
    mapping = {
        "Age":                     ["age","Age"],
        "Gender":                  ["gender","Gender"],
        "BMI":                     ["bmi","BMI"],
        "Smoking":                 ["smoking","Smoking"],
        "AlcoholConsumption":      ["alcohol_consumption","AlcoholConsumption","alcoholconsumption"],
        "PhysicalActivity":        ["physical_activity","PhysicalActivity"],
        "DietQuality":             ["diet_quality","DietQuality"],
        "SleepQuality":            ["sleep_quality","SleepQuality"],
        "FamilyHistoryDiabetes":   ["family_history_diabetes","FamilyHistoryDiabetes"],
        "GestationalDiabetes":     ["gestational_diabetes","GestationalDiabetes"],
        "PolycysticOvarySyndrome": ["polycystic_ovary_syndrome","PolycysticOvarySyndrome","pcos"],
        "PreviousPreDiabetes":     ["previous_pre_diabetes","PreviousPreDiabetes"],
        "Hypertension":            ["hypertension","Hypertension"],
        "SystolicBP":              ["systolic_bp","SystolicBP"],
        "DiastolicBP":             ["diastolic_bp","DiastolicBP"],
        "FastingBloodSugar":       ["fasting_blood_sugar","FastingBloodSugar","fastingbloodsugar"],
        "HbA1c":                   ["hba1c","HbA1c","hba1c"],
        "SerumCreatinine":         ["serum_creatinine","SerumCreatinine"],
        "BUNLevels":               ["bun_levels","BUNLevels"],
        "CholesterolTotal":        ["cholesterol_total","CholesterolTotal"],
        "CholesterolLDL":          ["cholesterol_ldl","CholesterolLDL"],
        "CholesterolHDL":          ["cholesterol_hdl","CholesterolHDL"],
        "CholesterolTriglycerides":["cholesterol_triglycerides","CholesterolTriglycerides"],
        "AntihypertensiveMedications":["antihypertensive_medications","AntihypertensiveMedications"],
        "Statins":                 ["statins","Statins"],
        "AntidiabeticMedications": ["antidiabetic_medications","AntidiabeticMedications"],
        "FrequentUrination":       ["frequent_urination","FrequentUrination"],
        "ExcessiveThirst":         ["excessive_thirst","ExcessiveThirst"],
        "UnexplainedWeightLoss":   ["unexplained_weight_loss","UnexplainedWeightLoss"],
        "FatigueLevels":           ["fatigue_levels","FatigueLevels"],
        "BlurredVision":           ["blurred_vision","BlurredVision"],
        "SlowHealingSores":        ["slow_healing_sores","SlowHealingSores"],
        "TinglingHandsFeet":       ["tingling_hands_feet","TinglingHandsFeet"],
        "QualityOfLifeScore":      ["quality_of_life_score","QualityOfLifeScore"],
        "MedicalCheckupsFrequency":["medical_checkups_frequency","MedicalCheckupsFrequency"],
        "MedicationAdherence":     ["medication_adherence","MedicationAdherence"],
        "HealthLiteracy":          ["health_literacy","HealthLiteracy"],
    }
    vec = []
    for feat in FEATURES:
        keys = mapping.get(feat, [feat.lower(), feat])
        val  = None
        for k in keys:
            if k in rec:
                val = rec[k]; break
        vec.append(sf(val))
    return np.array(vec, dtype=float)

def predict_prob(rec: dict):
    vec = patient_to_feature_vec(rec).reshape(1, -1)
    try:
        vec_s = SCALER.transform(vec)
        return float(MODEL.predict_proba(vec_s)[0][1]), "model"
    except Exception:
        hba = sf(rec.get("hba1c", rec.get("HbA1c", 5.5)))
        fbs = sf(rec.get("fasting_blood_sugar", rec.get("FastingBloodSugar", 100)))
        p   = min(1.0, max(0.0, (hba-4.5)/6.0*0.6 + (fbs-70)/200.0*0.4))
        return round(p, 4), "heuristic"

def risk_tier(p):
    if p >= 0.65: return "High",     "#f87171", "🔴"
    if p >= 0.35: return "Moderate", "#fbbf24", "🟡"
    return              "Low",       "#34d399", "🟢"

def simulate_projection(rec, years=5):
    pts = []
    for y in range(1, years+1):
        m = dict(rec)
        m["bmi"]               = str(sf(m.get("bmi",25))               + 0.25*y)
        m["fasting_blood_sugar"]= str(sf(m.get("fasting_blood_sugar",100)) + 2*y)
        p, _ = predict_prob(m)
        pts.append(round(p*100, 1))
    return pts

def ai_summary(rec, prob):
    hba = sf(rec.get("hba1c", rec.get("HbA1c",5.5)))
    bmi = sf(rec.get("bmi",   rec.get("BMI",25)))
    hs  = "normal" if hba<5.7 else ("pre-diabetes" if hba<6.5 else "diabetes range")
    bs  = "healthy" if bmi<25 else ("overweight" if bmi<30 else "obese")
    tier, _, icon = risk_tier(prob)
    if tier == "Low":
        t = f"HbA1c {hba:.1f}% ({hs}), BMI {bmi:.1f} ({bs}). Maintain current lifestyle; annual screening recommended."
    elif tier == "Moderate":
        t = f"HbA1c {hba:.1f}% ({hs}), BMI {bmi:.1f} ({bs}). Dietary adjustment and increased physical activity advised. Follow-up in 3–6 months."
    else:
        t = f"HbA1c {hba:.1f}% ({hs}), BMI {bmi:.1f} ({bs}). Urgent clinical review, medication assessment and lifestyle intervention recommended."
    proj = simulate_projection(rec)
    return f"{icon} **{tier} Risk ({prob*100:.1f}%)** — {t} Projected 5-yr: {' → '.join(str(x)+'%' for x in proj)}"

def generate_qr(pid):
    path = os.path.join(QR_FOLDER, f"{pid}.png")
    if not os.path.exists(path):
        qr = qrcode.QRCode(version=1, box_size=6, border=2)
        qr.add_data(pid); qr.make(fit=True)
        qr.make_image(fill_color="#6366f1", back_color="white").save(path)
    return path

def fmt(v, decimals=2):
    if v is None or v == "": return "—"
    try: return f"{float(v):.{decimals}f}"
    except: return str(v)

def gauge_chart(prob):
    tier, color, _ = risk_tier(prob)
    fig = go.Figure(go.Indicator(
        mode   = "gauge+number+delta",
        value  = round(prob*100, 1),
        number = {"suffix":"%","font":{"color":"white","family":"Outfit","size":36}},
        delta  = {"reference":50,"increasing":{"color":"#f87171"},"decreasing":{"color":"#34d399"}},
        gauge  = {
            "axis":{"range":[0,100],"tickcolor":"#475569","tickfont":{"color":"#475569"}},
            "bar" :{"color":color,"thickness":0.25},
            "bgcolor":"rgba(0,0,0,0)",
            "bordercolor":"rgba(0,0,0,0)",
            "steps":[
                {"range":[0, 35], "color":"rgba(52,211,153,0.08)"},
                {"range":[35,65], "color":"rgba(251,191,36,0.08)"},
                {"range":[65,100],"color":"rgba(248,113,113,0.08)"},
            ],
            "threshold":{"line":{"color":color,"width":3},"value":prob*100}
        },
        title={"text":f"<b>{tier} Risk</b>","font":{"color":color,"family":"Outfit","size":16}}
    ))
    fig.update_layout(height=260, margin=dict(t=40,b=10,l=20,r=20),
                      paper_bgcolor="rgba(0,0,0,0)", font=dict(color="white"))
    return fig

def proj_chart(pts):
    colors = ["#34d399" if p<35 else ("#fbbf24" if p<65 else "#f87171") for p in pts]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(1,6)), y=pts, mode="lines+markers",
        line=dict(color="#6366f1",width=3,shape="spline"),
        marker=dict(size=10, color=colors,
                    line=dict(color="rgba(255,255,255,0.2)",width=2)),
        fill="tozeroy", fillcolor="rgba(99,102,241,0.06)",
        name="Risk %"
    ))
    fig.add_hline(y=65, line=dict(color="#f87171",dash="dot",width=1),
                  annotation_text="High", annotation_font_color="#f87171")
    fig.add_hline(y=35, line=dict(color="#fbbf24",dash="dot",width=1),
                  annotation_text="Moderate", annotation_font_color="#fbbf24")
    fig.update_layout(
        title=dict(text="5-Year Risk Projection", font=dict(color="#94a3b8",size=14)),
        xaxis=dict(title="Year", tickvals=list(range(1,6)),
                   gridcolor="rgba(255,255,255,0.04)", color="#64748b"),
        yaxis=dict(title="Risk %", range=[0,100],
                   gridcolor="rgba(255,255,255,0.04)", color="#64748b"),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"), height=260, margin=dict(t=40,b=20,l=20,r=20),
        showlegend=False
    )
    return fig

def vitals_radar(rec):
    cats = ["HbA1c","BMI","BP","Cholesterol","Glucose","Age"]
    hba  = min(sf(rec.get("hba1c",5.5))/14*100, 100)
    bmi  = min(sf(rec.get("bmi",25))/50*100, 100)
    sbp  = min(sf(rec.get("systolic_bp",120))/200*100, 100)
    chol = min(sf(rec.get("cholesterol_total",180))/400*100, 100)
    fbs  = min(sf(rec.get("fasting_blood_sugar",100))/300*100, 100)
    age  = min(sf(rec.get("age",40))/100*100, 100)
    vals = [hba, bmi, sbp, chol, fbs, age]
    fig  = go.Figure(go.Scatterpolar(
        r=vals+[vals[0]], theta=cats+[cats[0]],
        fill="toself",
        fillcolor="rgba(99,102,241,0.12)",
        line=dict(color="#6366f1", width=2),
        marker=dict(color="#a5b4fc", size=6)
    ))
    fig.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(visible=True, range=[0,100],
                            gridcolor="rgba(255,255,255,0.06)",
                            tickfont=dict(color="#475569",size=9)),
            angularaxis=dict(gridcolor="rgba(255,255,255,0.06)",
                             tickfont=dict(color="#94a3b8",size=11))
        ),
        paper_bgcolor="rgba(0,0,0,0)", font=dict(color="white"),
        height=280, margin=dict(t=20,b=20,l=30,r=30),
        showlegend=False,
        title=dict(text="Health Profile Radar", font=dict(color="#94a3b8",size=13))
    )
    return fig

# ═══════════════════════════════════════════════════════════════════════════════
#  TOP HEADER
# ═══════════════════════════════════════════════════════════════════════════════
def render_header(subtitle="Smart Diabetes EMR"):
    sb_status = "🟢 Connected" if get_supabase() else "🟡 Demo Mode"
    ml_status = f"🧠 Model {'Loaded' if MODEL_SRC=='loaded' else 'Auto-trained'}" + \
                (f" · {MODEL_ACC}% acc" if MODEL_ACC else "")
    st.markdown(f"""
    <div class="topbar">
        <div style="font-size:32px">🩺</div>
        <div style="flex:1">
            <div class="topbar-title">GLUCO-LENS</div>
            <div class="topbar-sub">{subtitle}</div>
        </div>
        <div style="text-align:right; font-size:12px; color:#475569; line-height:2">
            {sb_status}<br>{ml_status}
        </div>
    </div>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
def render_sidebar():
    with st.sidebar:
        st.markdown(f"""
        <div class="sb-role">
            <div class="role-name">{'👑' if st.session_state.role=='Admin' else '👩‍⚕️' if st.session_state.role=='Doctor' else '🫀'} {st.session_state.role}</div>
            <div class="role-user">@{st.session_state.username}</div>
        </div>
        """, unsafe_allow_html=True)

        sb = get_supabase()
        st.markdown(f"""
        <div style="font-size:12px;color:#475569;margin-bottom:12px;padding:8px 12px;
                    background:rgba(255,255,255,0.02);border-radius:8px;">
            {'🟢 Supabase connected' if sb else '🟡 Demo data (add Supabase secrets to persist)'}
        </div>
        """, unsafe_allow_html=True)

        if MODEL_SRC == "trained":
            st.markdown(f"""
            <div style="font-size:12px;color:#475569;margin-bottom:16px;padding:8px 12px;
                        background:rgba(99,102,241,0.08);border-radius:8px;border:1px solid rgba(99,102,241,0.15)">
                🧠 RF auto-trained · {MODEL_ACC}% accuracy<br>
                <span style="color:#6366f1">Place diabetes_rf.pkl + scaler.pkl to use your own model</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        if st.button("🚪 Logout", use_container_width=True):
            for k in _defaults:
                st.session_state[k] = _defaults[k]
            st.rerun()

# ═══════════════════════════════════════════════════════════════════════════════
#  LOGIN
# ═══════════════════════════════════════════════════════════════════════════════
def show_login():
    render_header("Please sign in to continue")
    _, mid, _ = st.columns([1, 1.4, 1])
    with mid:
        st.markdown('<div class="login-card">', unsafe_allow_html=True)
        st.markdown('<div class="login-logo">GLUCO-LENS</div>', unsafe_allow_html=True)
        st.markdown('<div class="login-sub">Smart Diabetes EMR</div>', unsafe_allow_html=True)
        username = st.text_input("Username", placeholder="Enter username")
        password = st.text_input("Password", type="password", placeholder="Enter password")
        if st.button("Sign In →", use_container_width=True, type="primary"):
            u = username.lower().strip()
            if u in CREDENTIALS and CREDENTIALS[u]["password"] == password:
                st.session_state.logged_in = True
                st.session_state.role      = CREDENTIALS[u]["role"]
                st.session_state.username  = u
                st.rerun()
            else:
                st.error("Invalid credentials. Please try again.")
        st.markdown("""
        <div class="cred-badge">
            admin / admin123<br>doctor / doc123<br>patient / pat123
        </div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
#  ADMIN DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════
def admin_dashboard():
    render_header("Admin Dashboard")
    render_sidebar()
    df = db_fetch_all()

    # ── KPI row ───────────────────────────────────────────────────────────────
    total = len(df)
    hba_col = next((c for c in df.columns if c.lower()=="hba1c"), None)
    age_col = next((c for c in df.columns if c.lower()=="age"), None)
    avg_hba = pd.to_numeric(df[hba_col], errors="coerce").mean() if hba_col else float("nan")
    avg_age = pd.to_numeric(df[age_col], errors="coerce").mean() if age_col else float("nan")

    # compute risk for all patients
    all_probs = []
    for _, row in df.iterrows():
        p, _ = predict_prob(row.to_dict())
        all_probs.append(p)
    df["_risk"] = all_probs
    high_risk = int((np.array(all_probs) >= 0.65).sum())

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("👥 Patients",    total)
    c2.metric("⚠️ High Risk",   high_risk)
    c3.metric("📊 Avg HbA1c",  f"{avg_hba:.2f}" if not np.isnan(avg_hba) else "—")
    c4.metric("🎂 Avg Age",     f"{avg_age:.0f}" if not np.isnan(avg_age) else "—")

    st.markdown("<br>", unsafe_allow_html=True)
    tab1, tab2, tab3 = st.tabs(["📋 Patient Records", "➕ Add Patient", "📊 Analytics"])

    # ── Tab 1: records ────────────────────────────────────────────────────────
    with tab1:
        disp = df.drop(columns=["_risk"], errors="ignore")
        st.dataframe(disp, use_container_width=True, height=400)
        csv = disp.to_csv(index=False).encode()
        st.download_button("⬇️ Export CSV", data=csv,
                           file_name="gluco_lens_patients.csv", mime="text/csv")

    # ── Tab 2: add patient ────────────────────────────────────────────────────
    with tab2:
        st.markdown('<p class="sec-head">New Patient Record</p>', unsafe_allow_html=True)
        with st.form("add_pt"):
            c1,c2,c3 = st.columns(3)
            pid   = c1.text_input("Patient ID*", value=f"P{total+1:03d}")
            name  = c2.text_input("Full Name*")
            diag  = c3.text_input("Diagnosis")

            c1,c2,c3,c4 = st.columns(4)
            age_v = c1.number_input("Age",  18, 100, 40)
            gen_v = c2.selectbox("Gender", ["Male","Female"])
            bmi_v = c3.number_input("BMI",  10.0, 60.0, 25.0)
            hba_v = c4.number_input("HbA1c (%)", 3.0, 15.0, 5.5)

            c1,c2,c3,c4 = st.columns(4)
            fbs_v = c1.number_input("Fasting Blood Sugar", 50, 400, 100)
            chol_v= c2.number_input("Total Cholesterol",   50, 500, 180)
            sbp_v = c3.number_input("Systolic BP",         80, 220, 120)
            dbp_v = c4.number_input("Diastolic BP",        40, 130, 80)

            c1,c2 = st.columns(2)
            hyp_v = c1.selectbox("Hypertension",          ["No","Yes"])
            fam_v = c2.selectbox("Family History Diabetes",["No","Yes"])

            remarks_v = st.text_area("Doctor Remarks", height=80)

            if st.form_submit_button("✅ Save Patient", use_container_width=True):
                if not name.strip():
                    st.error("Name is required.")
                else:
                    row = {
                        "patient_id": pid.strip(),
                        "name": name.strip(),
                        "age": age_v, "gender": 1 if gen_v=="Male" else 0,
                        "bmi": bmi_v, "hba1c": hba_v,
                        "fasting_blood_sugar": fbs_v, "cholesterol_total": chol_v,
                        "systolic_bp": sbp_v, "diastolic_bp": dbp_v,
                        "hypertension": 1 if hyp_v=="Yes" else 0,
                        "family_history_diabetes": 1 if fam_v=="Yes" else 0,
                        "diagnosis": diag, "doctor_remarks": remarks_v,
                        "created_at": datetime.utcnow().isoformat()
                    }
                    ok = db_upsert(row)
                    if ok:
                        st.success(f"✅ Patient {pid} saved to database!")
                        try:
                            qp = generate_qr(pid.strip())
                            st.image(qp, width=140, caption=f"QR · {pid}")
                        except Exception:
                            pass
                        st.cache_data.clear()
                    else:
                        st.warning("⚠️ Supabase not connected — data not persisted. Add SUPABASE_URL & SUPABASE_KEY to Streamlit secrets.")

    # ── Tab 3: analytics ──────────────────────────────────────────────────────
    with tab3:
        if hba_col:
            hba_vals = pd.to_numeric(df[hba_col], errors="coerce").dropna()
            col1, col2 = st.columns(2)
            with col1:
                fig = px.histogram(hba_vals, nbins=20, title="HbA1c Distribution",
                                   color_discrete_sequence=["#6366f1"])
                fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                                  plot_bgcolor="rgba(0,0,0,0)",
                                  font=dict(color="#94a3b8"), height=280,
                                  margin=dict(t=40,b=10))
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                risk_counts = pd.Series(
                    ["High" if p>=0.65 else "Moderate" if p>=0.35 else "Low"
                     for p in all_probs]).value_counts()
                fig2 = px.pie(values=risk_counts.values, names=risk_counts.index,
                              title="Risk Distribution",
                              color_discrete_map={"High":"#f87171","Moderate":"#fbbf24","Low":"#34d399"},
                              hole=0.55)
                fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                                   font=dict(color="#94a3b8"), height=280,
                                   margin=dict(t=40,b=10))
                st.plotly_chart(fig2, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
#  DOCTOR DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════
def doctor_dashboard():
    render_header("Doctor Dashboard")
    render_sidebar()
    df = db_fetch_all()

    st.markdown('<p class="sec-head">Patient Lookup</p>', unsafe_allow_html=True)
    col1, col2 = st.columns([3,1])
    pid_in = col1.text_input("Patient ID", placeholder="e.g. P001", label_visibility="collapsed")
    fetch  = col2.button("🔍 Fetch", use_container_width=True)

    if fetch and pid_in.strip():
        pid_col = next((c for c in df.columns if c.lower()=="patient_id"), None)
        if pid_col:
            rows = df[df[pid_col].astype(str).str.strip() == pid_in.strip()]
            if not rows.empty:
                st.session_state.doc_record = rows.iloc[0].to_dict()
                st.session_state.doc_pid    = pid_in.strip()
            else:
                st.error(f"No patient found with ID `{pid_in.strip()}`")
        else:
            st.error("patient_id column not found in database.")

    rec = st.session_state.get("doc_record")
    pid = st.session_state.get("doc_pid")

    if rec:
        prob, src = predict_prob(rec)
        tier, color, icon = risk_tier(prob)

        name = rec.get("name", rec.get("Name", pid))
        st.markdown(f"""
        <div class="glass" style="border-left:3px solid {color};">
            <div style="display:flex;justify-content:space-between;align-items:center;">
                <div>
                    <div style="font-size:22px;font-weight:700;color:#e2e8f5;">{name}</div>
                    <div style="font-size:13px;color:#64748b;margin-top:4px;">
                        ID: {pid} &nbsp;·&nbsp;
                        Age: {fmt(rec.get('age',rec.get('Age','')),0)} &nbsp;·&nbsp;
                        {rec.get('diagnosis',rec.get('Diagnosis','—'))}
                    </div>
                </div>
                <div style="text-align:right;">
                    <div style="font-size:36px;font-weight:800;color:{color};">{prob*100:.1f}%</div>
                    <div style="font-size:13px;color:{color};">{icon} {tier} Risk</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # vitals strip
        v_keys = [("fasting_blood_sugar","FastingBloodSugar","Glucose"),
                  ("hba1c","HbA1c","HbA1c %"),
                  ("bmi","BMI","BMI"),
                  ("cholesterol_total","CholesterolTotal","Cholesterol"),
                  ("systolic_bp","SystolicBP","Systolic BP"),
                  ("diastolic_bp","DiastolicBP","Diastolic BP")]
        vc = st.columns(len(v_keys))
        for i,(k1,k2,lbl) in enumerate(v_keys):
            v = rec.get(k1, rec.get(k2,""))
            vc[i].metric(lbl, fmt(v))

        st.markdown("<br>", unsafe_allow_html=True)
        col_l, col_r = st.columns([3,2])

        with col_l:
            st.plotly_chart(proj_chart(simulate_projection(rec)), use_container_width=True)
            st.markdown('<p class="sec-head">Full Record</p>', unsafe_allow_html=True)
            clean = {k:fmt(v) for k,v in rec.items() if not k.startswith("_")}
            st.dataframe(pd.DataFrame(clean.items(), columns=["Field","Value"]),
                         use_container_width=True, hide_index=True, height=320)

        with col_r:
            st.plotly_chart(gauge_chart(prob),     use_container_width=True)
            st.plotly_chart(vitals_radar(rec),     use_container_width=True)
            try:
                st.image(generate_qr(pid), width=140, caption="Patient QR")
            except Exception:
                pass

        # AI summary
        st.markdown(f'<div class="ai-box">🤖 <b>AI Assessment</b><br><br>{ai_summary(rec, prob)}</div>',
                    unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        # doctor remarks
        st.markdown('<p class="sec-head">Doctor Remarks</p>', unsafe_allow_html=True)
        existing = rec.get("doctor_remarks", rec.get("DoctorRemarks",""))
        note = st.text_area("Add / edit remarks", value=existing, height=100,
                            label_visibility="collapsed")
        if st.button("💾 Save Remarks to Database"):
            updated = dict(rec)
            updated["doctor_remarks"] = note
            ok = db_upsert({k: updated[k] for k in updated if not k.startswith("_")})
            if ok:
                st.success("✅ Remarks saved to Supabase!")
                st.session_state.doc_record["doctor_remarks"] = note
                st.cache_data.clear()
            else:
                st.warning("⚠️ Supabase not configured. Remarks saved in session only.")
                st.session_state.doc_record["doctor_remarks"] = note
    else:
        st.markdown("""
        <div class="glass" style="text-align:center;padding:48px;color:#475569;">
            <div style="font-size:48px;margin-bottom:16px;">🔍</div>
            <div style="font-size:16px;">Enter a Patient ID above to view their record</div>
        </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
#  PATIENT PORTAL
# ═══════════════════════════════════════════════════════════════════════════════
def patient_portal():
    render_header("Patient Portal")
    render_sidebar()
    df = db_fetch_all()

    st.markdown('<p class="sec-head">View Your Health Record</p>', unsafe_allow_html=True)
    col1, col2 = st.columns([3,1])
    pid_in = col1.text_input("Your Patient ID", placeholder="e.g. P001", label_visibility="collapsed")
    fetch  = col2.button("View Record", use_container_width=True)

    if fetch and pid_in.strip():
        pid_col = next((c for c in df.columns if c.lower()=="patient_id"), None)
        if pid_col:
            rows = df[df[pid_col].astype(str).str.strip() == pid_in.strip()]
            if not rows.empty:
                st.session_state.pat_record = rows.iloc[0].to_dict()
                st.session_state.pat_pid    = pid_in.strip()
            else:
                st.error(f"No record found for `{pid_in.strip()}`")
        else:
            st.error("Database structure issue. Contact admin.")

    rec = st.session_state.get("pat_record")
    pid = st.session_state.get("pat_pid")

    if rec:
        prob, _ = predict_prob(rec)
        tier, color, icon = risk_tier(prob)
        name = rec.get("name", rec.get("Name", pid))

        st.markdown(f"""
        <div class="glass" style="border-left:3px solid {color};">
            <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:16px;">
                <div>
                    <div style="font-size:24px;font-weight:700;color:#e2e8f5;">Welcome, {name} 👋</div>
                    <div style="font-size:13px;color:#64748b;margin-top:4px;">Patient ID: {pid}</div>
                </div>
                <div style="text-align:center;
                            background:rgba(255,255,255,0.04);border-radius:16px;
                            padding:16px 28px;border:1px solid rgba(255,255,255,0.07);">
                    <div style="font-size:40px;font-weight:800;color:{color};">{prob*100:.1f}%</div>
                    <div style="font-size:12px;color:{color};letter-spacing:1px;margin-top:4px;">
                        {icon} {tier.upper()} RISK
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # vitals
        v_keys = [("fasting_blood_sugar","FastingBloodSugar","Glucose (mg/dL)"),
                  ("hba1c","HbA1c","HbA1c (%)"),
                  ("bmi","BMI","BMI"),
                  ("cholesterol_total","CholesterolTotal","Cholesterol"),
                  ("systolic_bp","SystolicBP","Systolic BP"),
                  ("diastolic_bp","DiastolicBP","Diastolic BP")]
        vc = st.columns(len(v_keys))
        for i,(k1,k2,lbl) in enumerate(v_keys):
            vc[i].metric(lbl, fmt(rec.get(k1, rec.get(k2,""))))

        st.markdown("<br>", unsafe_allow_html=True)
        col_l, col_r = st.columns([3,2])
        with col_l:
            st.plotly_chart(proj_chart(simulate_projection(rec)), use_container_width=True)
        with col_r:
            st.plotly_chart(gauge_chart(prob),  use_container_width=True)
            st.plotly_chart(vitals_radar(rec),  use_container_width=True)

        # AI summary
        st.markdown(f'<div class="ai-box">🤖 <b>Your Health Summary</b><br><br>{ai_summary(rec, prob)}</div>',
                    unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        # doctor remarks
        remarks = rec.get("doctor_remarks", rec.get("DoctorRemarks",""))
        if remarks:
            st.markdown(f"""
            <div class="glass">
                <p class="sec-head">📝 Doctor's Notes</p>
                <div style="color:#94a3b8;font-size:14px;line-height:1.7;">{remarks}</div>
            </div>""", unsafe_allow_html=True)

        # download
        st.download_button("⬇️ Download My Record",
                           data=pd.DataFrame([{k:v for k,v in rec.items() if not k.startswith("_")}]).to_csv(index=False),
                           file_name=f"{pid}_health_record.csv", mime="text/csv")

        # Lifestyle estimator
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<p class="sec-head">🌿 Lifestyle Risk Estimator</p>', unsafe_allow_html=True)
        st.caption("See how your daily habits affect your diabetes risk.")

        with st.form("lifestyle"):
            c1,c2 = st.columns(2)
            q1 = c1.select_slider("Exercise (mins/day)",    ["0","10-30","30-60","60+"],         value="30-60")
            q2 = c2.select_slider("Added sugar (tsp/day)",  ["0-5","6-10","11-20","20+"],         value="6-10")
            q3 = c1.select_slider("Veg/fruit servings/day", ["0-1","2-3","4-5","5+"],             value="2-3")
            q4 = c2.select_slider("Sleep (hrs/night)",      ["<5","5-6","6-7","7+"],              value="6-7")
            q5 = c1.select_slider("Fast food frequency",    ["Daily","Few/week","Weekly","Rarely"],value="Few/week")
            q6 = c2.select_slider("Stress level (0-10)",    ["0-2","3-5","6-8","9-10"],           value="3-5")
            sub = st.form_submit_button("Calculate →", use_container_width=True)

        if sub:
            sm = {"0":0,"10-30":1,"30-60":2,"60+":3,
                  "0-5":3,"6-10":2,"11-20":1,"20+":0,
                  "0-1":0,"2-3":1,"4-5":2,"5+":3,
                  "<5":0,"5-6":1,"6-7":2,"7+":3,
                  "Daily":0,"Few/week":1,"Weekly":2,"Rarely":3,
                  "0-2":3,"3-5":2,"6-8":1,"9-10":0}
            total   = sum(sm.get(q,0) for q in [q1,q2,q3,q4,q5,q6])
            mod     = (total - 9) / 18.0
            new_p   = float(np.clip(prob - mod*0.25, 0, 1))
            diff    = new_p - prob
            hba_b   = sf(rec.get("hba1c", rec.get("HbA1c",5.5)))
            hba_n   = hba_b + diff*2

            ca,cb,cc = st.columns(3)
            ca.metric("Current Risk",           f"{prob*100:.1f}%")
            cb.metric("Lifestyle-Adjusted Risk",f"{new_p*100:.1f}%", delta=f"{diff*100:+.1f}%")
            cc.metric("Est. HbA1c",             f"{hba_n:.2f}%",     delta=f"{(hba_n-hba_b):+.2f}")

            lm = {"Exercise":sm[q1],"Sugar":sm[q2],"Veg/Fruit":sm[q3],
                  "Sleep":sm[q4],"Fast Food":sm[q5],"Stress":sm[q6]}
            fig_ls = px.bar(
                pd.DataFrame({"Factor":list(lm.keys()),"Score":list(lm.values())}),
                x="Factor", y="Score", color="Score",
                color_continuous_scale=[[0,"#f87171"],[0.5,"#fbbf24"],[1,"#34d399"]],
                title="Lifestyle Score Breakdown (0=poor, 3=excellent)"
            )
            fig_ls.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                  font=dict(color="#94a3b8"), height=260, margin=dict(t=40,b=10))
            st.plotly_chart(fig_ls, use_container_width=True)
    else:
        st.markdown("""
        <div class="glass" style="text-align:center;padding:48px;color:#475569;">
            <div style="font-size:48px;margin-bottom:16px;">🫀</div>
            <div style="font-size:16px;">Enter your Patient ID above to view your health record</div>
        </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════
if not st.session_state.logged_in:
    show_login()
else:
    role = st.session_state.role
    if role == "Admin":
        admin_dashboard()
    elif role == "Doctor":
        doctor_dashboard()
    elif role == "Patient":
        patient_portal()
    else:
        st.error("Unknown role. Please log out.")
