"""
╔══════════════════════════════════════════════════════════════════════════════╗
║           GLUCO-LENS v5 — Next-Gen Secure Healthcare EMR                   ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  ARCHITECTURE HIGHLIGHTS                                                    ║
║                                                                             ║
║  1. JWT-BASED SECURE QR (Passwordless Healthcare Authentication)            ║
║     • Each patient gets a signed JWT embedded in their QR code             ║
║     • Token includes patient_id, role, and expiry (HMAC-SHA256)            ║
║     • Even if someone photographs the QR, expired tokens are rejected      ║
║                                                                             ║
║  2. SEVERITY SCORING ENGINE                                                 ║
║     • 0–100 composite score: HbA1c, FBS, BMI, BP, lifestyle                ║
║     • Weighted formula with clinical-grade calibration                     ║
║     • Low / Medium / High category bands                                   ║
║                                                                             ║
║  3. FUTURISTIC NEXT-GEN UI                                                  ║
║     • Glassmorphism + soft gradient system                                  ║
║     • Card-based layout, animated panels                                   ║
║     • Dark mode by default, Apple-Health-inspired                          ║
║                                                                             ║
║  4. ADVANCED ADMIN ANALYTICS                                                ║
║     • Pie, Bar, Line, Histogram, Heatmap, Radar charts                     ║
║     • Smart patient table with color-coded severity                        ║
║     • Filter system: age, gender, diagnosis, risk                          ║
║     • AI insights per patient                                               ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

# ─────────────────────────────────────────────────────────────────────────────
#  IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import joblib
import qrcode
import os
import io
import base64
import time
import hashlib
import hmac
import json
from datetime import datetime, timezone, timedelta

# ── Optional dependencies ─────────────────────────────────────────────────────
try:
    from supabase import create_client
    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False

try:
    import jwt as pyjwt
    JWT_AVAILABLE = True
except ImportError:
    JWT_AVAILABLE = False

try:
    from pyzbar import pyzbar as pyzbar_lib
    PYZBAR_AVAILABLE = True
except Exception:
    PYZBAR_AVAILABLE = False

from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ─────────────────────────────────────────────────────────────────────────────
#  CONSTANTS & CONFIG
# ─────────────────────────────────────────────────────────────────────────────
MODEL_PATH  = "diabetes_rf.pkl"
SCALER_PATH = "scaler.pkl"
QR_FOLDER   = "/tmp/qrcodes"
TABLE_NAME  = "patients"
os.makedirs(QR_FOLDER, exist_ok=True)

JWT_SECRET      = "gluco-lens-jwt-secret-2024-change-in-production"
JWT_EXPIRY_DAYS = 365

SUPABASE_URL = ""
SUPABASE_KEY = ""
try:
    SUPABASE_URL = st.secrets["SUPABASE_URL"]
    SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
except Exception:
    pass

try:
    JWT_SECRET = st.secrets.get("JWT_SECRET", JWT_SECRET)
except Exception:
    pass

CREDENTIALS = {
    "admin":  {"password": "admin123", "role": "Admin"},
    "doctor": {"password": "doc123",   "role": "Doctor"},
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

# ─────────────────────────────────────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GLUCO-LENS v5",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────────────────────
#  GLOBAL CSS  — Next-Gen Futuristic Dark UI
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Syne:wght@700;800&family=DM+Mono:wght@400;500&display=swap');

/* ── Reset & Base ── */
*, *::before, *::after { box-sizing: border-box; margin: 0; }

html, body, [data-testid="stAppViewContainer"] {
    background: #050914 !important;
    color: #dde4f0 !important;
    font-family: 'Space Grotesk', sans-serif !important;
}

/* ── Background mesh ── */
[data-testid="stAppViewContainer"]::before {
    content: '';
    position: fixed;
    inset: 0;
    background:
        radial-gradient(ellipse 90% 60% at 10% -5%,  rgba(79,70,229,0.22) 0%, transparent 55%),
        radial-gradient(ellipse 70% 50% at 90% 105%, rgba(6,182,212,0.16) 0%, transparent 55%),
        radial-gradient(ellipse 50% 40% at 50% 50%,  rgba(139,92,246,0.06) 0%, transparent 70%);
    pointer-events: none;
    z-index: 0;
}

[data-testid="stAppViewContainer"] > * { position: relative; z-index: 1; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: rgba(8,10,26,0.97) !important;
    border-right: 1px solid rgba(79,70,229,0.18) !important;
    backdrop-filter: blur(24px) !important;
}
[data-testid="stSidebar"] * {
    color: #b4bcd8 !important;
    font-family: 'Space Grotesk', sans-serif !important;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stDecoration"] { display: none; }

/* ── Glass card ── */
.card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 20px;
    padding: 24px;
    backdrop-filter: blur(16px);
    margin-bottom: 20px;
    box-shadow:
        0 8px 40px rgba(0,0,0,0.45),
        inset 0 1px 0 rgba(255,255,255,0.05);
    transition: border-color 0.3s;
}
.card:hover { border-color: rgba(79,70,229,0.25); }

.card-accent-indigo { border-left: 3px solid #6366f1; }
.card-accent-cyan   { border-left: 3px solid #06b6d4; }
.card-accent-violet { border-left: 3px solid #8b5cf6; }
.card-accent-rose   { border-left: 3px solid #f43f5e; }
.card-accent-amber  { border-left: 3px solid #f59e0b; }
.card-accent-green  { border-left: 3px solid #10b981; }

/* ── Top bar ── */
.topbar {
    background: linear-gradient(135deg, rgba(79,70,229,0.2), rgba(6,182,212,0.15));
    border: 1px solid rgba(99,102,241,0.25);
    border-radius: 22px;
    padding: 22px 36px;
    margin-bottom: 32px;
    display: flex;
    align-items: center;
    gap: 20px;
    box-shadow:
        0 0 80px rgba(79,70,229,0.12),
        inset 0 1px 0 rgba(255,255,255,0.08);
}
.topbar-logo {
    font-family: 'Syne', sans-serif;
    font-size: 30px;
    font-weight: 800;
    letter-spacing: 4px;
    background: linear-gradient(120deg, #818cf8, #38bdf8, #c084fc, #818cf8);
    background-size: 250% auto;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: logo-shimmer 4s linear infinite;
}
@keyframes logo-shimmer {
    0%   { background-position: 0% center; }
    100% { background-position: 250% center; }
}
.topbar-sub {
    font-size: 12px;
    color: #64748b;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-top: 3px;
}
.badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 5px 12px;
    border-radius: 100px;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.5px;
}
.badge-green  { background: rgba(16,185,129,0.12); border: 1px solid rgba(16,185,129,0.3);  color: #34d399; }
.badge-yellow { background: rgba(245,158,11,0.12);  border: 1px solid rgba(245,158,11,0.3);  color: #fbbf24; }
.badge-red    { background: rgba(244,63,94,0.12);   border: 1px solid rgba(244,63,94,0.3);   color: #fb7185; }
.badge-indigo { background: rgba(99,102,241,0.12);  border: 1px solid rgba(99,102,241,0.3);  color: #a5b4fc; }
.badge-cyan   { background: rgba(6,182,212,0.12);   border: 1px solid rgba(6,182,212,0.3);   color: #22d3ee; }

/* ── Section heading ── */
.sec-head {
    font-family: 'Syne', sans-serif;
    font-size: 16px;
    font-weight: 700;
    color: #c7d2fe;
    letter-spacing: 0.5px;
    margin-bottom: 18px;
    padding-bottom: 12px;
    border-bottom: 1px solid rgba(255,255,255,0.06);
}

/* ── KPI metric card ── */
.kpi {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 18px;
    padding: 20px 22px;
    position: relative;
    overflow: hidden;
    transition: transform 0.25s, border-color 0.25s;
}
.kpi:hover { transform: translateY(-2px); border-color: rgba(99,102,241,0.3); }
.kpi::after {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: var(--kpi-color, #6366f1);
    opacity: 0.8;
}
.kpi-label  { font-size: 11px; color: #475569; text-transform: uppercase; letter-spacing: 1.5px; margin-bottom: 8px; }
.kpi-value  { font-family: 'Syne', sans-serif; font-size: 34px; font-weight: 800; color: #e2e8f5; line-height: 1; }
.kpi-delta  { font-size: 12px; margin-top: 6px; }
.kpi-icon   { position: absolute; top: 18px; right: 18px; font-size: 24px; opacity: 0.35; }

/* ── Risk row color coding ── */
.risk-high   { color: #fb7185 !important; font-weight: 700; }
.risk-medium { color: #fbbf24 !important; font-weight: 600; }
.risk-low    { color: #34d399 !important; font-weight: 600; }

/* ── AI insight box ── */
.ai-box {
    background: linear-gradient(135deg, rgba(79,70,229,0.08), rgba(6,182,212,0.05));
    border: 1px solid rgba(99,102,241,0.18);
    border-left: 3px solid #6366f1;
    border-radius: 16px;
    padding: 20px 22px;
    font-size: 14px;
    line-height: 1.75;
    color: #b4bcd8;
    margin-top: 8px;
}
.ai-insight-row {
    background: rgba(99,102,241,0.05);
    border: 1px solid rgba(99,102,241,0.1);
    border-radius: 12px;
    padding: 14px 16px;
    margin-bottom: 8px;
}
.ai-insight-row .label {
    font-size: 11px;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 4px;
}
.ai-insight-row .value { font-size: 14px; color: #cbd5e1; }

/* ── Severity bar ── */
.sev-bar-bg {
    background: rgba(255,255,255,0.06);
    border-radius: 999px;
    height: 6px;
    margin-top: 6px;
    overflow: hidden;
}
.sev-bar-fill {
    height: 100%;
    border-radius: 999px;
    background: linear-gradient(90deg, #34d399, #fbbf24, #f43f5e);
    transition: width 0.6s ease;
}

/* ── QR / JWT ── */
.jwt-pill {
    background: rgba(99,102,241,0.08);
    border: 1px solid rgba(99,102,241,0.18);
    border-radius: 10px;
    padding: 10px 14px;
    font-family: 'DM Mono', monospace;
    font-size: 11px;
    color: #64748b;
    margin-top: 8px;
    line-height: 1.9;
}

/* ── Login card ── */
.login-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(99,102,241,0.22);
    border-radius: 28px;
    padding: 52px 48px;
    box-shadow: 0 32px 100px rgba(0,0,0,0.6), 0 0 100px rgba(79,70,229,0.08);
    width: 100%;
    max-width: 460px;
}
.login-logo {
    font-family: 'Syne', sans-serif;
    font-size: 44px;
    font-weight: 800;
    letter-spacing: 4px;
    background: linear-gradient(120deg, #818cf8, #38bdf8, #c084fc, #818cf8);
    background-size: 250% auto;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: logo-shimmer 4s linear infinite;
    text-align: center;
    margin-bottom: 4px;
}
.login-tagline {
    text-align: center;
    color: #334155;
    font-size: 12px;
    letter-spacing: 3px;
    text-transform: uppercase;
    margin-bottom: 36px;
}
.cred-hint {
    background: rgba(99,102,241,0.08);
    border: 1px solid rgba(99,102,241,0.18);
    border-radius: 12px;
    padding: 14px 18px;
    font-family: 'DM Mono', monospace;
    font-size: 12px;
    color: #64748b;
    line-height: 2;
    margin-top: 16px;
}

/* ── Scan zone ── */
.scan-zone {
    background: rgba(79,70,229,0.05);
    border: 2px dashed rgba(99,102,241,0.35);
    border-radius: 20px;
    padding: 32px 20px;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.scan-corner { position: absolute; width: 26px; height: 26px; border-color: #6366f1; border-style: solid; border-width: 0; }
.scan-corner.tl { top:10px; left:10px;   border-top-width:3px;    border-left-width:3px;   border-radius: 4px 0 0 0; }
.scan-corner.tr { top:10px; right:10px;  border-top-width:3px;    border-right-width:3px;  border-radius: 0 4px 0 0; }
.scan-corner.bl { bottom:10px; left:10px;  border-bottom-width:3px; border-left-width:3px;   border-radius: 0 0 0 4px; }
.scan-corner.br { bottom:10px; right:10px; border-bottom-width:3px; border-right-width:3px;  border-radius: 0 0 4px 0; }
.scan-line {
    position: absolute; left: 10px; right: 10px; height: 2px;
    background: linear-gradient(90deg, transparent, #6366f1, #38bdf8, #6366f1, transparent);
    animation: scanline 2.4s ease-in-out infinite;
    border-radius: 99px;
}
@keyframes scanline {
    0%   { top: 10px;  opacity: 1; }
    50%  { top: calc(100% - 14px); opacity: 0.7; }
    100% { top: 10px;  opacity: 1; }
}

.success-banner {
    background: rgba(16,185,129,0.1);
    border: 1px solid rgba(16,185,129,0.28);
    border-radius: 14px;
    padding: 18px 24px;
    color: #34d399;
    font-weight: 600;
    font-size: 15px;
    text-align: center;
    animation: fadeUp 0.4s ease;
}
@keyframes fadeUp { from { opacity:0; transform:translateY(-10px); } to { opacity:1; transform:none; } }

/* ── Inputs ── */
.stTextInput input, .stNumberInput input, .stTextArea textarea {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(99,102,241,0.22) !important;
    border-radius: 12px !important;
    color: #e2e8f5 !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 14px !important;
}
.stTextInput input:focus, .stNumberInput input:focus {
    border-color: rgba(99,102,241,0.55) !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,0.12) !important;
}
.stTextInput label, .stNumberInput label, .stTextArea label,
.stSelectbox label, .stSlider label, .stMultiSelect label {
    color: #64748b !important;
    font-size: 12px !important;
    font-weight: 500 !important;
    letter-spacing: 0.5px !important;
    text-transform: uppercase !important;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #4f46e5, #0891b2) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 12px 28px !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    box-shadow: 0 4px 24px rgba(79,70,229,0.35) !important;
    transition: opacity 0.2s, transform 0.2s !important;
    letter-spacing: 0.3px !important;
}
.stButton > button:hover { opacity: 0.88 !important; transform: translateY(-1px) !important; }
.stDownloadButton > button {
    background: rgba(99,102,241,0.12) !important;
    border: 1px solid rgba(99,102,241,0.3) !important;
    color: #a5b4fc !important;
    border-radius: 12px !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(255,255,255,0.02) !important;
    border-radius: 14px !important;
    padding: 4px !important;
    border: 1px solid rgba(255,255,255,0.05) !important;
    gap: 4px !important;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 10px !important;
    color: #475569 !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 500 !important;
    padding: 8px 18px !important;
    transition: background 0.2s !important;
}
.stTabs [aria-selected="true"] {
    background: rgba(99,102,241,0.18) !important;
    color: #a5b4fc !important;
}

/* ── Metrics ── */
[data-testid="metric-container"] {
    background: rgba(255,255,255,0.025) !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
    border-radius: 16px !important;
    padding: 18px !important;
}
[data-testid="metric-container"] label {
    color: #475569 !important;
    font-size: 11px !important;
    letter-spacing: 1px !important;
    text-transform: uppercase !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #e2e8f5 !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 800 !important;
}

/* ── Sidebar role pill ── */
.sb-role {
    background: linear-gradient(135deg, rgba(79,70,229,0.12), rgba(6,182,212,0.08));
    border: 1px solid rgba(99,102,241,0.2);
    border-radius: 14px;
    padding: 14px 18px;
    margin-bottom: 14px;
}
.sb-role-name { font-family: 'Syne', sans-serif; font-size: 19px; font-weight: 700; color: #a5b4fc !important; }
.sb-role-user { font-size: 12px; color: #475569 !important; margin-top: 3px; font-family: 'DM Mono', monospace; }

/* ── Alert system ── */
.alert-critical {
    background: rgba(244,63,94,0.08);
    border: 1px solid rgba(244,63,94,0.28);
    border-left: 4px solid #f43f5e;
    border-radius: 14px;
    padding: 16px 20px;
    margin-bottom: 10px;
    animation: pulseAlert 2s ease-in-out infinite;
}
@keyframes pulseAlert {
    0%, 100% { box-shadow: 0 0 0 0 rgba(244,63,94,0.15); }
    50%       { box-shadow: 0 0 0 8px rgba(244,63,94,0.0); }
}
.alert-critical .alert-title { color: #fb7185; font-weight: 700; font-size: 14px; }
.alert-critical .alert-body  { color: #94a3b8; font-size: 13px; margin-top: 4px; }

hr { border-color: rgba(255,255,255,0.05) !important; }

/* ── Dataframe ── */
[data-testid="stDataFrame"] { border-radius: 14px !important; overflow: hidden !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────
_defaults = dict(
    logged_in=False, role=None, username=None,
    doc_record=None, doc_pid=None,
    pat_record=None, pat_pid=None,
    last_scanned_token=None,
    admin_tab="overview",
)
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ─────────────────────────────────────────────────────────────────────────────
#  DATABASE
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def get_supabase():
    if not SUPABASE_AVAILABLE or not SUPABASE_URL or not SUPABASE_KEY:
        return None
    try:
        return create_client(SUPABASE_URL, SUPABASE_KEY)
    except Exception:
        return None

def _demo_df() -> pd.DataFrame:
    """Return a rich demo dataset when no real DB is connected."""
    np.random.seed(99)
    N = 60
    pids   = [f"P{str(i+1).zfill(3)}" for i in range(N)]
    names  = [
        "Arjun Sharma","Priya Nair","Rahul Verma","Sneha Patel","Karthik R",
        "Meena Iyer","Suresh Kumar","Divya Menon","Anil Gupta","Roshni Das",
        "Vijay Singh","Lakshmi T","Deepak Rao","Ananya B","Mohan L",
        "Kavitha S","Rajesh P","Uma Devi","Sanjay M","Nithya K",
        "Ajith C","Saranya V","Prakash N","Geetha R","Manoj T",
        "Pooja A","Ramesh B","Sunita G","Harish D","Leela M",
        "Balu N","Chandni P","Ganesh Q","Hema S","Indira T",
        "Jagadish U","Kavya V","Lavanya W","Murali X","Nandini Y",
        "Om Z","Padma AA","Qureshi BB","Radha CC","Saravanan DD",
        "Tanya EE","Uday FF","Vaishali GG","Wasim HH","Xena II",
        "Yogesh JJ","Zara KK","Alex LL","Bella MM","Carlos NN",
        "Deepa OO","Ethan PP","Fatima QQ","Gautam RR","Hina SS"
    ][:N]
    diag_options = ["Normal", "Pre-Diabetic", "Diabetic", "Normal", "Pre-Diabetic"]
    rows = []
    for i in range(N):
        age = int(np.random.uniform(22, 78))
        bmi = round(np.random.uniform(18, 42), 1)
        hba = round(np.random.uniform(4.5, 12.0), 1)
        fbs = int(np.random.uniform(70, 320))
        sbp = int(np.random.uniform(100, 190))
        dbp = int(np.random.uniform(60, 120))
        chol= int(np.random.uniform(140, 360))
        ldl = int(chol * np.random.uniform(0.45, 0.65))
        hdl = int(np.random.uniform(30, 90))
        tg  = int(np.random.uniform(80, 400))
        smoke  = int(np.random.binomial(1, 0.3))
        alc    = round(np.random.uniform(0, 5), 1)
        pa     = round(np.random.uniform(0, 5), 1)
        diet   = round(np.random.uniform(0, 5), 1)
        sleep  = round(np.random.uniform(4, 9), 1)
        fam    = int(np.random.binomial(1, 0.35))
        hyp    = int(np.random.binomial(1, 0.3))
        gender = int(np.random.binomial(1, 0.5))
        diag   = diag_options[i % len(diag_options)]
        rows.append({
            "patient_id": pids[i], "name": names[i], "age": age, "gender": gender,
            "bmi": bmi, "hba1c": hba, "fasting_blood_sugar": fbs,
            "systolic_bp": sbp, "diastolic_bp": dbp,
            "cholesterol_total": chol, "cholesterol_ldl": ldl,
            "cholesterol_hdl": hdl, "cholesterol_triglycerides": tg,
            "smoking": smoke, "alcohol_consumption": alc, "physical_activity": pa,
            "diet_quality": diet, "sleep_quality": sleep,
            "family_history_diabetes": fam, "hypertension": hyp,
            "gestational_diabetes": 0, "polycystic_ovary_syndrome": 0,
            "previous_pre_diabetes": int(diag == "Pre-Diabetic"),
            "serum_creatinine": round(np.random.uniform(0.5, 2.5), 2),
            "bun_levels": int(np.random.uniform(8, 50)),
            "antihypertensive_medications": int(hyp and np.random.binomial(1, 0.7)),
            "statins": int(np.random.binomial(1, 0.2)),
            "antidiabetic_medications": int(diag == "Diabetic" and np.random.binomial(1, 0.75)),
            "frequent_urination": int(hba > 7.5),
            "excessive_thirst": int(fbs > 200),
            "unexplained_weight_loss": int(np.random.binomial(1, 0.1)),
            "fatigue_levels": round(np.random.uniform(0, 5), 1),
            "blurred_vision": int(np.random.binomial(1, 0.12)),
            "slow_healing_sores": int(np.random.binomial(1, 0.1)),
            "tingling_hands_feet": int(np.random.binomial(1, 0.14)),
            "quality_of_life_score": round(np.random.uniform(20, 95), 1),
            "medical_checkups_frequency": round(np.random.uniform(0, 5), 1),
            "medication_adherence": round(np.random.uniform(0, 5), 1),
            "health_literacy": round(np.random.uniform(0, 5), 1),
            "diagnosis": diag,
            "doctor_remarks": "",
            "created_at": (datetime.now() - timedelta(days=np.random.randint(1, 365))).isoformat(),
        })
    return pd.DataFrame(rows)

@st.cache_data(ttl=60)
def db_fetch_all() -> pd.DataFrame:
    sb = get_supabase()
    if sb:
        try:
            res = sb.table(TABLE_NAME).select("*").execute()
            if res.data:
                return pd.DataFrame(res.data)
        except Exception as e:
            st.sidebar.caption(f"⚠️ DB: {e}")
    return _demo_df()

def db_upsert(row: dict) -> bool:
    sb = get_supabase()
    if sb:
        try:
            sb.table(TABLE_NAME).upsert(row, on_conflict="patient_id").execute()
            return True
        except Exception as e:
            st.error(f"DB write error: {e}")
    return False

# ─────────────────────────────────────────────────────────────────────────────
#  JWT / QR AUTH
# ─────────────────────────────────────────────────────────────────────────────
def generate_secure_qr(patient_id: str) -> str:
    now = datetime.now(tz=timezone.utc)
    exp = now + timedelta(days=JWT_EXPIRY_DAYS)
    payload = {
        "patient_id": patient_id, "role": "patient",
        "iat": int(now.timestamp()), "exp": int(exp.timestamp()),
    }
    if JWT_AVAILABLE:
        token = pyjwt.encode(payload, JWT_SECRET, algorithm="HS256")
    else:
        def b64url(data):
            if isinstance(data, str): data = data.encode()
            return base64.urlsafe_b64encode(data).rstrip(b"=").decode()
        header  = b64url(json.dumps({"alg":"HS256","typ":"JWT"}, separators=(',',':')))
        body    = b64url(json.dumps(payload, separators=(',',':')))
        signing = f"{header}.{body}"
        sig     = hmac.new(JWT_SECRET.encode(), signing.encode(), hashlib.sha256).digest()
        token   = f"{signing}.{b64url(sig)}"
    path = os.path.join(QR_FOLDER, f"{patient_id}_jwt.png")
    qr = qrcode.QRCode(version=None, error_correction=qrcode.constants.ERROR_CORRECT_M, box_size=8, border=3)
    qr.add_data(token)
    qr.make(fit=True)
    img = qr.make_image(fill_color="#6366f1", back_color="white")
    img.save(path)
    return path

def verify_qr_token(token: str) -> dict | None:
    if not token or not isinstance(token, str):
        return None
    try:
        if JWT_AVAILABLE:
            payload = pyjwt.decode(token, JWT_SECRET, algorithms=["HS256"])
            return payload if payload.get("role") == "patient" else None
        else:
            parts = token.split(".")
            if len(parts) != 3: return None
            h, b, s = parts
            signing = f"{h}.{b}"
            expected = hmac.new(JWT_SECRET.encode(), signing.encode(), hashlib.sha256).digest()
            pad = 4 - len(s) % 4
            actual = base64.urlsafe_b64decode(s + "=" * (pad % 4))
            if not hmac.compare_digest(expected, actual): return None
            pad = 4 - len(b) % 4
            payload = json.loads(base64.urlsafe_b64decode(b + "=" * (pad % 4)))
            if payload.get("exp", 0) < time.time(): return None
            return payload if payload.get("role") == "patient" else None
    except Exception:
        return None

def qr_from_image_file(pil_img) -> str | None:
    if PYZBAR_AVAILABLE:
        try:
            arr = np.array(pil_img.convert("RGB"))
            for r in pyzbar_lib.decode(arr):
                if r.type == "QRCODE":
                    return r.data.decode("utf-8")
        except Exception:
            pass
    return None

# ─────────────────────────────────────────────────────────────────────────────
#  ML MODEL
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="🧠 Training ML model…")
def load_or_train_model():
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        try:
            m = joblib.load(MODEL_PATH); sc = joblib.load(SCALER_PATH)
            return m, sc, list(getattr(m,"feature_names_in_",ML_FEATURES)), None, "loaded"
        except Exception:
            pass
    np.random.seed(42); N = 8000
    age = np.random.normal(50,15,N).clip(18,90);   bmi = np.random.normal(27,6,N).clip(15,55)
    hba = np.random.normal(5.8,1.2,N).clip(4,14);  fbs = np.random.normal(110,35,N).clip(60,400)
    sbp = np.random.normal(128,20,N).clip(80,220);  dbp = np.random.normal(82,12,N).clip(50,130)
    chol= np.random.normal(195,40,N).clip(100,400); ldl = chol*np.random.uniform(0.45,0.65,N)
    hdl = np.random.normal(50,12,N).clip(20,100);   tg  = np.random.normal(150,60,N).clip(50,500)
    creat=np.random.normal(0.95,0.25,N).clip(0.4,3);bun = np.random.normal(16,5,N).clip(5,60)
    smoke=np.random.binomial(1,0.22,N);alc=np.random.uniform(0,5,N);pa=np.random.uniform(0,5,N)
    diet=np.random.uniform(0,5,N);sleep=np.random.uniform(4,9,N)
    fam=np.random.binomial(1,0.30,N);gest=np.random.binomial(1,0.08,N)
    pcos=np.random.binomial(1,0.07,N);prev=np.random.binomial(1,0.18,N)
    hyp=np.random.binomial(1,0.30,N);antihyp=np.random.binomial(1,0.25,N)
    statins=np.random.binomial(1,0.20,N);antidiab=np.random.binomial(1,0.15,N)
    freq_ur=np.random.binomial(1,0.20,N);ex_thr=np.random.binomial(1,0.15,N)
    wt_loss=np.random.binomial(1,0.12,N);fatigue=np.random.uniform(0,5,N)
    blur=np.random.binomial(1,0.12,N);sores=np.random.binomial(1,0.10,N)
    tingle=np.random.binomial(1,0.14,N);qol=np.random.uniform(0,100,N)
    mcu=np.random.uniform(0,5,N);madh=np.random.uniform(0,5,N)
    hlitt=np.random.uniform(0,5,N);gender=np.random.binomial(1,0.50,N)
    score=((hba-4.5)*0.40+(fbs-70)*0.0025+(bmi-18)*0.035+(age-18)*0.010
           +(sbp-90)*0.008+fam*0.55+smoke*0.30+prev*0.50+hyp*0.25+gest*0.45+pcos*0.40-pa*0.08-diet*0.06)
    prob_t=1/(1+np.exp(-(score-3.5)))
    label=(np.random.uniform(0,1,N)<prob_t).astype(int)
    X=np.column_stack([age,gender,bmi,smoke,alc,pa,diet,sleep,fam,gest,pcos,prev,hyp,
                       sbp,dbp,fbs,hba,creat,bun,chol,ldl,hdl,tg,antihyp,statins,
                       antidiab,freq_ur,ex_thr,wt_loss,fatigue,blur,sores,tingle,
                       qol,mcu,madh,hlitt])
    X_tr,X_te,y_tr,y_te=train_test_split(X,label,test_size=0.2,random_state=42)
    sc=StandardScaler(); X_tr_s=sc.fit_transform(X_tr); X_te_s=sc.transform(X_te)
    m=RandomForestClassifier(n_estimators=300,max_depth=12,min_samples_leaf=5,n_jobs=-1,random_state=42)
    m.fit(X_tr_s,y_tr)
    acc=accuracy_score(y_te,m.predict(X_te_s))
    m.feature_names_in_=np.array(ML_FEATURES)
    try: joblib.dump(m,MODEL_PATH); joblib.dump(sc,SCALER_PATH)
    except Exception: pass
    return m, sc, ML_FEATURES, round(acc*100,1), "trained"

MODEL, SCALER, FEATURES, MODEL_ACC, MODEL_SRC = load_or_train_model()

# ─────────────────────────────────────────────────────────────────────────────
#  SEVERITY SCORING ENGINE  (0–100 composite score)
# ─────────────────────────────────────────────────────────────────────────────
def severity_score(rec: dict) -> float:
    """
    Clinically-weighted composite severity score 0–100.
    Weights:
        HbA1c            → 30 pts  (most critical)
        Fasting glucose  → 20 pts
        BMI              → 15 pts
        Systolic BP      → 15 pts
        Lifestyle        → 12 pts  (smoking, activity, diet, sleep)
        Family / History →  8 pts
    """
    def g(k1, k2=None, default=0.0):
        v = rec.get(k1, rec.get(k2, default) if k2 else default)
        try: return float(v) if v not in (None, "") else float(default)
        except: return float(default)

    # HbA1c: <5.7=0, 5.7–6.4=15, 6.5–7.9=22, 8–9.9=27, ≥10=30
    hba = g("hba1c","HbA1c",5.5)
    if   hba >= 10.0: hba_s = 30.0
    elif hba >= 8.0:  hba_s = 27.0
    elif hba >= 6.5:  hba_s = 22.0
    elif hba >= 5.7:  hba_s = 15.0
    else:             hba_s = 0.0

    # Fasting glucose: <100=0, 100–125=8, 126–179=14, ≥180=20
    fbs = g("fasting_blood_sugar","FastingBloodSugar",90)
    if   fbs >= 180: fbs_s = 20.0
    elif fbs >= 126: fbs_s = 14.0
    elif fbs >= 100: fbs_s = 8.0
    else:            fbs_s = 0.0

    # BMI: <18.5=5(underweight), 18.5–24.9=0, 25–29.9=5, 30–34.9=10, ≥35=15
    bmi = g("bmi","BMI",22)
    if   bmi >= 35:   bmi_s = 15.0
    elif bmi >= 30:   bmi_s = 10.0
    elif bmi >= 25:   bmi_s = 5.0
    elif bmi < 18.5:  bmi_s = 5.0
    else:             bmi_s = 0.0

    # Systolic BP: <120=0, 120–129=5, 130–139=9, 140–159=12, ≥160=15
    sbp = g("systolic_bp","SystolicBP",120)
    if   sbp >= 160: sbp_s = 15.0
    elif sbp >= 140: sbp_s = 12.0
    elif sbp >= 130: sbp_s = 9.0
    elif sbp >= 120: sbp_s = 5.0
    else:            sbp_s = 0.0

    # Lifestyle (12 pts max) — higher activity/diet/sleep = better
    smoke  = g("smoking","Smoking",0)
    pa     = g("physical_activity","PhysicalActivity",2.5)     # 0–5
    diet   = g("diet_quality","DietQuality",2.5)               # 0–5
    sleep  = g("sleep_quality","SleepQuality",6)               # hrs
    ls_s   = smoke * 4.0 + (1 - pa/5) * 3.0 + (1 - diet/5) * 3.0 + max(0, (1 - sleep/8)) * 2.0
    ls_s   = min(ls_s, 12.0)

    # Family / history (8 pts)
    fam    = g("family_history_diabetes","FamilyHistoryDiabetes",0)
    prev   = g("previous_pre_diabetes","PreviousPreDiabetes",0)
    hist_s = fam * 4.0 + prev * 4.0

    total = hba_s + fbs_s + bmi_s + sbp_s + ls_s + hist_s
    return round(min(max(total, 0), 100), 1)

def sev_category(score: float):
    if score >= 61: return "High",   "#f43f5e", "🔴"
    if score >= 31: return "Medium", "#f59e0b", "🟡"
    return             "Low",    "#10b981", "🟢"

# ─────────────────────────────────────────────────────────────────────────────
#  ML UTILITIES
# ─────────────────────────────────────────────────────────────────────────────
def sf(x, d=0.0):
    try:
        if x is None or x=="" or (isinstance(x,float) and np.isnan(x)): return float(d)
        return float(x)
    except: return float(d)

def patient_to_feature_vec(rec):
    mapping = {
        "Age":["age","Age"],"Gender":["gender","Gender"],"BMI":["bmi","BMI"],
        "Smoking":["smoking","Smoking"],"AlcoholConsumption":["alcohol_consumption","AlcoholConsumption"],
        "PhysicalActivity":["physical_activity","PhysicalActivity"],"DietQuality":["diet_quality","DietQuality"],
        "SleepQuality":["sleep_quality","SleepQuality"],"FamilyHistoryDiabetes":["family_history_diabetes","FamilyHistoryDiabetes"],
        "GestationalDiabetes":["gestational_diabetes","GestationalDiabetes"],
        "PolycysticOvarySyndrome":["polycystic_ovary_syndrome","PolycysticOvarySyndrome","pcos"],
        "PreviousPreDiabetes":["previous_pre_diabetes","PreviousPreDiabetes"],
        "Hypertension":["hypertension","Hypertension"],"SystolicBP":["systolic_bp","SystolicBP"],
        "DiastolicBP":["diastolic_bp","DiastolicBP"],"FastingBloodSugar":["fasting_blood_sugar","FastingBloodSugar"],
        "HbA1c":["hba1c","HbA1c"],"SerumCreatinine":["serum_creatinine","SerumCreatinine"],
        "BUNLevels":["bun_levels","BUNLevels"],"CholesterolTotal":["cholesterol_total","CholesterolTotal"],
        "CholesterolLDL":["cholesterol_ldl","CholesterolLDL"],"CholesterolHDL":["cholesterol_hdl","CholesterolHDL"],
        "CholesterolTriglycerides":["cholesterol_triglycerides","CholesterolTriglycerides"],
        "AntihypertensiveMedications":["antihypertensive_medications","AntihypertensiveMedications"],
        "Statins":["statins","Statins"],"AntidiabeticMedications":["antidiabetic_medications","AntidiabeticMedications"],
        "FrequentUrination":["frequent_urination","FrequentUrination"],"ExcessiveThirst":["excessive_thirst","ExcessiveThirst"],
        "UnexplainedWeightLoss":["unexplained_weight_loss","UnexplainedWeightLoss"],
        "FatigueLevels":["fatigue_levels","FatigueLevels"],"BlurredVision":["blurred_vision","BlurredVision"],
        "SlowHealingSores":["slow_healing_sores","SlowHealingSores"],"TinglingHandsFeet":["tingling_hands_feet","TinglingHandsFeet"],
        "QualityOfLifeScore":["quality_of_life_score","QualityOfLifeScore"],
        "MedicalCheckupsFrequency":["medical_checkups_frequency","MedicalCheckupsFrequency"],
        "MedicationAdherence":["medication_adherence","MedicationAdherence"],
        "HealthLiteracy":["health_literacy","HealthLiteracy"],
    }
    vec = []
    for feat in FEATURES:
        keys = mapping.get(feat,[feat.lower(),feat])
        val = None
        for k in keys:
            if k in rec: val=rec[k]; break
        vec.append(sf(val))
    return np.array(vec, dtype=float)

def predict_prob(rec):
    vec = patient_to_feature_vec(rec).reshape(1,-1)
    try:
        return float(MODEL.predict_proba(SCALER.transform(vec))[0][1]), "model"
    except Exception:
        hba = sf(rec.get("hba1c",rec.get("HbA1c",5.5)))
        fbs = sf(rec.get("fasting_blood_sugar",rec.get("FastingBloodSugar",100)))
        p = min(1.0, max(0.0, (hba-4.5)/6.0*0.6 + (fbs-70)/200.0*0.4))
        return round(p,4), "heuristic"

def risk_tier(p):
    if p >= 0.65: return "High",     "#f43f5e", "🔴"
    if p >= 0.35: return "Moderate", "#f59e0b", "🟡"
    return              "Low",      "#10b981", "🟢"

def simulate_projection(rec, years=5):
    pts = []
    for y in range(1, years+1):
        m = dict(rec)
        m["bmi"] = str(sf(m.get("bmi",25)) + 0.25*y)
        m["fasting_blood_sugar"] = str(sf(m.get("fasting_blood_sugar",100)) + 2*y)
        p, _ = predict_prob(m)
        pts.append(round(p*100, 1))
    return pts

def fmt(v, decimals=2):
    if v is None or v == "": return "—"
    try: return f"{float(v):.{decimals}f}"
    except: return str(v)

def enrich_df(df: pd.DataFrame) -> pd.DataFrame:
    """Add severity_score, severity_label, ml_risk columns to DataFrame."""
    sev_scores, sev_labels, ml_risks = [], [], []
    for _, row in df.iterrows():
        rec  = row.to_dict()
        sc   = severity_score(rec)
        cat, _, _ = sev_category(sc)
        prob, _   = predict_prob(rec)
        sev_scores.append(sc)
        sev_labels.append(cat)
        ml_risks.append(round(prob * 100, 1))
    df = df.copy()
    df["severity_score"] = sev_scores
    df["severity_label"] = sev_labels
    df["ml_risk_pct"]    = ml_risks
    return df

# ─────────────────────────────────────────────────────────────────────────────
#  AI INSIGHTS ENGINE
# ─────────────────────────────────────────────────────────────────────────────
def build_ai_insights(rec: dict, prob: float, sev: float) -> dict:
    """Return structured AI explanation for a patient record."""
    hba = sf(rec.get("hba1c","HbA1c"), 5.5)
    fbs = sf(rec.get("fasting_blood_sugar", rec.get("FastingBloodSugar", 100)))
    bmi = sf(rec.get("bmi", rec.get("BMI", 25)))
    sbp = sf(rec.get("systolic_bp", rec.get("SystolicBP", 120)))
    smoke  = sf(rec.get("smoking", rec.get("Smoking", 0)))
    fam    = sf(rec.get("family_history_diabetes", rec.get("FamilyHistoryDiabetes", 0)))
    pa     = sf(rec.get("physical_activity", rec.get("PhysicalActivity", 2.5)))
    tier, color, icon = risk_tier(prob)

    factors = []
    if hba >= 6.5:  factors.append(("HbA1c", f"{hba:.1f}%", "Diabetes range — primary driver", "#f43f5e"))
    elif hba >= 5.7:factors.append(("HbA1c", f"{hba:.1f}%", "Pre-diabetes range",              "#f59e0b"))
    if fbs >= 126:  factors.append(("Fasting Glucose", f"{fbs:.0f} mg/dL", "Elevated — significant risk", "#f43f5e"))
    elif fbs >= 100:factors.append(("Fasting Glucose", f"{fbs:.0f} mg/dL", "Impaired fasting glucose",     "#f59e0b"))
    if bmi >= 30:   factors.append(("BMI", f"{bmi:.1f}", "Obese — strong diabetes predictor",  "#f43f5e"))
    elif bmi >= 25: factors.append(("BMI", f"{bmi:.1f}", "Overweight — moderate risk",         "#f59e0b"))
    if sbp >= 140:  factors.append(("Systolic BP", f"{sbp:.0f} mmHg", "Stage 2 hypertension", "#f43f5e"))
    elif sbp >= 130:factors.append(("Systolic BP", f"{sbp:.0f} mmHg", "Stage 1 hypertension", "#f59e0b"))
    if smoke:       factors.append(("Smoking", "Yes", "Increases insulin resistance",          "#f59e0b"))
    if fam:         factors.append(("Family History", "Positive", "Genetic predisposition",    "#f59e0b"))

    # Recommendations
    diet_recs    = []
    exercise_recs= []
    medical_recs = []
    if hba >= 6.5 or fbs >= 126:
        diet_recs.append("Follow a low-glycaemic index (GI) diet — reduce white rice, refined flour, sugary drinks")
        medical_recs.append("Urgent endocrinology review for medication/insulin assessment")
    if bmi >= 25:
        diet_recs.append("Caloric deficit of 300–500 kcal/day; prioritise fibre-rich foods")
        exercise_recs.append("30 min moderate cardio (walking, cycling) 5×/week")
    if pa < 2.0:
        exercise_recs.append("Gradual activity ramp-up: start with 10-min post-meal walks")
    if sbp >= 130:
        diet_recs.append("Low-sodium diet (<1500 mg/day); limit processed foods")
        medical_recs.append("Blood pressure medication review within 1 month")
    if not diet_recs:    diet_recs.append("Maintain current balanced diet; continue routine monitoring")
    if not exercise_recs:exercise_recs.append("Maintain current activity level; aim for 150 min/week")
    if not medical_recs: medical_recs.append("Annual screening; next HbA1c in 12 months")

    follow_up = {
        "High":     "Within 2 weeks",
        "Moderate": "Within 3 months",
        "Low":      "Annual check-up",
    }[tier]

    return {
        "tier": tier, "color": color, "icon": icon, "prob": prob, "sev": sev,
        "factors": factors,
        "diet":     diet_recs,
        "exercise": exercise_recs,
        "medical":  medical_recs,
        "follow_up": follow_up,
    }

# ─────────────────────────────────────────────────────────────────────────────
#  CHARTS — shared helpers
# ─────────────────────────────────────────────────────────────────────────────
CHART_STYLE = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#64748b", family="Space Grotesk"),
    margin=dict(t=44, b=20, l=20, r=20),
)

def apply_chart_style(fig, title="", height=260):
    fig.update_layout(
        **CHART_STYLE,
        height=height,
        title=dict(text=title, font=dict(color="#94a3b8", size=13, family="Space Grotesk")),
        xaxis=dict(gridcolor="rgba(255,255,255,0.04)", color="#475569"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.04)", color="#475569"),
    )
    return fig

def gauge_chart(prob):
    tier, color, _ = risk_tier(prob)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(prob*100, 1),
        number={"suffix":"%","font":{"color":"white","family":"Syne","size":38}},
        gauge={
            "axis":{"range":[0,100],"tickcolor":"#334155","tickfont":{"color":"#334155","size":9}},
            "bar":{"color":color,"thickness":0.22},
            "bgcolor":"rgba(0,0,0,0)","bordercolor":"rgba(0,0,0,0)",
            "steps":[
                {"range":[0,35],  "color":"rgba(16,185,129,0.07)"},
                {"range":[35,65], "color":"rgba(245,158,11,0.07)"},
                {"range":[65,100],"color":"rgba(244,63,94,0.07)"},
            ],
            "threshold":{"line":{"color":color,"width":3},"value":prob*100},
        },
        title={"text":f"<b>{tier} Risk</b>","font":{"color":color,"family":"Space Grotesk","size":14}},
    ))
    fig.update_layout(height=240, margin=dict(t=44,b=8,l=20,r=20), paper_bgcolor="rgba(0,0,0,0)", font=dict(color="white"))
    return fig

def proj_chart(pts):
    colors = ["#10b981" if p<35 else ("#f59e0b" if p<65 else "#f43f5e") for p in pts]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(1,6)), y=pts, mode="lines+markers",
        line=dict(color="#6366f1", width=3, shape="spline"),
        marker=dict(size=9, color=colors, line=dict(color="rgba(255,255,255,0.15)", width=2)),
        fill="tozeroy", fillcolor="rgba(99,102,241,0.05)", name="Risk %"
    ))
    fig.add_hline(y=65, line=dict(color="#f43f5e", dash="dot", width=1), annotation_text="High",     annotation_font_color="#f43f5e")
    fig.add_hline(y=35, line=dict(color="#f59e0b", dash="dot", width=1), annotation_text="Moderate", annotation_font_color="#f59e0b")
    fig = apply_chart_style(fig, "5-Year Risk Projection", 260)
    fig.update_xaxis(title="Year", tickvals=list(range(1,6)))
    fig.update_yaxis(title="Risk %", range=[0,100])
    return fig

def vitals_radar(rec):
    cats = ["HbA1c","BMI","BP","Cholesterol","Glucose","Age"]
    vals = [
        min(sf(rec.get("hba1c",5.5))/14*100, 100),
        min(sf(rec.get("bmi",25))/50*100, 100),
        min(sf(rec.get("systolic_bp",120))/200*100, 100),
        min(sf(rec.get("cholesterol_total",180))/400*100, 100),
        min(sf(rec.get("fasting_blood_sugar",100))/300*100, 100),
        min(sf(rec.get("age",40))/100*100, 100),
    ]
    fig = go.Figure(go.Scatterpolar(
        r=vals+[vals[0]], theta=cats+[cats[0]], fill="toself",
        fillcolor="rgba(99,102,241,0.1)",
        line=dict(color="#6366f1", width=2),
        marker=dict(color="#a5b4fc", size=5),
    ))
    fig.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(visible=True,range=[0,100],gridcolor="rgba(255,255,255,0.05)",tickfont=dict(color="#334155",size=8)),
            angularaxis=dict(gridcolor="rgba(255,255,255,0.05)",tickfont=dict(color="#94a3b8",size=10)),
        ),
        paper_bgcolor="rgba(0,0,0,0)", font=dict(color="white"),
        height=270, margin=dict(t=20,b=20,l=30,r=30), showlegend=False,
        title=dict(text="Health Profile Radar", font=dict(color="#94a3b8",size=12)),
    )
    return fig

# ─────────────────────────────────────────────────────────────────────────────
#  LAYOUT HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def render_header(subtitle="Smart Diabetes EMR"):
    sb     = "🟢 Connected" if get_supabase() else "🟡 Demo Mode"
    ml_lbl = f"{'Loaded' if MODEL_SRC=='loaded' else 'Auto-trained'}"
    ml_acc = f" · {MODEL_ACC}%" if MODEL_ACC else ""
    jwt_lbl= "🔐 JWT Active" if JWT_AVAILABLE else "🔐 JWT (fallback)"
    st.markdown(f"""
    <div class="topbar">
        <div style="font-size:34px;line-height:1">🩺</div>
        <div style="flex:1">
            <div class="topbar-logo">GLUCO-LENS</div>
            <div class="topbar-sub">{subtitle}</div>
        </div>
        <div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap;justify-content:flex-end">
            <span class="badge {'badge-green' if 'Connected' in sb else 'badge-yellow'}">{sb}</span>
            <span class="badge badge-indigo">🧠 RF {ml_lbl}{ml_acc}</span>
            <span class="badge badge-cyan">{jwt_lbl}</span>
        </div>
    </div>""", unsafe_allow_html=True)

def render_sidebar():
    with st.sidebar:
        role_icon = {"Admin":"👑","Doctor":"👩‍⚕️","Patient":"🫀"}.get(st.session_state.role,"")
        uname = st.session_state.username or st.session_state.pat_pid or "—"
        st.markdown(f"""
        <div class="sb-role">
            <div class="sb-role-name">{role_icon} {st.session_state.role}</div>
            <div class="sb-role-user">@{uname}</div>
        </div>""", unsafe_allow_html=True)
        st.markdown("---")
        if st.button("🚪 Sign Out", use_container_width=True):
            for k in _defaults: st.session_state[k] = _defaults[k]
            st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
#  LOGIN SCREEN
# ─────────────────────────────────────────────────────────────────────────────
def show_login():
    render_header("Secure Healthcare Login")
    _, mid, _ = st.columns([1, 1.3, 1])
    with mid:
        st.markdown('<div class="login-card">', unsafe_allow_html=True)
        st.markdown('<div class="login-logo">GLUCO-LENS</div>', unsafe_allow_html=True)
        st.markdown('<div class="login-tagline">Smart Diabetes EMR · v5</div>', unsafe_allow_html=True)

        tab_staff, tab_patient = st.tabs(["👩‍⚕️  Staff Login", "📷  Patient QR"])

        with tab_staff:
            username = st.text_input("Username", placeholder="admin or doctor", key="li_user")
            password = st.text_input("Password", type="password", key="li_pass")
            if st.button("Sign In →", use_container_width=True, type="primary", key="li_btn"):
                u = username.lower().strip()
                if u in CREDENTIALS and CREDENTIALS[u]["password"] == password:
                    st.session_state.logged_in = True
                    st.session_state.role      = CREDENTIALS[u]["role"]
                    st.session_state.username  = u
                    st.rerun()
                else:
                    st.error("Invalid credentials. Try admin/admin123 or doctor/doc123")
            st.markdown('<div class="cred-hint">admin / admin123<br>doctor / doc123</div>', unsafe_allow_html=True)

        with tab_patient:
            _patient_qr_login_widget()

        st.markdown("</div>", unsafe_allow_html=True)

def _patient_qr_login_widget():
    raw_token = st.session_state.get("last_scanned_token")
    if raw_token and not st.session_state.logged_in:
        _attempt_jwt_login(raw_token)
        return
    st.markdown("""
    <div class="scan-zone">
        <div class="scan-corner tl"></div><div class="scan-corner tr"></div>
        <div class="scan-corner bl"></div><div class="scan-corner br"></div>
        <div class="scan-line"></div>
        <div style="color:#475569;font-size:13px;margin-bottom:12px;">Upload your secure QR</div>
    </div>""", unsafe_allow_html=True)
    uploaded = st.file_uploader("QR Image", type=["png","jpg","jpeg"],
                                label_visibility="collapsed", key="qr_upload_login")
    if uploaded:
        img   = Image.open(uploaded)
        token = qr_from_image_file(img)
        if token:
            st.session_state.last_scanned_token = token
            _attempt_jwt_login(token)
        else:
            st.error("❌ Could not decode QR. Ensure pyzbar/libzbar is installed.")

def _attempt_jwt_login(token: str):
    payload = verify_qr_token(token)
    if payload:
        pid = payload.get("patient_id","")
        df  = db_fetch_all()
        pid_col = next((c for c in df.columns if c.lower()=="patient_id"), None)
        rec = None
        if pid_col:
            rows = df[df[pid_col].astype(str).str.strip() == pid]
            if not rows.empty:
                rec = rows.iloc[0].to_dict()
        if rec:
            st.session_state.logged_in  = True
            st.session_state.role       = "Patient"
            st.session_state.pat_pid    = pid
            st.session_state.pat_record = rec
            st.session_state.last_scanned_token = None
            st.markdown('<div class="success-banner">✅ Secured login via QR · Passwordless</div>', unsafe_allow_html=True)
            time.sleep(0.8)
            st.rerun()
        else:
            st.error("⚠️ QR verified but patient record not found.")
    else:
        st.error("❌ Invalid or expired QR token.")
        st.session_state.last_scanned_token = None

# ─────────────────────────────────────────────────────────────────────────────
#  PATIENT PORTAL
# ─────────────────────────────────────────────────────────────────────────────
def patient_portal():
    render_header("Patient Portal")
    render_sidebar()
    pid = st.session_state.pat_pid
    rec = st.session_state.pat_record
    if not pid or not rec:
        st.error("⚠️ Session error. Please log out and scan your QR again.")
        return
    _render_patient_record(rec, pid)

def _render_patient_record(rec, pid):
    prob, _   = predict_prob(rec)
    sev        = severity_score(rec)
    tier, color, icon = risk_tier(prob)
    sev_cat, sev_color, sev_icon = sev_category(sev)
    name = rec.get("name", rec.get("Name", pid))
    insights = build_ai_insights(rec, prob, sev)

    # ── Patient hero card ──
    st.markdown(f"""
    <div class="card" style="border-left:3px solid {color};">
        <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:20px;">
            <div>
                <div style="font-family:'Syne',sans-serif;font-size:26px;font-weight:800;color:#e2e8f5;">
                    Welcome back, {name} 👋
                </div>
                <div style="font-size:12px;color:#475569;margin-top:6px;font-family:'DM Mono',monospace;">
                    ID: {pid}
                </div>
            </div>
            <div style="display:flex;gap:16px;">
                <div style="text-align:center;padding:18px 28px;background:rgba(255,255,255,0.03);
                            border-radius:16px;border:1px solid rgba(255,255,255,0.06);">
                    <div style="font-family:'Syne',sans-serif;font-size:38px;font-weight:800;color:{color};">
                        {prob*100:.1f}%
                    </div>
                    <div style="font-size:11px;color:{color};letter-spacing:1px;margin-top:4px;text-transform:uppercase;">
                        {icon} ML Risk
                    </div>
                </div>
                <div style="text-align:center;padding:18px 28px;background:rgba(255,255,255,0.03);
                            border-radius:16px;border:1px solid rgba(255,255,255,0.06);">
                    <div style="font-family:'Syne',sans-serif;font-size:38px;font-weight:800;color:{sev_color};">
                        {sev:.0f}
                    </div>
                    <div style="font-size:11px;color:{sev_color};letter-spacing:1px;margin-top:4px;text-transform:uppercase;">
                        {sev_icon} Severity
                    </div>
                </div>
            </div>
        </div>
    </div>""", unsafe_allow_html=True)

    # ── Vitals strip ──
    v_keys = [
        ("fasting_blood_sugar","FastingBloodSugar","Glucose","mg/dL"),
        ("hba1c","HbA1c","HbA1c","%"),
        ("bmi","BMI","BMI",""),
        ("cholesterol_total","CholesterolTotal","Cholesterol","mg/dL"),
        ("systolic_bp","SystolicBP","Systolic BP","mmHg"),
        ("diastolic_bp","DiastolicBP","Diastolic BP","mmHg"),
    ]
    cols = st.columns(len(v_keys))
    for i,(k1,k2,lbl,unit) in enumerate(v_keys):
        val = rec.get(k1, rec.get(k2,""))
        cols[i].metric(f"{lbl} ({unit})" if unit else lbl, fmt(val))

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Charts row ──
    cl, cr = st.columns([3, 2])
    with cl: st.plotly_chart(proj_chart(simulate_projection(rec)), use_container_width=True)
    with cr:
        st.plotly_chart(gauge_chart(prob), use_container_width=True)
        st.plotly_chart(vitals_radar(rec), use_container_width=True)

    # ── AI Insights ──
    st.markdown('<p class="sec-head">🤖 AI Health Insights</p>', unsafe_allow_html=True)
    _render_ai_insights(insights)

    # ── Doctor remarks ──
    remarks = rec.get("doctor_remarks","")
    if remarks:
        st.markdown(f"""
        <div class="card card-accent-cyan">
            <p class="sec-head" style="margin-bottom:10px;">📝 Doctor's Notes</p>
            <div style="color:#94a3b8;font-size:14px;line-height:1.75;">{remarks}</div>
        </div>""", unsafe_allow_html=True)

    # ── Download ──
    st.download_button(
        "⬇️ Download My Record",
        data=pd.DataFrame([{k:v for k,v in rec.items() if not k.startswith("_")}]).to_csv(index=False),
        file_name=f"{pid}_record.csv", mime="text/csv",
    )

    # ── Lifestyle estimator ──
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<p class="sec-head">🌿 Lifestyle Risk Estimator</p>', unsafe_allow_html=True)
    with st.form("lifestyle_form"):
        c1, c2 = st.columns(2)
        q1 = c1.select_slider("Exercise (mins/day)", ["0","10-30","30-60","60+"], value="30-60")
        q2 = c2.select_slider("Added sugar (tsp/day)", ["0-5","6-10","11-20","20+"], value="6-10")
        q3 = c1.select_slider("Veg/fruit servings/day", ["0-1","2-3","4-5","5+"], value="2-3")
        q4 = c2.select_slider("Sleep (hrs/night)", ["<5","5-6","6-7","7+"], value="6-7")
        q5 = c1.select_slider("Fast food frequency", ["Daily","Few/week","Weekly","Rarely"], value="Few/week")
        q6 = c2.select_slider("Stress level", ["0-2","3-5","6-8","9-10"], value="3-5")
        sub = st.form_submit_button("Calculate Impact →", use_container_width=True)
    if sub:
        sm = {"0":0,"10-30":1,"30-60":2,"60+":3,"0-5":3,"6-10":2,"11-20":1,"20+":0,
              "0-1":0,"2-3":1,"4-5":2,"5+":3,"<5":0,"5-6":1,"6-7":2,"7+":3,
              "Daily":0,"Few/week":1,"Weekly":2,"Rarely":3,"0-2":3,"3-5":2,"6-8":1,"9-10":0}
        total = sum(sm.get(q,0) for q in [q1,q2,q3,q4,q5,q6])
        mod = (total-9)/18.0; new_p = float(np.clip(prob-mod*0.25,0,1)); diff = new_p-prob
        hba_b = sf(rec.get("hba1c",5.5)); hba_n = hba_b + diff*2
        ca,cb,cc = st.columns(3)
        ca.metric("Current Risk",    f"{prob*100:.1f}%")
        cb.metric("Adjusted Risk",   f"{new_p*100:.1f}%", delta=f"{diff*100:+.1f}%")
        cc.metric("Est. HbA1c",      f"{hba_n:.2f}%",     delta=f"{(hba_n-hba_b):+.2f}")
        lm = {"Exercise":sm[q1],"Sugar":sm[q2],"Veg/Fruit":sm[q3],"Sleep":sm[q4],"Fast Food":sm[q5],"Stress":sm[q6]}
        fig = px.bar(pd.DataFrame({"Factor":list(lm.keys()),"Score":list(lm.values())}),
                     x="Factor", y="Score", color="Score",
                     color_continuous_scale=[[0,"#f43f5e"],[0.5,"#f59e0b"],[1,"#10b981"]],
                     title="Lifestyle Score (0=poor · 3=excellent)")
        apply_chart_style(fig, height=250)
        st.plotly_chart(fig, use_container_width=True)

def _render_ai_insights(ins: dict):
    color = ins["color"]
    st.markdown(f"""
    <div class="ai-box">
        <div style="font-size:15px;font-weight:700;color:{color};margin-bottom:14px;">
            {ins['icon']} {ins['tier']} Risk · ML Probability {ins['prob']*100:.1f}% · 
            Severity Score {ins['sev']:.0f}/100
        </div>
    """, unsafe_allow_html=True)

    if ins["factors"]:
        st.markdown("<div style='font-size:12px;color:#475569;letter-spacing:1px;text-transform:uppercase;margin-bottom:8px;'>Key Risk Drivers</div>", unsafe_allow_html=True)
        cols = st.columns(min(len(ins["factors"]), 3))
        for i, (name, val, desc, fc) in enumerate(ins["factors"]):
            cols[i % 3].markdown(f"""
            <div class="ai-insight-row">
                <div class="label">{name}</div>
                <div class="value" style="color:{fc};font-weight:700;">{val}</div>
                <div style="font-size:11px;color:#475569;margin-top:2px;">{desc}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    r1, r2, r3 = st.columns(3)
    with r1:
        st.markdown(f"""
        <div class="card card-accent-green" style="height:100%">
            <div class="sec-head" style="font-size:13px;color:#34d399;">🥗 Diet</div>
            {''.join(f'<div style="font-size:13px;color:#94a3b8;margin-bottom:6px;">• {d}</div>' for d in ins['diet'])}
        </div>""", unsafe_allow_html=True)
    with r2:
        st.markdown(f"""
        <div class="card card-accent-cyan" style="height:100%">
            <div class="sec-head" style="font-size:13px;color:#22d3ee;">🏃 Exercise</div>
            {''.join(f'<div style="font-size:13px;color:#94a3b8;margin-bottom:6px;">• {d}</div>' for d in ins['exercise'])}
        </div>""", unsafe_allow_html=True)
    with r3:
        st.markdown(f"""
        <div class="card card-accent-violet" style="height:100%">
            <div class="sec-head" style="font-size:13px;color:#c084fc;">🏥 Medical</div>
            {''.join(f'<div style="font-size:13px;color:#94a3b8;margin-bottom:6px;">• {d}</div>' for d in ins['medical'])}
            <div style="margin-top:10px;font-size:12px;font-family:'DM Mono',monospace;
                        color:#6366f1;">Follow-up: {ins['follow_up']}</div>
        </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  DOCTOR DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────
def doctor_dashboard():
    render_header("Doctor Dashboard")
    render_sidebar()
    df = db_fetch_all()

    st.markdown('<p class="sec-head">🔍 Patient Lookup</p>', unsafe_allow_html=True)
    c1, c2 = st.columns([3, 1])
    pid_in = c1.text_input("Patient ID", placeholder="e.g. P001", label_visibility="collapsed")
    fetch  = c2.button("🔍 Fetch Record", use_container_width=True)

    if fetch and pid_in.strip():
        pid_col = next((c for c in df.columns if c.lower()=="patient_id"), None)
        if pid_col:
            rows = df[df[pid_col].astype(str).str.strip() == pid_in.strip()]
            if not rows.empty:
                st.session_state.doc_record = rows.iloc[0].to_dict()
                st.session_state.doc_pid    = pid_in.strip()
            else:
                st.error(f"No patient found: `{pid_in.strip()}`")

    rec = st.session_state.get("doc_record")
    pid = st.session_state.get("doc_pid")

    if rec:
        prob, _ = predict_prob(rec)
        sev     = severity_score(rec)
        tier, color, icon = risk_tier(prob)
        sev_cat, sev_color, sev_icon = sev_category(sev)
        name = rec.get("name", rec.get("Name", pid))
        insights = build_ai_insights(rec, prob, sev)

        # Hero
        st.markdown(f"""
        <div class="card" style="border-left:3px solid {color};">
            <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:16px;">
                <div>
                    <div style="font-family:'Syne',sans-serif;font-size:24px;font-weight:800;">{name}</div>
                    <div style="font-size:12px;color:#475569;margin-top:5px;font-family:'DM Mono',monospace;">
                        ID: {pid} &nbsp;·&nbsp; Age: {fmt(rec.get('age',''),0)} &nbsp;·&nbsp; {rec.get('diagnosis','—')}
                    </div>
                </div>
                <div style="display:flex;gap:12px;">
                    <span class="badge" style="background:rgba(0,0,0,0.2);border:1px solid {color};color:{color};font-size:13px;">
                        {icon} ML {prob*100:.1f}%
                    </span>
                    <span class="badge" style="background:rgba(0,0,0,0.2);border:1px solid {sev_color};color:{sev_color};font-size:13px;">
                        {sev_icon} Sev {sev:.0f}/100
                    </span>
                </div>
            </div>
        </div>""", unsafe_allow_html=True)

        # Vitals
        v_keys = [
            ("fasting_blood_sugar","FastingBloodSugar","Glucose"),
            ("hba1c","HbA1c","HbA1c %"),("bmi","BMI","BMI"),
            ("cholesterol_total","CholesterolTotal","Cholesterol"),
            ("systolic_bp","SystolicBP","Systolic BP"),("diastolic_bp","DiastolicBP","Diastolic BP"),
        ]
        vc = st.columns(len(v_keys))
        for i,(k1,k2,lbl) in enumerate(v_keys):
            vc[i].metric(lbl, fmt(rec.get(k1, rec.get(k2,""))))

        st.markdown("<br>", unsafe_allow_html=True)
        cl, cr = st.columns([3, 2])
        with cl:
            st.plotly_chart(proj_chart(simulate_projection(rec)), use_container_width=True)
            st.markdown('<p class="sec-head">Full Record</p>', unsafe_allow_html=True)
            clean = {k: fmt(v) for k,v in rec.items() if not k.startswith("_")}
            st.dataframe(pd.DataFrame(clean.items(), columns=["Field","Value"]),
                         use_container_width=True, hide_index=True, height=320)
        with cr:
            st.plotly_chart(gauge_chart(prob), use_container_width=True)
            st.plotly_chart(vitals_radar(rec), use_container_width=True)
            try:
                qp = generate_secure_qr(pid)
                st.image(qp, width=130, caption="Secure JWT QR")
                st.markdown(f"""
                <div class="jwt-pill">🔐 HS256 · Exp: {JWT_EXPIRY_DAYS}d<br>
                patient_id + role + iat + exp</div>""", unsafe_allow_html=True)
            except Exception:
                pass

        st.markdown('<p class="sec-head">🤖 AI Assessment</p>', unsafe_allow_html=True)
        _render_ai_insights(insights)

        st.markdown('<p class="sec-head">📝 Doctor Remarks</p>', unsafe_allow_html=True)
        note = st.text_area("Remarks", value=rec.get("doctor_remarks",""), height=100, label_visibility="collapsed")
        if st.button("💾 Save Remarks"):
            updated = dict(rec); updated["doctor_remarks"] = note
            ok = db_upsert({k:updated[k] for k in updated if not k.startswith("_")})
            if ok:
                st.success("✅ Saved to database!")
                st.session_state.doc_record["doctor_remarks"] = note
                st.cache_data.clear()
            else:
                st.warning("⚠️ Supabase not configured — saved in session only.")
                st.session_state.doc_record["doctor_remarks"] = note
    else:
        st.markdown("""
        <div class="card" style="text-align:center;padding:60px;color:#334155;">
            <div style="font-size:52px;margin-bottom:18px;">🔍</div>
            <div style="font-size:16px;font-family:'Space Grotesk',sans-serif;">
                Enter a Patient ID above to view their full record
            </div>
        </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  ADMIN DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────
def admin_dashboard():
    render_header("Admin Dashboard")
    render_sidebar()

    raw_df = db_fetch_all()
    df     = enrich_df(raw_df)

    # ── FILTERS (sidebar) ────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### 🔎 Filters")
        age_col = next((c for c in df.columns if c.lower()=="age"), None)
        if age_col:
            ages = pd.to_numeric(df[age_col], errors="coerce").dropna()
            a_min, a_max = int(ages.min()), int(ages.max())
            age_range = st.slider("Age Range", a_min, a_max, (a_min, a_max))
        else:
            age_range = (0, 120)

        gender_col = next((c for c in df.columns if c.lower()=="gender"), None)
        gender_filter = st.multiselect("Gender", ["Male","Female","All"], default=["All"])

        diag_col = next((c for c in df.columns if c.lower()=="diagnosis"), None)
        diag_opts = ["All"] + sorted(df[diag_col].dropna().unique().tolist()) if diag_col else ["All"]
        diag_filter = st.multiselect("Diagnosis", diag_opts, default=["All"])

        risk_filter = st.multiselect("Risk Level", ["High","Moderate","Low"], default=["High","Moderate","Low"])

        st.markdown("---")

    # ── Apply filters ──
    fdf = df.copy()
    if age_col:
        fdf[age_col] = pd.to_numeric(fdf[age_col], errors="coerce")
        fdf = fdf[(fdf[age_col] >= age_range[0]) & (fdf[age_col] <= age_range[1])]
    if gender_col and "All" not in gender_filter:
        gender_map = {"Male":1, "Female":0}
        allowed = [gender_map.get(g,g) for g in gender_filter]
        fdf = fdf[fdf[gender_col].isin(allowed)]
    if diag_col and "All" not in diag_filter:
        fdf = fdf[fdf[diag_col].isin(diag_filter)]
    if risk_filter:
        tier_map = {"High":"High","Moderate":"Moderate","Low":"Low"}
        allowed_tiers = [tier_map[r] for r in risk_filter if r in tier_map]
        fdf_prob = np.array(fdf["ml_risk_pct"]) / 100
        tier_labels = [risk_tier(p)[0] for p in fdf_prob]
        fdf = fdf[[t in allowed_tiers for t in tier_labels]]

    # ── OVERVIEW METRICS ─────────────────────────────────────────────────────
    total    = len(fdf)
    hba_col  = next((c for c in fdf.columns if c.lower()=="hba1c"), None)
    bmi_col  = next((c for c in fdf.columns if c.lower()=="bmi"), None)
    hyp_col  = next((c for c in fdf.columns if c.lower()=="hypertension"), None)

    avg_hba  = pd.to_numeric(fdf[hba_col], errors="coerce").mean() if hba_col else float("nan")
    avg_bmi  = pd.to_numeric(fdf[bmi_col], errors="coerce").mean() if bmi_col else float("nan")
    hyp_rate = pd.to_numeric(fdf[hyp_col], errors="coerce").mean()*100 if hyp_col else float("nan")
    high_risk= int((fdf["ml_risk_pct"] >= 65).sum())
    crit_pct = round(high_risk/total*100, 1) if total else 0

    diab_count = 0
    prediab_count = 0
    normal_count  = 0
    if diag_col:
        diab_count    = int((fdf[diag_col].str.lower().str.contains("diabet",na=False) & ~fdf[diag_col].str.lower().str.contains("pre",na=False)).sum())
        prediab_count = int(fdf[diag_col].str.lower().str.contains("pre",na=False).sum())
        normal_count  = int(fdf[diag_col].str.lower().str.contains("normal",na=False).sum())

    # ── Alerts ──
    critical_pts = fdf[fdf["severity_score"] >= 75].sort_values("severity_score", ascending=False).head(3)
    if len(critical_pts):
        st.markdown('<p class="sec-head">🚨 Critical Alerts</p>', unsafe_allow_html=True)
        for _, row in critical_pts.iterrows():
            n  = row.get("name", row.get("patient_id","—"))
            sc = row.get("severity_score", 0)
            rl = row.get("ml_risk_pct", 0)
            hb = fmt(row.get(hba_col or "hba1c",""), 1)
            st.markdown(f"""
            <div class="alert-critical">
                <div class="alert-title">⚠️ {n} &nbsp;—&nbsp; Severity {sc}/100 · ML Risk {rl}%</div>
                <div class="alert-body">HbA1c: {hb}% &nbsp;·&nbsp; Immediate clinical review recommended</div>
            </div>""", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

    # KPI row
    kpi_data = [
        ("Total Patients",    total,                       "👥", "#6366f1"),
        ("High Risk",         f"{high_risk} ({crit_pct}%)", "⚠️","#f43f5e"),
        ("Diabetic",          diab_count,                  "🩸","#f59e0b"),
        ("Pre-Diabetic",      prediab_count,               "🟡","#8b5cf6"),
        ("Avg HbA1c",         f"{avg_hba:.2f}%" if not np.isnan(avg_hba) else "—", "📊","#06b6d4"),
        ("Avg BMI",           f"{avg_bmi:.1f}"  if not np.isnan(avg_bmi) else "—", "⚖️","#10b981"),
        ("Hypertension Rate", f"{hyp_rate:.1f}%" if not np.isnan(hyp_rate) else "—","💉","#ec4899"),
    ]
    kpi_cols = st.columns(len(kpi_data))
    for i, (lbl, val, em, clr) in enumerate(kpi_data):
        kpi_cols[i].markdown(f"""
        <div class="kpi" style="--kpi-color:{clr}">
            <div class="kpi-icon">{em}</div>
            <div class="kpi-label">{lbl}</div>
            <div class="kpi-value" style="font-size:{'22px' if len(str(val))>6 else '30px'}">{val}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── TABS ──────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Analytics",
        "📋 Patient Table",
        "➕ Add Patient",
        "🔍 Advanced Insights",
    ])

    # ── TAB 1: ANALYTICS ─────────────────────────────────────────────────────
    with tab1:
        _render_analytics(fdf, hba_col, bmi_col, diag_col, gender_col, hyp_col, age_col)

    # ── TAB 2: PATIENT TABLE ─────────────────────────────────────────────────
    with tab2:
        _render_smart_table(fdf, hba_col, bmi_col, diag_col)

    # ── TAB 3: ADD PATIENT ───────────────────────────────────────────────────
    with tab3:
        _render_add_patient(df)

    # ── TAB 4: ADVANCED INSIGHTS ─────────────────────────────────────────────
    with tab4:
        _render_advanced_insights(fdf, hba_col, age_col)


def _render_analytics(fdf, hba_col, bmi_col, diag_col, gender_col, hyp_col, age_col):
    """Full analytics panel with 6+ charts."""

    st.markdown('<p class="sec-head">Visual Analytics</p>', unsafe_allow_html=True)

    # Row 1: Diagnosis distribution + Risk distribution
    c1, c2 = st.columns(2)
    with c1:
        if diag_col:
            vc = fdf[diag_col].value_counts()
            fig = px.pie(
                values=vc.values, names=vc.index,
                title="Diagnosis Distribution",
                color_discrete_map={"Diabetic":"#f43f5e","Pre-Diabetic":"#f59e0b","Normal":"#10b981"},
                hole=0.52,
            )
            apply_chart_style(fig, height=280)
            st.plotly_chart(fig, use_container_width=True)

    with c2:
        if len(fdf):
            rc = pd.Series(["High" if p>=65 else ("Moderate" if p>=35 else "Low")
                            for p in fdf["ml_risk_pct"]]).value_counts()
            fig2 = px.pie(
                values=rc.values, names=rc.index,
                title="ML Risk Distribution",
                color_discrete_map={"High":"#f43f5e","Moderate":"#f59e0b","Low":"#10b981"},
                hole=0.52,
            )
            apply_chart_style(fig2, height=280)
            st.plotly_chart(fig2, use_container_width=True)

    # Row 2: HbA1c histogram + BMI histogram
    c3, c4 = st.columns(2)
    with c3:
        if hba_col:
            hba_vals = pd.to_numeric(fdf[hba_col], errors="coerce").dropna()
            fig3 = px.histogram(hba_vals, nbins=22, title="HbA1c Distribution",
                                color_discrete_sequence=["#6366f1"])
            fig3.add_vline(x=5.7, line=dict(color="#f59e0b",dash="dot"), annotation_text="Pre-DM",annotation_font_color="#f59e0b")
            fig3.add_vline(x=6.5, line=dict(color="#f43f5e",dash="dot"), annotation_text="DM",    annotation_font_color="#f43f5e")
            apply_chart_style(fig3, height=260)
            st.plotly_chart(fig3, use_container_width=True)

    with c4:
        if bmi_col:
            bmi_vals = pd.to_numeric(fdf[bmi_col], errors="coerce").dropna()
            fig4 = px.histogram(bmi_vals, nbins=20, title="BMI Distribution",
                                color_discrete_sequence=["#06b6d4"])
            fig4.add_vline(x=25, line=dict(color="#f59e0b",dash="dot"), annotation_text="Overweight",annotation_font_color="#f59e0b")
            fig4.add_vline(x=30, line=dict(color="#f43f5e",dash="dot"), annotation_text="Obese",    annotation_font_color="#f43f5e")
            apply_chart_style(fig4, height=260)
            st.plotly_chart(fig4, use_container_width=True)

    # Row 3: Age vs Diabetes bar + Gender breakdown
    c5, c6 = st.columns(2)
    with c5:
        if age_col and diag_col:
            tmp = fdf.copy()
            tmp[age_col] = pd.to_numeric(tmp[age_col], errors="coerce")
            tmp = tmp.dropna(subset=[age_col])
            tmp["age_group"] = pd.cut(tmp[age_col], bins=[0,30,40,50,60,70,120],
                                       labels=["<30","30-40","40-50","50-60","60-70","70+"])
            grp = tmp.groupby(["age_group", diag_col], observed=True).size().reset_index(name="count")
            fig5 = px.bar(grp, x="age_group", y="count", color=diag_col,
                          title="Age Group vs Diagnosis",
                          color_discrete_map={"Diabetic":"#f43f5e","Pre-Diabetic":"#f59e0b","Normal":"#10b981"},
                          barmode="group")
            apply_chart_style(fig5, height=280)
            st.plotly_chart(fig5, use_container_width=True)

    with c6:
        if gender_col and diag_col:
            tmp2 = fdf.copy()
            tmp2["gender_lbl"] = pd.to_numeric(tmp2[gender_col], errors="coerce").map({1:"Male",0:"Female"})
            grp2 = tmp2.groupby(["gender_lbl", diag_col]).size().reset_index(name="count")
            fig6 = px.bar(grp2, x="gender_lbl", y="count", color=diag_col,
                          title="Gender vs Diagnosis",
                          color_discrete_map={"Diabetic":"#f43f5e","Pre-Diabetic":"#f59e0b","Normal":"#10b981"},
                          barmode="stack")
            apply_chart_style(fig6, height=280)
            st.plotly_chart(fig6, use_container_width=True)

    # Row 4: Severity score distribution + HbA1c vs BMI scatter
    c7, c8 = st.columns(2)
    with c7:
        fig7 = px.histogram(fdf["severity_score"], nbins=20,
                             title="Severity Score Distribution",
                             color_discrete_sequence=["#8b5cf6"])
        fig7.add_vline(x=31, line=dict(color="#f59e0b",dash="dot"), annotation_text="Medium",annotation_font_color="#f59e0b")
        fig7.add_vline(x=61, line=dict(color="#f43f5e",dash="dot"), annotation_text="High",  annotation_font_color="#f43f5e")
        apply_chart_style(fig7, height=260)
        st.plotly_chart(fig7, use_container_width=True)

    with c8:
        if hba_col and bmi_col:
            tmp3 = fdf.copy()
            tmp3[hba_col] = pd.to_numeric(tmp3[hba_col], errors="coerce")
            tmp3[bmi_col] = pd.to_numeric(tmp3[bmi_col], errors="coerce")
            tmp3 = tmp3.dropna(subset=[hba_col, bmi_col])
            color_col = diag_col if diag_col else "severity_label"
            fig8 = px.scatter(tmp3, x=bmi_col, y=hba_col, color=color_col,
                              title="HbA1c vs BMI",
                              color_discrete_map={"Diabetic":"#f43f5e","Pre-Diabetic":"#f59e0b","Normal":"#10b981",
                                                  "High":"#f43f5e","Medium":"#f59e0b","Low":"#10b981"},
                              opacity=0.75)
            fig8.add_hline(y=5.7, line=dict(color="#f59e0b",dash="dot",width=1))
            fig8.add_hline(y=6.5, line=dict(color="#f43f5e",dash="dot",width=1))
            apply_chart_style(fig8, height=260)
            st.plotly_chart(fig8, use_container_width=True)

    # Row 5: Correlation heatmap
    st.markdown('<p class="sec-head" style="margin-top:8px;">Feature Correlation Heatmap</p>', unsafe_allow_html=True)
    corr_cols = [c for c in [hba_col, bmi_col,
                              next((x for x in fdf.columns if x.lower()=="fasting_blood_sugar"), None),
                              next((x for x in fdf.columns if x.lower()=="systolic_bp"), None),
                              next((x for x in fdf.columns if x.lower()=="cholesterol_total"), None),
                              next((x for x in fdf.columns if x.lower()=="age"), None),
                              next((x for x in fdf.columns if x.lower()=="smoking"), None),
                              next((x for x in fdf.columns if x.lower()=="physical_activity"), None),
                              "severity_score","ml_risk_pct",
                              ] if c is not None]
    corr_cols = list(dict.fromkeys(corr_cols))  # dedupe
    if len(corr_cols) >= 3:
        corr_df = fdf[corr_cols].apply(pd.to_numeric, errors="coerce").corr()
        fig9 = go.Figure(go.Heatmap(
            z=corr_df.values, x=corr_df.columns, y=corr_df.columns,
            colorscale=[[0,"#1e1b4b"],[0.5,"#6366f1"],[1,"#f43f5e"]],
            zmin=-1, zmax=1,
            text=np.round(corr_df.values,2), texttemplate="%{text}",
            colorbar=dict(tickfont=dict(color="#64748b")),
        ))
        fig9.update_layout(**CHART_STYLE, height=380, title=dict(text="Pearson Correlation", font=dict(color="#94a3b8",size=13)))
        st.plotly_chart(fig9, use_container_width=True)

    # Export button
    csv_data = fdf.drop(columns=["severity_score","severity_label","ml_risk_pct"], errors="ignore").to_csv(index=False).encode()
    st.download_button("⬇️ Export Filtered CSV", data=csv_data,
                       file_name="gluco_lens_filtered.csv", mime="text/csv")


def _render_smart_table(fdf, hba_col, bmi_col, diag_col):
    """Color-coded, searchable, sortable patient table."""
    st.markdown('<p class="sec-head">Smart Patient Table — Sorted by Severity</p>', unsafe_allow_html=True)

    search = st.text_input("🔍 Search by name or patient ID", placeholder="Type to filter…", key="table_search")

    display_cols = ["patient_id","name","age","gender","diagnosis",
                    "severity_score","severity_label","ml_risk_pct",
                    hba_col or "hba1c", bmi_col or "bmi",
                    "systolic_bp","fasting_blood_sugar"]
    display_cols = [c for c in display_cols if c in fdf.columns]

    tdf = fdf[display_cols].copy()
    tdf = tdf.sort_values("severity_score", ascending=False)

    if search.strip():
        mask = (
            tdf.get("name","").astype(str).str.lower().str.contains(search.lower(), na=False) |
            tdf.get("patient_id","").astype(str).str.lower().str.contains(search.lower(), na=False)
        )
        tdf = tdf[mask]

    # Rename for display
    tdf = tdf.rename(columns={
        "patient_id":"ID","name":"Name","age":"Age","gender":"Sex",
        "diagnosis":"Diagnosis","severity_score":"Sev Score",
        "severity_label":"Sev Level","ml_risk_pct":"ML Risk %",
        hba_col or "hba1c":"HbA1c","bmi_col" or "bmi":"BMI",
        "systolic_bp":"SBP","fasting_blood_sugar":"FBS",
    })
    if "Sex" in tdf.columns:
        tdf["Sex"] = pd.to_numeric(tdf["Sex"], errors="coerce").map({1:"M",0:"F"})

    st.dataframe(
        tdf.reset_index(drop=True),
        use_container_width=True,
        height=480,
        hide_index=True,
    )
    st.caption(f"Showing {len(tdf)} of {len(fdf)} patients | 🔴 Sev≥61 · 🟡 31–60 · 🟢 <31")

    # Expandable AI detail for selected patient
    st.markdown("<br>", unsafe_allow_html=True)
    sel_pid = st.text_input("Enter Patient ID for AI Detail →", placeholder="e.g. P001", key="detail_pid")
    if sel_pid.strip():
        pid_col_name = next((c for c in fdf.columns if c.lower()=="patient_id"), None)
        if pid_col_name:
            rows = fdf[fdf[pid_col_name].astype(str).str.strip() == sel_pid.strip()]
            if not rows.empty:
                rec = rows.iloc[0].to_dict()
                prob, _ = predict_prob(rec)
                sev = severity_score(rec)
                ins = build_ai_insights(rec, prob, sev)
                _render_ai_insights(ins)
            else:
                st.warning("Patient not found.")


def _render_add_patient(df):
    """New patient form with QR generation."""
    st.markdown('<p class="sec-head">➕ Register New Patient</p>', unsafe_allow_html=True)
    total = len(df)
    with st.form("add_pt_v5"):
        c1, c2, c3 = st.columns(3)
        pid  = c1.text_input("Patient ID*", value=f"P{total+1:03d}")
        name = c2.text_input("Full Name*")
        diag = c3.text_input("Diagnosis (Normal/Pre-Diabetic/Diabetic)")
        c1, c2, c3, c4 = st.columns(4)
        age_v  = c1.number_input("Age",   18, 100, 40)
        gen_v  = c2.selectbox("Gender", ["Male","Female"])
        bmi_v  = c3.number_input("BMI",   10.0, 60.0, 25.0)
        hba_v  = c4.number_input("HbA1c (%)", 3.0, 15.0, 5.5)
        c1, c2, c3, c4 = st.columns(4)
        fbs_v  = c1.number_input("Fasting Blood Sugar", 50, 400, 100)
        chol_v = c2.number_input("Total Cholesterol",   50, 500, 180)
        sbp_v  = c3.number_input("Systolic BP",         80, 220, 120)
        dbp_v  = c4.number_input("Diastolic BP",        40, 130, 80)
        c1, c2, c3 = st.columns(3)
        hyp_v  = c1.selectbox("Hypertension", ["No","Yes"])
        fam_v  = c2.selectbox("Family History DM", ["No","Yes"])
        smk_v  = c3.selectbox("Smoking", ["No","Yes"])
        remarks_v = st.text_area("Doctor Remarks", height=80)

        if st.form_submit_button("✅ Save Patient & Generate Secure QR", use_container_width=True):
            if not name.strip():
                st.error("Name is required.")
            else:
                row = {
                    "patient_id": pid.strip(), "name": name.strip(),
                    "age": age_v, "gender": 1 if gen_v=="Male" else 0,
                    "bmi": bmi_v, "hba1c": hba_v, "fasting_blood_sugar": fbs_v,
                    "cholesterol_total": chol_v, "systolic_bp": sbp_v, "diastolic_bp": dbp_v,
                    "hypertension": 1 if hyp_v=="Yes" else 0,
                    "family_history_diabetes": 1 if fam_v=="Yes" else 0,
                    "smoking": 1 if smk_v=="Yes" else 0,
                    "diagnosis": diag, "doctor_remarks": remarks_v,
                    "created_at": datetime.utcnow().isoformat()
                }
                # Preview severity for new patient
                sev = severity_score(row)
                prob, _ = predict_prob(row)
                tier, color, icon = risk_tier(prob)
                st.markdown(f"""
                <div class="card" style="border-left:3px solid {color};">
                    <b style="color:{color}">{icon} Predicted Risk: {tier} ({prob*100:.1f}%) · Severity Score: {sev}/100</b>
                </div>""", unsafe_allow_html=True)

                ok = db_upsert(row)
                try:
                    qp = generate_secure_qr(pid.strip())
                    col_a, col_b = st.columns([1, 2])
                    with col_a:
                        st.image(qp, width=180, caption=f"JWT QR · {pid}")
                    with col_b:
                        st.markdown(f"""
                        <div class="jwt-pill">
                            🔐 Secure JWT Token<br>
                            Patient: {pid.strip()}<br>
                            Algorithm: HS256<br>
                            Expires: {JWT_EXPIRY_DAYS} days<br><br>
                            Patient scans → auto-login (no password)
                        </div>""", unsafe_allow_html=True)
                except Exception as e:
                    st.warning(f"QR gen error: {e}")

                if ok:
                    st.success(f"✅ Patient {pid} saved!")
                    st.cache_data.clear()
                else:
                    st.warning("⚠️ Supabase not connected. Add SUPABASE_URL & SUPABASE_KEY to secrets.")


def _render_advanced_insights(fdf, hba_col, age_col):
    """Advanced analytics: lifestyle, smoking, family history."""
    st.markdown('<p class="sec-head">🔬 Advanced Insights</p>', unsafe_allow_html=True)

    smoke_col = next((c for c in fdf.columns if c.lower()=="smoking"), None)
    fam_col   = next((c for c in fdf.columns if c.lower()=="family_history_diabetes"), None)
    pa_col    = next((c for c in fdf.columns if c.lower()=="physical_activity"), None)
    diet_col  = next((c for c in fdf.columns if c.lower()=="diet_quality"), None)

    c1, c2 = st.columns(2)

    # Smoking impact on HbA1c
    with c1:
        if smoke_col and hba_col:
            tmp = fdf.copy()
            tmp[hba_col]    = pd.to_numeric(tmp[hba_col],    errors="coerce")
            tmp[smoke_col]  = pd.to_numeric(tmp[smoke_col],  errors="coerce")
            tmp["Smoker"]   = tmp[smoke_col].map({1:"Smoker",0:"Non-Smoker"})
            fig = px.box(tmp, x="Smoker", y=hba_col, color="Smoker",
                         title="Smoking Impact on HbA1c",
                         color_discrete_map={"Smoker":"#f43f5e","Non-Smoker":"#10b981"})
            apply_chart_style(fig, height=280)
            st.plotly_chart(fig, use_container_width=True)

    # Family history impact
    with c2:
        if fam_col and hba_col:
            tmp2 = fdf.copy()
            tmp2[hba_col]  = pd.to_numeric(tmp2[hba_col], errors="coerce")
            tmp2[fam_col]  = pd.to_numeric(tmp2[fam_col], errors="coerce")
            tmp2["Family History"] = tmp2[fam_col].map({1:"Positive",0:"Negative"})
            fig2 = px.box(tmp2, x="Family History", y=hba_col, color="Family History",
                          title="Family History Impact on HbA1c",
                          color_discrete_map={"Positive":"#f59e0b","Negative":"#06b6d4"})
            apply_chart_style(fig2, height=280)
            st.plotly_chart(fig2, use_container_width=True)

    c3, c4 = st.columns(2)

    # Lifestyle vs ML risk scatter
    with c3:
        if pa_col:
            tmp3 = fdf.copy()
            tmp3[pa_col] = pd.to_numeric(tmp3[pa_col], errors="coerce")
            fig3 = px.scatter(tmp3, x=pa_col, y="ml_risk_pct", color="severity_label",
                              title="Physical Activity vs ML Risk",
                              color_discrete_map={"High":"#f43f5e","Medium":"#f59e0b","Low":"#10b981"},
                              trendline="ols",
                              trendline_color_override="#6366f1")
            apply_chart_style(fig3, height=280)
            st.plotly_chart(fig3, use_container_width=True)

    # Diet quality vs severity
    with c4:
        if diet_col:
            tmp4 = fdf.copy()
            tmp4[diet_col] = pd.to_numeric(tmp4[diet_col], errors="coerce")
            tmp4["diet_bin"] = pd.cut(tmp4[diet_col], bins=[0,2,3.5,5],
                                       labels=["Poor (<2)","Fair (2–3.5)","Good (>3.5)"])
            grp = tmp4.groupby("diet_bin", observed=True)["severity_score"].mean().reset_index()
            fig4 = px.bar(grp, x="diet_bin", y="severity_score",
                          title="Diet Quality vs Avg Severity",
                          color="severity_score",
                          color_continuous_scale=[[0,"#10b981"],[0.5,"#f59e0b"],[1,"#f43f5e"]])
            apply_chart_style(fig4, height=280)
            st.plotly_chart(fig4, use_container_width=True)

    # Top 10 highest severity
    st.markdown('<p class="sec-head" style="margin-top:12px;">🔴 Top 10 Highest Severity Patients</p>', unsafe_allow_html=True)
    top10 = fdf.sort_values("severity_score", ascending=False).head(10)
    disp_cols = [c for c in ["patient_id","name","severity_score","severity_label","ml_risk_pct",
                              hba_col, age_col] if c and c in top10.columns]
    st.dataframe(top10[disp_cols].reset_index(drop=True), use_container_width=True, hide_index=True)

    # Export enriched data
    enriched_csv = fdf.to_csv(index=False).encode()
    st.download_button("⬇️ Export Enriched Dataset (with Severity + ML Risk)",
                       data=enriched_csv, file_name="gluco_lens_enriched.csv", mime="text/csv")

# ─────────────────────────────────────────────────────────────────────────────
#  MAIN ROUTER
# ─────────────────────────────────────────────────────────────────────────────
if not st.session_state.logged_in:
    show_login()
else:
    role = st.session_state.role
    if   role == "Admin":   admin_dashboard()
    elif role == "Doctor":  doctor_dashboard()
    elif role == "Patient": patient_portal()
    else:
        st.error("Unknown role. Please sign out.")