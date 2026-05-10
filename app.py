import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import joblib
import qrcode
import os
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

# ── Paths ─────────────────────────────────────────────────────────────────────
_BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH    = os.path.join(_BASE_DIR, "models", "diabetes_xgb.pkl")
FEATURE_PATH  = os.path.join(_BASE_DIR, "models", "feature_columns.pkl")
QR_FOLDER     = "/tmp/qrcodes"
TABLE_NAME    = "patients"
os.makedirs(QR_FOLDER, exist_ok=True)

# ── Auth / secrets ────────────────────────────────────────────────────────────
JWT_SECRET      = "gluco-lens-jwt-secret-2024-change-in-production"
JWT_EXPIRY_DAYS = 365
SUPABASE_URL    = ""
SUPABASE_KEY    = ""
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

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GLUCO-LENS", page_icon="🩺",
    layout="wide", initial_sidebar_state="expanded",
)

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
[data-testid="stSidebar"] {
    background: rgba(10,12,28,0.95) !important;
    border-right: 1px solid rgba(99,102,241,0.2) !important;
    backdrop-filter: blur(20px) !important;
}
[data-testid="stSidebar"] * { color: #c7d2fe !important; font-family:'Outfit',sans-serif !important; }
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stDecoration"] { display: none; }
.glass {
    background: rgba(255,255,255,0.035);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 20px; padding: 24px;
    backdrop-filter: blur(12px); margin-bottom: 20px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.06);
}
.topbar {
    background: linear-gradient(135deg, rgba(99,102,241,0.25), rgba(14,165,233,0.2));
    border: 1px solid rgba(99,102,241,0.3); border-radius: 20px;
    padding: 20px 32px; margin-bottom: 28px;
    display: flex; align-items: center; gap: 16px;
    box-shadow: 0 0 60px rgba(99,102,241,0.15), inset 0 1px 0 rgba(255,255,255,0.1);
}
.topbar-title {
    font-size: 28px; font-weight: 800; letter-spacing: 3px;
    background: linear-gradient(135deg, #a5b4fc, #38bdf8, #a5b4fc);
    background-size: 200% auto;
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    animation: shimmer 3s linear infinite;
}
.topbar-sub { font-size: 13px; color: #94a3b8; letter-spacing: 1px; margin-top: 2px; }
@keyframes shimmer { 0%{background-position:0%} 100%{background-position:200%} }
.success-badge {
    background: rgba(52,211,153,0.12); border: 1px solid rgba(52,211,153,0.3);
    border-radius: 14px; padding: 16px 24px; color: #34d399;
    font-weight: 600; font-size: 16px; text-align: center; margin-bottom: 16px;
    animation: fadeIn 0.4s ease;
}
@keyframes fadeIn { from{opacity:0;transform:translateY(-8px)} to{opacity:1;transform:none} }
.jwt-info {
    background: rgba(99,102,241,0.06); border: 1px solid rgba(99,102,241,0.15);
    border-left: 3px solid #6366f1; border-radius: 10px; padding: 10px 14px;
    font-family: 'JetBrains Mono', monospace; font-size: 11px; color: #64748b;
    margin-top: 8px;
}
.stTextInput input, .stNumberInput input, .stTextArea textarea {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(99,102,241,0.25) !important;
    border-radius: 12px !important; color: #e2e8f5 !important;
    font-family: 'Outfit', sans-serif !important; font-size: 14px !important;
}
.stTextInput label, .stNumberInput label, .stTextArea label,
.stSelectbox label, .stSlider label {
    color: #94a3b8 !important; font-size: 13px !important; font-weight: 500 !important;
}
.stButton > button {
    background: linear-gradient(135deg, #6366f1, #0ea5e9) !important;
    color: white !important; border: none !important; border-radius: 12px !important;
    padding: 12px 28px !important; font-family: 'Outfit', sans-serif !important;
    font-weight: 600 !important; font-size: 14px !important;
    box-shadow: 0 4px 20px rgba(99,102,241,0.35) !important;
    transition: opacity 0.2s, transform 0.2s !important;
}
.stButton > button:hover { opacity: 0.88 !important; transform: translateY(-1px) !important; }
.stDownloadButton > button {
    background: rgba(99,102,241,0.15) !important;
    border: 1px solid rgba(99,102,241,0.35) !important;
    color: #a5b4fc !important; border-radius: 12px !important;
}
.stTabs [data-baseweb="tab-list"] {
    background: rgba(255,255,255,0.03) !important; border-radius: 12px !important;
    padding: 4px !important; border: 1px solid rgba(255,255,255,0.06) !important;
}
.stTabs [data-baseweb="tab"] { border-radius: 8px !important; color: #64748b !important; }
.stTabs [aria-selected="true"] { background: rgba(99,102,241,0.2) !important; color: #a5b4fc !important; }
[data-testid="metric-container"] {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 14px !important; padding: 16px !important;
}
[data-testid="metric-container"] label { color: #64748b !important; font-size: 12px !important; }
[data-testid="metric-container"] [data-testid="stMetricValue"] { color: #e2e8f5 !important; font-weight: 700 !important; }
.sec-head {
    font-size: 18px; font-weight: 700; color: #c7d2fe; letter-spacing: 0.5px;
    margin-bottom: 16px; padding-bottom: 10px;
    border-bottom: 1px solid rgba(255,255,255,0.07);
}
.ai-box {
    background: linear-gradient(135deg, rgba(99,102,241,0.08), rgba(14,165,233,0.06));
    border: 1px solid rgba(99,102,241,0.2); border-left: 3px solid #6366f1;
    border-radius: 14px; padding: 18px 20px; font-size: 14px;
    line-height: 1.7; color: #cbd5e1;
}
.sb-role {
    background: rgba(99,102,241,0.12); border: 1px solid rgba(99,102,241,0.2);
    border-radius: 10px; padding: 12px 16px; margin-bottom: 12px;
}
.sb-role .role-name { font-size: 18px; font-weight: 700; color: #a5b4fc; }
.sb-role .role-user { font-size: 12px; color: #64748b; margin-top: 2px; }
.login-card {
    background: rgba(255,255,255,0.04); border: 1px solid rgba(99,102,241,0.25);
    border-radius: 24px; padding: 48px 44px; width: 100%; max-width: 440px;
    box-shadow: 0 24px 80px rgba(0,0,0,0.6), 0 0 80px rgba(99,102,241,0.08);
}
.login-logo {
    font-size: 42px; font-weight: 800; letter-spacing: 4px;
    background: linear-gradient(135deg, #a5b4fc 0%, #38bdf8 50%, #a5b4fc 100%);
    background-size: 200% auto; -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: shimmer 3s linear infinite; text-align: center; margin-bottom: 4px;
}
.login-sub {
    text-align: center; color: #475569; font-size: 13px;
    letter-spacing: 2px; text-transform: uppercase; margin-bottom: 32px;
}
.cred-badge {
    background: rgba(99,102,241,0.1); border: 1px solid rgba(99,102,241,0.2);
    border-radius: 10px; padding: 12px 16px;
    font-family: 'JetBrains Mono', monospace; font-size: 12px;
    color: #94a3b8; margin-top: 20px; line-height: 1.8;
}
hr { border-color: rgba(255,255,255,0.06) !important; }
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
_defaults = dict(
    logged_in=False, role=None, username=None,
    doc_record=None, doc_pid=None,
    pat_record=None, pat_pid=None,
)
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ══════════════════════════════════════════════════════════════════════════════
#  SUPABASE
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource
def get_supabase():
    if not SUPABASE_AVAILABLE or not SUPABASE_URL or not SUPABASE_KEY:
        return None
    try:
        return create_client(SUPABASE_URL, SUPABASE_KEY)
    except Exception:
        return None

def _demo_df() -> pd.DataFrame:
    return pd.DataFrame([
        {"patient_id": "P001", "name": "Demo Patient", "age": 45,
         "gender": "Male", "bmi": 28.5, "hba1c": 6.2,
         "fasting_blood_sugar": 118, "cholesterol_total": 195,
         "systolic_bp": 130, "diastolic_bp": 82,
         "hypertension": 1, "heart_disease": 0,
         "smoking_history": "former", "family_history_diabetes": 1,
         "diagnosis": "Pre-diabetic", "doctor_remarks": "Monitor closely."},
    ])

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


# ══════════════════════════════════════════════════════════════════════════════
#  JWT / SECURE QR
# ══════════════════════════════════════════════════════════════════════════════

def generate_secure_qr(patient_id: str) -> str:
    now = datetime.now(tz=timezone.utc)
    exp = now + timedelta(days=JWT_EXPIRY_DAYS)
    payload = {"patient_id": patient_id, "role": "patient",
               "iat": int(now.timestamp()), "exp": int(exp.timestamp())}
    if JWT_AVAILABLE:
        token = pyjwt.encode(payload, JWT_SECRET, algorithm="HS256")
    else:
        def b64url(data):
            if isinstance(data, str):
                data = data.encode()
            return base64.urlsafe_b64encode(data).rstrip(b"=").decode()
        header  = b64url(json.dumps({"alg": "HS256", "typ": "JWT"}, separators=(",", ":")))
        body    = b64url(json.dumps(payload, separators=(",", ":")))
        signing = f"{header}.{body}"
        sig     = hmac.new(JWT_SECRET.encode(), signing.encode(), hashlib.sha256).digest()
        token   = f"{signing}.{b64url(sig)}"
    path = os.path.join(QR_FOLDER, f"{patient_id}_jwt.png")
    qr = qrcode.QRCode(version=None,
                       error_correction=qrcode.constants.ERROR_CORRECT_M,
                       box_size=8, border=3)
    qr.add_data(token)
    qr.make(fit=True)
    qr.make_image(fill_color="#6366f1", back_color="white").save(path)
    return path

def verify_qr_token(token: str):
    if not token or not isinstance(token, str):
        return None
    try:
        if JWT_AVAILABLE:
            payload = pyjwt.decode(token, JWT_SECRET, algorithms=["HS256"])
            return payload if payload.get("role") == "patient" else None
        parts = token.split(".")
        if len(parts) != 3:
            return None
        header_b64, body_b64, sig_b64 = parts
        signing  = f"{header_b64}.{body_b64}"
        expected = hmac.new(JWT_SECRET.encode(), signing.encode(), hashlib.sha256).digest()
        pad      = 4 - len(sig_b64) % 4
        actual   = base64.urlsafe_b64decode(sig_b64 + "=" * (pad % 4))
        if not hmac.compare_digest(expected, actual):
            return None
        pad     = 4 - len(body_b64) % 4
        payload = json.loads(base64.urlsafe_b64decode(body_b64 + "=" * (pad % 4)))
        if payload.get("exp", 0) < time.time():
            return None
        return payload if payload.get("role") == "patient" else None
    except Exception:
        return None

def qr_from_image_file(pil_img):
    if PYZBAR_AVAILABLE:
        try:
            arr     = np.array(pil_img.convert("RGB"))
            results = pyzbar_lib.decode(arr)
            for r in results:
                if r.type == "QRCODE":
                    return r.data.decode("utf-8")
        except Exception:
            pass
    return None


# ══════════════════════════════════════════════════════════════════════════════
#  ML — XGBoost loader (no RandomForest, no StandardScaler, no LabelEncoder)
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="🧠 Loading XGBoost model…")
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"❌ Model file not found: `{MODEL_PATH}`\n\n"
                 "Run your training script to generate `models/diabetes_xgb.pkl`.")
        st.stop()
    if not os.path.exists(FEATURE_PATH):
        st.error(f"❌ Feature columns file not found: `{FEATURE_PATH}`\n\n"
                 "Run `create_feature_columns.py` to generate it.")
        st.stop()
    model        = joblib.load(MODEL_PATH)
    feature_cols = joblib.load(FEATURE_PATH)
    return model, list(feature_cols)

MODEL, FEATURE_COLUMNS = load_model()

# ── Dummy lookup tables (mirrors pd.get_dummies drop_first=True) ──────────────
# gender sorted: [Female, Male, Other] → Female dropped
# smoking sorted: [No Info, current, ever, former, never, not current] → No Info dropped

_GENDER_DUMMIES = {
    "Male":   {"gender_Male": 1, "gender_Other": 0},
    "Female": {"gender_Male": 0, "gender_Other": 0},
    "Other":  {"gender_Male": 0, "gender_Other": 1},
}

_SMOKING_DUMMIES = {
    "current":     {"smoking_history_current": 1, "smoking_history_ever": 0,
                    "smoking_history_former": 0,  "smoking_history_never": 0,
                    "smoking_history_not current": 0},
    "ever":        {"smoking_history_current": 0, "smoking_history_ever": 1,
                    "smoking_history_former": 0,  "smoking_history_never": 0,
                    "smoking_history_not current": 0},
    "former":      {"smoking_history_current": 0, "smoking_history_ever": 0,
                    "smoking_history_former": 1,  "smoking_history_never": 0,
                    "smoking_history_not current": 0},
    "never":       {"smoking_history_current": 0, "smoking_history_ever": 0,
                    "smoking_history_former": 0,  "smoking_history_never": 1,
                    "smoking_history_not current": 0},
    "not current": {"smoking_history_current": 0, "smoking_history_ever": 0,
                    "smoking_history_former": 0,  "smoking_history_never": 0,
                    "smoking_history_not current": 1},
    "No Info":     {"smoking_history_current": 0, "smoking_history_ever": 0,
                    "smoking_history_former": 0,  "smoking_history_never": 0,
                    "smoking_history_not current": 0},
}


# ══════════════════════════════════════════════════════════════════════════════
#  PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════

def _safe_float(x, default: float = 0.0) -> float:
    if x is None or x == "":
        return default
    if isinstance(x, float) and np.isnan(x):
        return default
    try:
        return float(x)
    except Exception:
        return default

def preprocess_input(patient: dict) -> pd.DataFrame:
    """Convert a raw patient dict into a model-ready single-row DataFrame."""

    def _get(keys, default=0.0):
        for k in keys:
            if k in patient and patient[k] is not None and patient[k] != "":
                try:
                    return float(patient[k])
                except (ValueError, TypeError):
                    pass
        return float(default)

    # Numeric features
    row = {
        "age":                 _get(["age", "Age"]),
        "hypertension":        _get(["hypertension", "Hypertension"]),
        "heart_disease":       _get(["heart_disease", "HeartDisease"]),
        "bmi":                 _get(["bmi", "BMI"]),
        "HbA1c_level":         _get(["hba1c", "HbA1c", "hba1c_level", "HbA1c_level"]),
        "blood_glucose_level": _get(["fasting_blood_sugar", "FastingBloodSugar",
                                     "blood_glucose_level"]),
    }

    # Gender → dummies
    raw_gender = patient.get("gender", patient.get("Gender", "Female"))
    if isinstance(raw_gender, (int, float)):
        gender_str = "Male" if int(raw_gender) == 1 else "Female"
    else:
        g = str(raw_gender).strip().lower()
        gender_str = "Male" if g in ("male", "m") else "Other" if g == "other" else "Female"
    row.update(_GENDER_DUMMIES.get(gender_str, _GENDER_DUMMIES["Female"]))

    # Smoking → dummies
    raw_smoking = patient.get("smoking_history",
                  patient.get("SmokingHistory", patient.get("smoking", "No Info")))
    if isinstance(raw_smoking, (int, float)):
        smoking_str = "current" if int(raw_smoking) == 1 else "never"
    else:
        s = str(raw_smoking).strip().lower()
        _map = {"yes": "current", "no": "never", "former": "former",
                "current": "current", "ever": "ever", "never": "never",
                "not current": "not current", "no info": "No Info", "": "No Info"}
        smoking_str = _map.get(s, "No Info")
    row.update(_SMOKING_DUMMIES.get(smoking_str, _SMOKING_DUMMIES["No Info"]))

    # Build DataFrame, pad missing cols, reorder to exact training order
    df = pd.DataFrame([row])
    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = 0
    return df[FEATURE_COLUMNS]


def predict_prob(patient: dict) -> tuple:
    try:
        df   = preprocess_input(patient)
        prob = float(MODEL.predict_proba(df)[0][1])
        return round(prob, 4), "xgboost"
    except Exception as exc:
        hba = _safe_float(patient.get("hba1c", patient.get("HbA1c", 5.5)))
        fbs = _safe_float(patient.get("fasting_blood_sugar",
                          patient.get("FastingBloodSugar", 100)))
        p   = min(1.0, max(0.0, (hba - 4.5) / 6.0 * 0.6 + (fbs - 70) / 200.0 * 0.4))
        return round(p, 4), f"heuristic ({exc})"


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def risk_tier(p: float) -> tuple:
    if p >= 0.65: return "High",     "#f87171", "🔴"
    if p >= 0.35: return "Moderate", "#fbbf24", "🟡"
    return "Low", "#34d399", "🟢"

def simulate_projection(rec: dict, years: int = 5) -> list:
    pts = []
    for y in range(1, years + 1):
        m = dict(rec)
        m["bmi"]                 = str(_safe_float(m.get("bmi", 25)) + 0.25 * y)
        m["fasting_blood_sugar"] = str(_safe_float(m.get("fasting_blood_sugar", 100)) + 2 * y)
        p, _ = predict_prob(m)
        pts.append(round(p * 100, 1))
    return pts

def ai_summary(rec: dict, prob: float) -> str:
    hba  = _safe_float(rec.get("hba1c", rec.get("HbA1c", 5.5)))
    bmi  = _safe_float(rec.get("bmi",   rec.get("BMI",   25.0)))
    hs   = "normal" if hba < 5.7 else ("pre-diabetes range" if hba < 6.5 else "diabetes range")
    bs   = "healthy" if bmi < 25 else ("overweight" if bmi < 30 else "obese")
    tier, _, icon = risk_tier(prob)
    msgs = {
        "Low":      f"HbA1c {hba:.1f}% ({hs}), BMI {bmi:.1f} ({bs}). Maintain current lifestyle; annual screening recommended.",
        "Moderate": f"HbA1c {hba:.1f}% ({hs}), BMI {bmi:.1f} ({bs}). Dietary adjustment and increased physical activity advised. Follow-up in 3–6 months.",
        "High":     f"HbA1c {hba:.1f}% ({hs}), BMI {bmi:.1f} ({bs}). Urgent clinical review, medication assessment and lifestyle intervention recommended.",
    }
    proj = simulate_projection(rec)
    return (f"{icon} **{tier} Risk ({prob * 100:.1f}%)** — {msgs[tier]} "
            f"Projected 5-yr: {' → '.join(str(x) + '%' for x in proj)}")

def fmt(v, decimals: int = 2) -> str:
    if v is None or v == "":
        return "—"
    try:
        return f"{float(v):.{decimals}f}"
    except Exception:
        return str(v)


# ══════════════════════════════════════════════════════════════════════════════
#  CHARTS
# ══════════════════════════════════════════════════════════════════════════════

def gauge_chart(prob: float):
    tier, color, _ = risk_tier(prob)
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta", value=round(prob * 100, 1),
        number={"suffix": "%", "font": {"color": "white", "family": "Outfit", "size": 36}},
        delta={"reference": 50, "increasing": {"color": "#f87171"},
               "decreasing": {"color": "#34d399"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#475569",
                     "tickfont": {"color": "#475569"}},
            "bar": {"color": color, "thickness": 0.25},
            "bgcolor": "rgba(0,0,0,0)", "bordercolor": "rgba(0,0,0,0)",
            "steps": [{"range": [0,  35], "color": "rgba(52,211,153,0.08)"},
                      {"range": [35, 65], "color": "rgba(251,191,36,0.08)"},
                      {"range": [65,100], "color": "rgba(248,113,113,0.08)"}],
            "threshold": {"line": {"color": color, "width": 3}, "value": prob * 100},
        },
        title={"text": f"<b>{tier} Risk</b>",
               "font": {"color": color, "family": "Outfit", "size": 16}},
    ))
    fig.update_layout(height=260, margin=dict(t=40, b=10, l=20, r=20),
                      paper_bgcolor="rgba(0,0,0,0)", font=dict(color="white"))
    return fig

def proj_chart(pts: list):
    colors = ["#34d399" if p < 35 else ("#fbbf24" if p < 65 else "#f87171") for p in pts]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(1, 6)), y=pts, mode="lines+markers",
        line=dict(color="#6366f1", width=3, shape="spline"),
        marker=dict(size=10, color=colors,
                    line=dict(color="rgba(255,255,255,0.2)", width=2)),
        fill="tozeroy", fillcolor="rgba(99,102,241,0.06)", name="Risk %",
    ))
    fig.add_hline(y=65, line=dict(color="#f87171", dash="dot", width=1),
                  annotation_text="High", annotation_font_color="#f87171")
    fig.add_hline(y=35, line=dict(color="#fbbf24", dash="dot", width=1),
                  annotation_text="Moderate", annotation_font_color="#fbbf24")
    fig.update_layout(
        title=dict(text="5-Year Risk Projection", font=dict(color="#94a3b8", size=14)),
        xaxis=dict(title="Year", tickvals=list(range(1, 6)),
                   gridcolor="rgba(255,255,255,0.04)", color="#64748b"),
        yaxis=dict(title="Risk %", range=[0, 100],
                   gridcolor="rgba(255,255,255,0.04)", color="#64748b"),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"), height=260,
        margin=dict(t=40, b=20, l=20, r=20), showlegend=False,
    )
    return fig

def vitals_radar(rec: dict):
    cats = ["HbA1c", "BMI", "BP", "Cholesterol", "Glucose", "Age"]
    vals = [
        min(_safe_float(rec.get("hba1c",            rec.get("HbA1c",            5.5))) / 14  * 100, 100),
        min(_safe_float(rec.get("bmi",              rec.get("BMI",              25)))  / 50  * 100, 100),
        min(_safe_float(rec.get("systolic_bp",      rec.get("SystolicBP",       120))) / 200 * 100, 100),
        min(_safe_float(rec.get("cholesterol_total",rec.get("CholesterolTotal", 180))) / 400 * 100, 100),
        min(_safe_float(rec.get("fasting_blood_sugar",rec.get("FastingBloodSugar",100)))/300 * 100, 100),
        min(_safe_float(rec.get("age",              rec.get("Age",              40)))  / 100 * 100, 100),
    ]
    fig = go.Figure(go.Scatterpolar(
        r=vals + [vals[0]], theta=cats + [cats[0]], fill="toself",
        fillcolor="rgba(99,102,241,0.12)", line=dict(color="#6366f1", width=2),
        marker=dict(color="#a5b4fc", size=6),
    ))
    fig.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(visible=True, range=[0, 100],
                            gridcolor="rgba(255,255,255,0.06)",
                            tickfont=dict(color="#475569", size=9)),
            angularaxis=dict(gridcolor="rgba(255,255,255,0.06)",
                             tickfont=dict(color="#94a3b8", size=11)),
        ),
        paper_bgcolor="rgba(0,0,0,0)", font=dict(color="white"),
        height=280, margin=dict(t=20, b=20, l=30, r=30), showlegend=False,
        title=dict(text="Health Profile Radar", font=dict(color="#94a3b8", size=13)),
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  HEADER / SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

def render_header(subtitle: str = "Smart Diabetes EMR"):
    sb  = "🟢 Connected" if get_supabase() else "🟡 Demo Mode"
    ml  = "🧠 XGBoost · Loaded"
    jwt = "🔐 JWT QR Active" if JWT_AVAILABLE else "🔐 JWT (fallback)"
    st.markdown(f"""
    <div class="topbar">
        <div style="font-size:32px">🩺</div>
        <div style="flex:1">
            <div class="topbar-title">GLUCO-LENS</div>
            <div class="topbar-sub">{subtitle}</div>
        </div>
        <div style="text-align:right;font-size:12px;color:#475569;line-height:2">
            {sb} &nbsp;·&nbsp; {ml}<br>{jwt}
        </div>
    </div>""", unsafe_allow_html=True)

def render_sidebar():
    with st.sidebar:
        role_icon = {"Admin": "👑", "Doctor": "👩‍⚕️", "Patient": "🫀"}.get(
            st.session_state.role, "")
        st.markdown(f"""
        <div class="sb-role">
            <div class="role-name">{role_icon} {st.session_state.role}</div>
            <div class="role-user">@{st.session_state.username or st.session_state.pat_pid}</div>
        </div>""", unsafe_allow_html=True)
        sb = get_supabase()
        st.markdown(f"""
        <div style="font-size:12px;color:#475569;margin-bottom:12px;
                    padding:8px 12px;background:rgba(255,255,255,0.02);border-radius:8px;">
            {'🟢 Supabase connected' if sb else '🟡 Demo data (add secrets to persist)'}
        </div>""", unsafe_allow_html=True)
        st.markdown("""
        <div style="font-size:12px;color:#475569;margin-bottom:16px;padding:8px 12px;
                    background:rgba(99,102,241,0.08);border-radius:8px;
                    border:1px solid rgba(99,102,241,0.15)">
            🧠 XGBoost model · Loaded
        </div>""", unsafe_allow_html=True)
        st.markdown("---")
        if st.button("🚪 Logout", use_container_width=True):
            for k in _defaults:
                st.session_state[k] = _defaults[k]
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
#  LOGIN
# ══════════════════════════════════════════════════════════════════════════════

def show_login():
    render_header("Secure Healthcare Login")
    _, mid, _ = st.columns([1, 1.4, 1])
    with mid:
        st.markdown('<div class="login-card">', unsafe_allow_html=True)
        st.markdown('<div class="login-logo">GLUCO-LENS</div>', unsafe_allow_html=True)
        st.markdown('<div class="login-sub">Smart Diabetes EMR</div>', unsafe_allow_html=True)

        tab_staff, tab_patient = st.tabs(["👩‍⚕️ Staff Login", "📷 Patient QR Login"])

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
                    st.error("Invalid credentials.")
            st.markdown('<div class="cred-badge">admin / admin123<br>doctor / doc123</div>',
                        unsafe_allow_html=True)

        with tab_patient:
            _patient_qr_login_widget()

        st.markdown("</div>", unsafe_allow_html=True)

def _patient_qr_login_widget():
    raw_token = st.session_state.get("last_scanned_token")
    if raw_token and not st.session_state.logged_in:
        _attempt_jwt_login(raw_token)
        return
    st.markdown("""<div style="color:#64748b;font-size:12px;text-align:center;margin-bottom:8px;">
        Upload your QR image to log in</div>""", unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload QR Image", type=["png", "jpg", "jpeg"],
                                label_visibility="collapsed", key="qr_upload_login")
    if uploaded:
        img   = Image.open(uploaded)
        token = qr_from_image_file(img)
        if token:
            st.session_state.last_scanned_token = token
            _attempt_jwt_login(token)
        else:
            st.error("❌ Could not read QR. Ensure pyzbar/libzbar is installed.")

def _attempt_jwt_login(token: str):
    payload = verify_qr_token(token)
    if payload:
        pid     = payload.get("patient_id", "")
        df      = db_fetch_all()
        pid_col = next((c for c in df.columns if c.lower() == "patient_id"), None)
        rec     = None
        if pid_col:
            rows = df[df[pid_col].astype(str).str.strip() == pid]
            if not rows.empty:
                rec = rows.iloc[0].to_dict()
        if rec:
            st.session_state.logged_in  = True
            st.session_state.role       = "Patient"
            st.session_state.username   = None
            st.session_state.pat_pid    = pid
            st.session_state.pat_record = rec
            st.session_state.last_scanned_token = None
            st.markdown('<div class="success-badge">✅ Logged in securely via QR · Passwordless Auth</div>',
                        unsafe_allow_html=True)
            time.sleep(0.8)
            st.rerun()
        else:
            st.error("⚠️ QR verified but patient record not found.")
    else:
        st.error("❌ Invalid or expired QR token.")
        st.session_state.last_scanned_token = None


# ══════════════════════════════════════════════════════════════════════════════
#  PATIENT RECORD RENDERER
# ══════════════════════════════════════════════════════════════════════════════

def _render_patient_record(rec: dict, pid: str):
    prob, src = predict_prob(rec)
    tier, color, icon = risk_tier(prob)
    name = rec.get("name", rec.get("Name", pid))

    st.markdown(f"""
    <div class="glass" style="border-left:3px solid {color};">
        <div style="display:flex;justify-content:space-between;align-items:center;
                    flex-wrap:wrap;gap:16px;">
            <div>
                <div style="font-size:24px;font-weight:700;color:#e2e8f5;">
                    Welcome, {name} 👋</div>
                <div style="font-size:13px;color:#64748b;margin-top:4px;">
                    Patient ID: {pid} · via {src}</div>
            </div>
            <div style="text-align:center;background:rgba(255,255,255,0.04);
                        border-radius:16px;padding:16px 28px;
                        border:1px solid rgba(255,255,255,0.07);">
                <div style="font-size:40px;font-weight:800;color:{color};">
                    {prob * 100:.1f}%</div>
                <div style="font-size:12px;color:{color};letter-spacing:1px;margin-top:4px;">
                    {icon} {tier.upper()} RISK</div>
            </div>
        </div>
    </div>""", unsafe_allow_html=True)

    v_keys = [("fasting_blood_sugar","FastingBloodSugar","Glucose"),
              ("hba1c","HbA1c","HbA1c %"), ("bmi","BMI","BMI"),
              ("cholesterol_total","CholesterolTotal","Cholesterol"),
              ("systolic_bp","SystolicBP","Systolic BP"),
              ("diastolic_bp","DiastolicBP","Diastolic BP")]
    vc = st.columns(len(v_keys))
    for i, (k1, k2, lbl) in enumerate(v_keys):
        vc[i].metric(lbl, fmt(rec.get(k1, rec.get(k2, ""))))

    st.markdown("<br>", unsafe_allow_html=True)
    cl, cr = st.columns([3, 2])
    with cl:
        st.plotly_chart(proj_chart(simulate_projection(rec)), use_container_width=True)
    with cr:
        st.plotly_chart(gauge_chart(prob), use_container_width=True)
        st.plotly_chart(vitals_radar(rec), use_container_width=True)

    st.markdown(f'<div class="ai-box">🤖 <b>Your Health Summary</b><br><br>'
                f'{ai_summary(rec, prob)}</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    remarks = rec.get("doctor_remarks", rec.get("DoctorRemarks", ""))
    if remarks:
        st.markdown(f"""<div class="glass"><p class="sec-head">📝 Doctor's Notes</p>
        <div style="color:#94a3b8;font-size:14px;line-height:1.7;">{remarks}</div>
        </div>""", unsafe_allow_html=True)

    st.download_button("⬇️ Download My Record",
        data=pd.DataFrame([{k: v for k, v in rec.items()
                            if not k.startswith("_")}]).to_csv(index=False),
        file_name=f"{pid}_record.csv", mime="text/csv")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<p class="sec-head">🌿 Lifestyle Risk Estimator</p>',
                unsafe_allow_html=True)
    with st.form("lifestyle_form"):
        c1, c2 = st.columns(2)
        q1 = c1.select_slider("Exercise (mins/day)",    ["0","10-30","30-60","60+"],   value="30-60")
        q2 = c2.select_slider("Added sugar (tsp/day)",  ["0-5","6-10","11-20","20+"],  value="6-10")
        q3 = c1.select_slider("Veg/fruit servings/day", ["0-1","2-3","4-5","5+"],      value="2-3")
        q4 = c2.select_slider("Sleep (hrs/night)",      ["<5","5-6","6-7","7+"],       value="6-7")
        q5 = c1.select_slider("Fast food frequency",
                               ["Daily","Few/week","Weekly","Rarely"],                 value="Few/week")
        q6 = c2.select_slider("Stress level (0-10)",    ["0-2","3-5","6-8","9-10"],    value="3-5")
        sub = st.form_submit_button("Calculate →", use_container_width=True)

    if sub:
        sm = {"0":0,"10-30":1,"30-60":2,"60+":3,"0-5":3,"6-10":2,"11-20":1,"20+":0,
              "0-1":0,"2-3":1,"4-5":2,"5+":3,"<5":0,"5-6":1,"6-7":2,"7+":3,
              "Daily":0,"Few/week":1,"Weekly":2,"Rarely":3,"0-2":3,"3-5":2,"6-8":1,"9-10":0}
        total  = sum(sm.get(q, 0) for q in [q1, q2, q3, q4, q5, q6])
        mod    = (total - 9) / 18.0
        new_p  = float(np.clip(prob - mod * 0.25, 0, 1))
        diff   = new_p - prob
        hba_b  = _safe_float(rec.get("hba1c", rec.get("HbA1c", 5.5)))
        hba_n  = hba_b + diff * 2
        ca, cb, cc = st.columns(3)
        ca.metric("Current Risk",  f"{prob * 100:.1f}%")
        cb.metric("Adjusted Risk", f"{new_p * 100:.1f}%", delta=f"{diff * 100:+.1f}%")
        cc.metric("Est. HbA1c",    f"{hba_n:.2f}%",       delta=f"{hba_n - hba_b:+.2f}")
        lm = {"Exercise": sm[q1], "Sugar": sm[q2], "Veg/Fruit": sm[q3],
              "Sleep": sm[q4], "Fast Food": sm[q5], "Stress": sm[q6]}
        fig = px.bar(pd.DataFrame({"Factor": list(lm.keys()), "Score": list(lm.values())}),
                     x="Factor", y="Score", color="Score",
                     color_continuous_scale=[[0,"#f87171"],[0.5,"#fbbf24"],[1,"#34d399"]],
                     title="Lifestyle Score Breakdown (0=poor, 3=excellent)")
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          font=dict(color="#94a3b8"), height=260, margin=dict(t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PATIENT PORTAL
# ══════════════════════════════════════════════════════════════════════════════

def patient_portal():
    render_header("Patient Portal")
    render_sidebar()
    pid = st.session_state.pat_pid
    rec = st.session_state.pat_record
    if not pid or not rec:
        st.error("⚠️ Session error. Please log out and scan your QR again.")
        return
    _render_patient_record(rec, pid)


# ══════════════════════════════════════════════════════════════════════════════
#  DOCTOR DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════

def doctor_dashboard():
    render_header("Doctor Dashboard")
    render_sidebar()
    df = db_fetch_all()

    st.markdown('<p class="sec-head">Patient Lookup</p>', unsafe_allow_html=True)
    c1, c2 = st.columns([3, 1])
    pid_in = c1.text_input("Patient ID", placeholder="e.g. P001",
                            label_visibility="collapsed")
    fetch  = c2.button("🔍 Fetch", use_container_width=True)

    if fetch and pid_in.strip():
        pid_col = next((c for c in df.columns if c.lower() == "patient_id"), None)
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
        prob, src = predict_prob(rec)
        tier, color, icon = risk_tier(prob)
        name = rec.get("name", rec.get("Name", pid))
        st.markdown(f"""
        <div class="glass" style="border-left:3px solid {color};">
            <div style="display:flex;justify-content:space-between;align-items:center;">
                <div>
                    <div style="font-size:22px;font-weight:700;">{name}</div>
                    <div style="font-size:13px;color:#64748b;margin-top:4px;">
                        ID: {pid} · Age: {fmt(rec.get('age',''), 0)}
                        · {rec.get('diagnosis','—')} · {src}
                    </div>
                </div>
                <div style="text-align:right;">
                    <div style="font-size:36px;font-weight:800;color:{color};">
                        {prob * 100:.1f}%</div>
                    <div style="font-size:13px;color:{color};">{icon} {tier} Risk</div>
                </div>
            </div>
        </div>""", unsafe_allow_html=True)

        v_keys = [("fasting_blood_sugar","FastingBloodSugar","Glucose"),
                  ("hba1c","HbA1c","HbA1c %"), ("bmi","BMI","BMI"),
                  ("cholesterol_total","CholesterolTotal","Cholesterol"),
                  ("systolic_bp","SystolicBP","Systolic BP"),
                  ("diastolic_bp","DiastolicBP","Diastolic BP")]
        vc = st.columns(len(v_keys))
        for i, (k1, k2, lbl) in enumerate(v_keys):
            vc[i].metric(lbl, fmt(rec.get(k1, rec.get(k2, ""))))

        st.markdown("<br>", unsafe_allow_html=True)
        cl, cr = st.columns([3, 2])
        with cl:
            st.plotly_chart(proj_chart(simulate_projection(rec)), use_container_width=True)
            st.markdown('<p class="sec-head">Full Record</p>', unsafe_allow_html=True)
            clean = {k: fmt(v) for k, v in rec.items() if not k.startswith("_")}
            st.dataframe(pd.DataFrame(clean.items(), columns=["Field", "Value"]),
                         use_container_width=True, hide_index=True, height=320)
        with cr:
            st.plotly_chart(gauge_chart(prob), use_container_width=True)
            st.plotly_chart(vitals_radar(rec), use_container_width=True)
            try:
                qp = generate_secure_qr(pid)
                st.image(qp, width=140, caption="Secure JWT QR")
                st.markdown(f"""<div class="jwt-info">
                    🔐 JWT · HS256 · Exp: {JWT_EXPIRY_DAYS}d<br>
                    Payload: patient_id + role + iat + exp
                </div>""", unsafe_allow_html=True)
            except Exception:
                pass

        st.markdown(f'<div class="ai-box">🤖 <b>AI Assessment</b><br><br>'
                    f'{ai_summary(rec, prob)}</div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<p class="sec-head">Doctor Remarks</p>', unsafe_allow_html=True)
        note = st.text_area("Remarks", value=rec.get("doctor_remarks", ""),
                             height=100, label_visibility="collapsed")
        if st.button("💾 Save Remarks to Database"):
            updated = dict(rec)
            updated["doctor_remarks"] = note
            ok = db_upsert({k: updated[k] for k in updated if not k.startswith("_")})
            if ok:
                st.success("✅ Saved to Supabase!")
                st.session_state.doc_record["doctor_remarks"] = note
                st.cache_data.clear()
            else:
                st.warning("⚠️ Supabase not configured. Saved in session only.")
                st.session_state.doc_record["doctor_remarks"] = note
    else:
        st.markdown("""<div class="glass" style="text-align:center;padding:48px;color:#475569;">
            <div style="font-size:48px;margin-bottom:16px;">🔍</div>
            <div style="font-size:16px;">Enter a Patient ID to view their record</div>
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  ADMIN DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════

def admin_dashboard():
    render_header("Admin Dashboard")
    render_sidebar()
    df = db_fetch_all()

    total   = len(df)
    hba_col = next((c for c in df.columns if c.lower() == "hba1c"), None)
    age_col = next((c for c in df.columns if c.lower() == "age"),   None)
    avg_hba = pd.to_numeric(df[hba_col], errors="coerce").mean() if hba_col else float("nan")
    avg_age = pd.to_numeric(df[age_col], errors="coerce").mean() if age_col else float("nan")

    all_probs = []
    for _, row in df.iterrows():
        p, _ = predict_prob(row.to_dict())
        all_probs.append(p)
    df["_risk"] = all_probs
    high_risk   = int((np.array(all_probs) >= 0.65).sum())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("👥 Patients",  total)
    c2.metric("⚠️ High Risk", high_risk)
    c3.metric("📊 Avg HbA1c", f"{avg_hba:.2f}" if not np.isnan(avg_hba) else "—")
    c4.metric("🎂 Avg Age",   f"{avg_age:.0f}"  if not np.isnan(avg_age) else "—")

    st.markdown("<br>", unsafe_allow_html=True)
    tab1, tab2, tab3 = st.tabs(["📋 Patient Records", "➕ Add Patient", "📊 Analytics"])

    with tab1:
        disp = df.drop(columns=["_risk"], errors="ignore")
        st.dataframe(disp, use_container_width=True, height=400)
        st.download_button("⬇️ Export CSV",
            data=disp.to_csv(index=False).encode(),
            file_name="gluco_lens_patients.csv", mime="text/csv")

    with tab2:
        st.markdown('<p class="sec-head">New Patient Record</p>', unsafe_allow_html=True)
        with st.form("add_pt"):
            c1, c2, c3 = st.columns(3)
            pid_v  = c1.text_input("Patient ID*", value=f"P{total + 1:03d}")
            name_v = c2.text_input("Full Name*")
            diag_v = c3.text_input("Diagnosis")

            c1, c2, c3, c4 = st.columns(4)
            age_v = c1.number_input("Age",       18, 100,  40)
            gen_v = c2.selectbox("Gender", ["Male", "Female", "Other"])
            bmi_v = c3.number_input("BMI",       10.0, 60.0, 25.0)
            hba_v = c4.number_input("HbA1c (%)",  3.0, 15.0,  5.5)

            c1, c2, c3, c4 = st.columns(4)
            fbs_v  = c1.number_input("Fasting Blood Sugar", 50, 400, 100)
            chol_v = c2.number_input("Total Cholesterol",   50, 500, 180)
            sbp_v  = c3.number_input("Systolic BP",         80, 220, 120)
            dbp_v  = c4.number_input("Diastolic BP",        40, 130,  80)

            c1, c2, c3 = st.columns(3)
            hyp_v = c1.selectbox("Hypertension",           ["No", "Yes"])
            hd_v  = c2.selectbox("Heart Disease",          ["No", "Yes"])
            smk_v = c3.selectbox("Smoking History",
                                 ["never", "former", "current", "ever",
                                  "not current", "No Info"])
            remarks_v = st.text_area("Doctor Remarks", height=80)

            if st.form_submit_button("✅ Save Patient & Generate Secure QR",
                                     use_container_width=True):
                if not name_v.strip():
                    st.error("Name is required.")
                else:
                    row_data = {
                        "patient_id":          pid_v.strip(),
                        "name":                name_v.strip(),
                        "age":                 age_v,
                        "gender":              gen_v,
                        "bmi":                 bmi_v,
                        "hba1c":               hba_v,
                        "fasting_blood_sugar": fbs_v,
                        "cholesterol_total":   chol_v,
                        "systolic_bp":         sbp_v,
                        "diastolic_bp":        dbp_v,
                        "hypertension":        1 if hyp_v == "Yes" else 0,
                        "heart_disease":       1 if hd_v  == "Yes" else 0,
                        "smoking_history":     smk_v,
                        "diagnosis":           diag_v,
                        "doctor_remarks":      remarks_v,
                        "created_at":          datetime.utcnow().isoformat(),
                    }
                    test_prob, _ = predict_prob(row_data)
                    t_name, t_color, t_icon = risk_tier(test_prob)
                    st.markdown(f"""<div class="glass"
                        style="border-left:3px solid {t_color};margin-bottom:12px;">
                        <div style="font-size:16px;font-weight:600;">
                            {t_icon} Predicted Risk:
                            <span style="color:{t_color}">
                                {test_prob * 100:.1f}% ({t_name})
                            </span>
                        </div>
                        <div style="font-size:12px;color:#64748b;margin-top:4px;">
                            XGBoost prediction verified ✓
                        </div>
                    </div>""", unsafe_allow_html=True)

                    ok = db_upsert(row_data)
                    try:
                        qp = generate_secure_qr(pid_v.strip())
                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.image(qp, width=180, caption=f"JWT QR · {pid_v}")
                        with col_b:
                            st.markdown(f"""<div class="jwt-info" style="margin-top:0">
                                🔐 Secure JWT Token<br>Patient: {pid_v.strip()}<br>
                                Algorithm: HS256<br>Expires: {JWT_EXPIRY_DAYS} days<br><br>
                                Patient scans → Auto-login
                            </div>""", unsafe_allow_html=True)
                    except Exception as e:
                        st.warning(f"QR gen error: {e}")

                    if ok:
                        st.success(f"✅ Patient {pid_v} saved!")
                        st.cache_data.clear()
                    else:
                        st.warning("⚠️ Supabase not connected — add secrets to persist.")

    with tab3:
        col1, col2 = st.columns(2)
        if hba_col:
            hba_vals = pd.to_numeric(df[hba_col], errors="coerce").dropna()
            with col1:
                fig = px.histogram(hba_vals, nbins=20, title="HbA1c Distribution",
                                   color_discrete_sequence=["#6366f1"])
                fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                                  plot_bgcolor="rgba(0,0,0,0)",
                                  font=dict(color="#94a3b8"), height=280,
                                  margin=dict(t=40, b=10))
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            rc = pd.Series(
                ["High" if p >= 0.65 else "Moderate" if p >= 0.35 else "Low"
                 for p in all_probs]
            ).value_counts()
            fig2 = px.pie(values=rc.values, names=rc.index,
                          title="Risk Distribution",
                          color_discrete_map={"High":"#f87171","Moderate":"#fbbf24","Low":"#34d399"},
                          hole=0.55)
            fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                               font=dict(color="#94a3b8"), height=280,
                               margin=dict(t=40, b=10))
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<p class="sec-head">🔬 Model Feature Importances (XGBoost)</p>',
                    unsafe_allow_html=True)
        try:
            fi_df = pd.DataFrame({
                "Feature":    FEATURE_COLUMNS,
                "Importance": MODEL.feature_importances_,
            }).sort_values("Importance", ascending=False).head(13)
            fig3 = px.bar(fi_df, x="Importance", y="Feature", orientation="h",
                          color="Importance",
                          color_continuous_scale=[[0,"#6366f1"],[1,"#38bdf8"]],
                          title="Feature Importances")
            fig3.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                               plot_bgcolor="rgba(0,0,0,0)",
                               font=dict(color="#94a3b8"), height=420,
                               margin=dict(t=40, b=10),
                               yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig3, use_container_width=True)
        except Exception as e:
            st.caption(f"Feature importance unavailable: {e}")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN ROUTER
# ══════════════════════════════════════════════════════════════════════════════

if not st.session_state.logged_in:
    show_login()
else:
    role = st.session_state.role
    if   role == "Admin":   admin_dashboard()
    elif role == "Doctor":  doctor_dashboard()
    elif role == "Patient": patient_portal()
    else:                   st.error("Unknown role. Please log out.")