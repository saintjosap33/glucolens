# GLUCO-LENS v4 — Setup Guide

## Streamlit Secrets (`.streamlit/secrets.toml`)
```toml
SUPABASE_URL = "https://xxxx.supabase.co"
SUPABASE_KEY = "eyJhbGci..."
JWT_SECRET   = "your-long-random-secret-change-this"
```

## Supabase Table SQL
```sql
create table patients (
  patient_id            text primary key,
  name                  text,
  age                   numeric,
  gender                int,
  bmi                   numeric,
  hba1c                 numeric,
  fasting_blood_sugar   numeric,
  cholesterol_total     numeric,
  systolic_bp           numeric,
  diastolic_bp          numeric,
  hypertension          int default 0,
  family_history_diabetes int default 0,
  smoking               int default 0,
  physical_activity     numeric default 3,
  diet_quality          numeric default 3,
  sleep_quality         numeric default 3,
  alcohol_consumption   numeric default 0,
  serum_creatinine      numeric default 0.9,
  bun_levels            numeric default 15,
  cholesterol_ldl       numeric default 100,
  cholesterol_hdl       numeric default 50,
  cholesterol_triglycerides numeric default 150,
  diagnosis             text,
  doctor_remarks        text,
  created_at            text
);
```

## QR Auth Flow
1. Admin adds patient → JWT QR auto-generated (HS256 signed, 365-day expiry)
2. Patient scans QR with app camera (WebRTC) or uploads image
3. Token verified → patient auto-logged in → sees ONLY their record
4. No password needed. Revoking = regenerating QR.

## packages.txt (Streamlit Cloud system deps)
```
libzbar0
libzbar-dev
```
This enables pyzbar to decode QR codes from camera frames.

## Demo Credentials
| Role    | Login          | Auth Method            |
|---------|----------------|------------------------|
| Admin   | admin/admin123 | Password               |
| Doctor  | doctor/doc123  | Password               |
| Patient | —              | JWT QR scan or manual ID |
