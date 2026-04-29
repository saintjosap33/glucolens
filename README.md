# 🏥 GlucoLens – Smart EMR System with QR-Based Patient Login

GlucoLens is a **Smart Electronic Medical Record (EMR) system** built using **Streamlit and Supabase**, featuring **secure QR-based patient login** for seamless and passwordless authentication.

---

## 🚀 Features

* 🔐 **QR-Based Login (JWT Authentication)**
* 🧑‍⚕️ Patient data management using Supabase
* 📊 Health metrics tracking (BMI, HbA1c, BP, etc.)
* ⚡ Real-time data fetching and display
* 📱 QR scan → Instant patient dashboard
* ☁️ Deployed on Streamlit Cloud

---

## 🧠 How It Works

```text
QR Code (JWT Token)
        ↓
Decode Token
        ↓
Extract patient_id
        ↓
Fetch from Supabase
        ↓
Login Patient
```

---

## 🗄️ Database Schema (Supabase)

Table: `patients`

| Column                  | Type      |
| ----------------------- | --------- |
| patient_id              | TEXT      |
| name                    | TEXT      |
| age                     | INT       |
| gender                  | INT       |
| bmi                     | FLOAT     |
| hba1c                   | FLOAT     |
| fasting_blood_sugar     | INT       |
| cholesterol_total       | INT       |
| systolic_bp             | INT       |
| diastolic_bp            | INT       |
| hypertension            | INT       |
| family_history_diabetes | INT       |
| smoking                 | INT       |
| physical_activity       | INT       |
| diet_quality            | INT       |
| sleep_quality           | INT       |
| diagnosis               | TEXT      |
| doctor_remarks          | TEXT      |
| created_at              | TIMESTAMP |

---

## 🔐 QR Token Structure

QR codes store a JWT token:

```json
{
  "patient_id": "P001",
  "role": "patient",
  "iat": <issued_time>,
  "exp": <expiry_time>
}
```

---

## 🛠️ Tech Stack

* **Frontend**: Streamlit
* **Backend / DB**: Supabase
* **QR Detection**: OpenCV
* **Authentication**: JWT

---

## ⚙️ Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/your-username/glucolens.git
cd glucolens
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the app

```bash
streamlit run app.py
```

---

## 🌐 Deployment

* Hosted on **Streamlit Cloud**
* Removed system dependencies (`libzbar`) for compatibility
* Uses `opencv-python-headless` for QR scanning

---

## ⚠️ Common Issues & Fixes

### ❌ QR verified but patient not found

✔ Cause: Patient ID in QR does not exist in DB
✔ Fix: Insert patient into database before generating QR

---

### ❌ QR not scanning

✔ Use OpenCV instead of pyzbar
✔ Ensure image quality is clear

---

### ❌ Column not found error

✔ Ensure schema matches exactly
✔ Example: `diet_quality`, `doctor_remarks`

---

## ✅ Best Practices

* Always **insert patient → then generate QR**
* Normalize values:

```python
pid = str(pid).strip().upper()
```

* Keep JWT tokens **short-lived (expiry time)**

---

## 🚀 Future Enhancements

* 📦 Bulk patient dataset generation
* 🧑‍⚕️ Doctor dashboard
* 📊 Data analytics & visualization
* 🔐 Advanced QR security (token tracking)
* 📱 Mobile-friendly UI

---

## 👨‍💻 Author

Adithya – Smart Healthcare System Developer

---

## 📌 Project Status

✅ QR Login Working
✅ Supabase Integration Working
✅ Deployment Successful

🚧 Enhancements in progress...

---

## ⭐ Contribute

Feel free to fork, improve, and submit PRs!

---
