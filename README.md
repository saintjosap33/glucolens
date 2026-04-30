# GlucoLens – Smart EMR System with QR-Based Patient Login

GlucoLens is a smart Electronic Medical Record (EMR) system built using Streamlit and Supabase. It introduces a secure QR-based login mechanism that enables patients to access their medical data instantly without relying on traditional password-based authentication.

Overview

The system is designed to simplify patient data access while maintaining security and usability. By leveraging QR codes embedded with JWT tokens, GlucoLens enables seamless authentication and real-time data retrieval from the database.

Key Features
QR-based patient login using JWT authentication
Centralized patient data management with Supabase
Tracking of critical health metrics such as BMI, HbA1c, blood pressure, and more
Real-time data retrieval and display
Instant dashboard access through QR scan
Deployment-ready on Streamlit Cloud
System Workflow
QR Code (JWT Token)
↓
Token Decoding
↓
Extract patient\_id
↓
Fetch patient data from Supabase
↓
Authenticate and display dashboard
Database Schema

Table: patients

Column	Type
patient\_id	TEXT
name	TEXT
age	INT
gender	INT
bmi	FLOAT
hba1c	FLOAT
fasting\_blood\_sugar	INT
cholesterol\_total	INT
systolic\_bp	INT
diastolic\_bp	INT
hypertension	INT
family\_history\_diabetes	INT
smoking	INT
physical\_activity	INT
diet\_quality	INT
sleep\_quality	INT
diagnosis	TEXT
doctor\_remarks	TEXT
created\_at	TIMESTAMP
QR Token Structure

Each QR code stores a JWT token containing the following payload:

{
"patient\_id": "P001",
"role": "patient",
"iat": <issued\_time>,
"exp": <expiry\_time>
}
Technology Stack
Frontend: Streamlit
Backend and Database: Supabase
QR Detection: OpenCV
Authentication: JSON Web Tokens (JWT)
Installation and Setup

1. Clone the repository
git clone https://github.com/your-username/glucolens.git
cd glucolens
2. Install dependencies
pip install -r requirements.txt
3. Run the application
streamlit run app.py
Deployment

The application is deployed on Streamlit Cloud.

System-level dependencies such as libzbar were removed to ensure compatibility with the deployment environment. QR scanning is handled using opencv-python-headless.

Common Issues and Solutions
QR verified but patient not found

Cause: The patient ID encoded in the QR does not exist in the database
Solution: Insert the patient record into the database before generating the QR code

QR not scanning

Cause: Incompatible libraries or low image quality
Solution: Use OpenCV for scanning and ensure the QR image is clear

Column not found error

Cause: Mismatch between application code and database schema
Solution: Verify that all required columns such as diet\_quality and doctor\_remarks exist

Best Practices
Always insert patient data before generating QR codes
Normalize patient IDs to avoid mismatches:
pid = str(pid).strip().upper()
Use short-lived JWT tokens for improved security
Future Enhancements
Bulk patient dataset generation
Doctor-facing dashboard
Data analytics and visualization
Enhanced QR security with token tracking
Mobile-friendly interface
Author

Adithya
Smart Healthcare System Developer

Project Status

The system is functional with QR-based login, Supabase integration, and deployment completed. Additional features and improvements are currently in progress.

Contribution

Contributions are welcome. You can fork the repository, make improvements, and submit pull requests.

