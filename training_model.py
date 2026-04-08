import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier


# ─────────────────────────────────────────────────────────────
# LOAD DATASET
# ─────────────────────────────────────────────────────────────
df = pd.read_csv("data/diabetes_prediction_dataset.csv")


# ─────────────────────────────────────────────────────────────
# DATA CLEANING
# ─────────────────────────────────────────────────────────────
df = df.drop_duplicates()
df = df.dropna()


# ─────────────────────────────────────────────────────────────
# ENCODING CATEGORICAL COLUMNS
# ─────────────────────────────────────────────────────────────
df = pd.get_dummies(
    df,
    columns=["gender", "smoking_history"],
    drop_first=True
)


# ─────────────────────────────────────────────────────────────
# FEATURES & TARGET
# ─────────────────────────────────────────────────────────────
X = df.drop("diabetes", axis=1)
y = df["diabetes"]


# ─────────────────────────────────────────────────────────────
# TRAIN / TEST SPLIT
# ─────────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=10
)


# ─────────────────────────────────────────────────────────────
# MODEL CREATION
# ─────────────────────────────────────────────────────────────
model = XGBClassifier(
    n_estimators=300,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)


# ─────────────────────────────────────────────────────────────
# TRAIN MODEL
# ─────────────────────────────────────────────────────────────
model.fit(X_train, y_train)


# ─────────────────────────────────────────────────────────────
# PREDICTIONS
# ─────────────────────────────────────────────────────────────
predictions = model.predict(X_test)


# ─────────────────────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────────────────────
accuracy = accuracy_score(y_test, predictions)

print("\n✅ MODEL TRAINED SUCCESSFULLY")
print(f"Accuracy: {accuracy:.4f}\n")

print("Classification Report:\n")
print(classification_report(y_test, predictions))


# ─────────────────────────────────────────────────────────────
# CREATE MODELS DIRECTORY
# ─────────────────────────────────────────────────────────────
os.makedirs("models", exist_ok=True)


# ─────────────────────────────────────────────────────────────
# SAVE FEATURE COLUMNS
# ─────────────────────────────────────────────────────────────
joblib.dump(
    list(X.columns),
    "models/feature_columns.pkl"
)

print("✅ Saved: models/feature_columns.pkl")


# ─────────────────────────────────────────────────────────────
# SAVE TRAINED MODEL
# ─────────────────────────────────────────────────────────────
joblib.dump(
    model,
    "models/diabetes_xgb.pkl"
)

print("✅ Saved: models/diabetes_xgb.pkl")