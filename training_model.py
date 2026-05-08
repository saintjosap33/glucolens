import pandas as pd
from sklearn.model_selection import train_test_split
import joblib
from sklearn.metrics import accuracy_score,classification_report
from xgboost import XGBClassifier

# LOAD DATASET
df = pd.read_csv("data/diabetes_prediction_dataset.csv")

# CLEAN DATA
df = df.drop_duplicates()
df = df.dropna()

# ENCODING
df = pd.get_dummies(
    df,
    columns=["gender", "smoking_history"],
    drop_first=True
)

# FEATURES + LABELS
X = df.drop("diabetes", axis=1)
y = df["diabetes"]

# VIEW DATA
print(df.head())


# SPLITTING DATA
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=10
)

model = XGBClassifier(
    n_estimators=300,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)


model.fit(X_train,y_train)

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test,predictions)

print("Accuracy ",accuracy)
print(classification_report(y_test, predictions))

joblib.dump(model, "models/diabetes_xgb.pkl")