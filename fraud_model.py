import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
import joblib

df = pd.read_csv("synthetic_fraud_dataset.csv")  # ✏️ changed from fraud_oracle.csv

X = df.drop("fraud_label", axis=1)
y = df["fraud_label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(
    n_estimators=20,       # ✏️ was 200 → now 20
    max_depth=8,           # ✏️ was 12 → now 8
    random_state=42,
    class_weight="balanced"
)

model.fit(X_train, y_train)
joblib.dump(model, "fraud_model.pkl", compress=3)  # ✏️ added compress
print("✅ Fraud model saved")