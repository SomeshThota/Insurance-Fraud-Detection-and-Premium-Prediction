import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import joblib

df = pd.read_csv("synthetic_insurance_dataset.csv")

# Regression Model (Premium Amount)
X_reg = df[["Age", "VehiclePrice", "VehicleAge", "PastClaims", "DrivingExperience"]]
y_reg = df["Premium"]

X_train, X_test, y_train, y_test = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

reg_model = RandomForestRegressor(
    n_estimators=20,   # ✏️ was 200
    max_depth=8,       # ✏️ added
    random_state=42
)
reg_model.fit(X_train, y_train)
joblib.dump(reg_model, "premium_amount_model.pkl", compress=3)  # ✏️ added compress=3
print("✅ Premium amount model saved")

# Classification Model (Plan Type)
X_clf = df[["Age", "VehiclePrice", "VehicleAge", "PastClaims", "DrivingExperience"]]
y_clf = df["PlanType"]

X_train, X_test, y_train, y_test = train_test_split(X_clf, y_clf, test_size=0.2, random_state=42)

clf_model = RandomForestClassifier(
    n_estimators=20,   # ✏️ was 200
    max_depth=8,       # ✏️ added
    random_state=42
)
clf_model.fit(X_train, y_train)
joblib.dump(clf_model, "premium_plan_model.pkl", compress=3)  # ✏️ added compress=3
print("✅ Premium plan model saved")