import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

data = pd.read_csv("aug_train.csv")

label_enc = LabelEncoder()
for col in data.select_dtypes(include=['object']).columns:
    data[col] = label_enc.fit_transform(data[col])

features = ["Gender", "Age", "Vehicle_Age", "Vehicle_Damage", "Annual_Premium", "Vintage"]
X = data[features]
y = data["Response"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✏️ reduced n_estimators + added max_depth
model = RandomForestClassifier(
    n_estimators=20,    # was 100 default → now 20
    max_depth=8,        # limits tree depth → smaller file
    random_state=42
)
model.fit(X_train, y_train)

joblib.dump(model, "premium_model.pkl", compress=3)  # ✏️ compress=3 shrinks file
print("✅ Premium model saved")