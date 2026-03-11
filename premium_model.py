import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

# Load dataset
data = pd.read_csv("aug_train.csv")

# Encode categorical columns
label_enc = LabelEncoder()
for col in data.select_dtypes(include=['object']).columns:
    data[col] = label_enc.fit_transform(data[col])

# Select only 6 meaningful features
features = ["Gender", "Age", "Vehicle_Age", "Vehicle_Damage", "Annual_Premium", "Vintage"]
X = data[features]
y = data["Response"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "premium_model.pkl")

print("✅ Premium model retrained with 6 features")
