import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load dataset
df = pd.read_csv("synthetic_fraud_dataset.csv")

# Features and target
X = df.drop("fraud_label", axis=1)
y = df["fraud_label"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build model
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=12,
    random_state=42,
    class_weight="balanced"  # handle fraud imbalance
)

# Train
model.fit(X_train, y_train)

# Cross-validation score
cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")
print(f"✅ Model trained. CV Accuracy: {cv_scores.mean():.3f}")

# Save model
joblib.dump(model, "fraud_model.pkl")
print("✅ Fraud detection model saved as fraud_model.pkl")
