import pandas as pd
import numpy as np
import random

# Number of synthetic records
N = 10000

# Define possible values
vehicle_types = ["Sedan", "SUV", "Sports Car", "Luxury Car", "Utility", "Electric", "Commercial"]
accident_areas = ["Urban", "Rural"]
plan_types = ["Basic", "Standard", "Full Coverage", "Luxury Plan", "EV Plan"]

coverage_includes_options = [
    "Body, Engine, Theft, Accident",
    "Body, Engine, Natural Disaster",
    "Body, Theft, Personal Accident",
    "Full Coverage (Body, Engine, Theft, Accident, Natural Disaster, Personal Accident)",
    "EV Coverage (Battery, Electronics, Theft, Accident)"
]

coverage_excludes_options = [
    "Wear & Tear, Drunk Driving",
    "Unauthorized Modifications, Racing",
    "Driving without License, Drunk Driving",
    "Wear & Tear, Unauthorized Modifications",
    "Battery Degradation, Unauthorized Modifications"
]

# Generate synthetic data
data = {
    "CustomerID": range(1, N+1),
    "Age": np.random.randint(18, 70, N),
    "Gender": np.random.choice(["Male", "Female"], N),
    "DrivingExperience": np.random.randint(1, 40, N),
    "IncomeLevel": np.random.choice(["Low", "Medium", "High", "Very High"], N),
    "Region": np.random.choice(accident_areas, N),
    "VehicleType": np.random.choice(vehicle_types, N),
    "VehicleAge": np.random.randint(0, 15, N),
    "VehiclePrice": np.random.randint(500000, 100000000, N),  # ₹5 lakh to ₹10 crore
    "SafetyFeatures": np.random.choice(["Basic", "Advanced", "Premium"], N),
    "PastClaims": np.random.randint(0, 5, N),
    "ClaimSeverity": np.random.choice(["Minor", "Major", "Total Loss"], N),
    "AccidentArea": np.random.choice(accident_areas, N),
    "CoverageIncludes": np.random.choice(coverage_includes_options, N),
    "CoverageExcludes": np.random.choice(coverage_excludes_options, N),
}

df = pd.DataFrame(data)

# Calculate Premiums based on rules
df["Premium"] = (
    (df["VehiclePrice"] * np.random.uniform(0.02, 0.05)) +  # 2–5% of vehicle price
    (df["PastClaims"] * 10000) -                            # more claims → higher premium
    (df["DrivingExperience"] * 500)                         # more experience → lower premium
).astype(int)

# Assign Plan Type based on rules
def assign_plan(row):
    if row["VehiclePrice"] > 10000000:  # > ₹1 crore
        return "Full Coverage"
    elif row["VehicleType"] == "Luxury Car":
        return "Luxury Plan"
    elif row["VehicleType"] == "Electric":
        return "EV Plan"
    elif row["Premium"] < 20000:
        return "Basic"
    else:
        return "Standard"

df["PlanType"] = df.apply(assign_plan, axis=1)

# Save dataset
df.to_csv("synthetic_insurance_dataset.csv", index=False)

print("✅ Synthetic dataset created with 10,000 records!")
