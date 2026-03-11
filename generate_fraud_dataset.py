import pandas as pd
import numpy as np

# Number of rows
N = 10000

# Random seed for reproducibility
np.random.seed(42)

# Generate features
policy_tenure_days = np.random.randint(1, 3650, N)
premium_amount = np.random.uniform(5000, 80000, N)
coverage_amount = np.random.uniform(100000, 2000000, N)
policy_recently_upgraded = np.random.randint(0, 2, N)

customer_age = np.random.randint(18, 80, N)
num_previous_claims = np.random.randint(0, 10, N)
prior_fraud_flag = np.random.randint(0, 2, N)
late_premium_history = np.random.randint(0, 2, N)

claim_amount = np.random.uniform(5000, 1500000, N)
repair_estimate = np.random.uniform(5000, 1500000, N)
claim_delay_days = np.random.randint(0, 60, N)
police_report_filed = np.random.randint(0, 2, N)
witness_present = np.random.randint(0, 2, N)
photos_submitted = np.random.randint(0, 2, N)

accident_time = np.random.randint(0, 2, N)  # 0=Day, 1=Night
accident_type = np.random.randint(0, 2, N)  # 0=Multi, 1=Single
weather_condition = np.random.randint(0, 3, N)  # 0=Clear, 1=Rain, 2=Fog

vehicle_age = np.random.randint(0, 20, N)
vehicle_market_value = np.random.uniform(50000, 2000000, N)
injury_reported = np.random.randint(0, 2, N)

# Fraud logic scoring
fraud_score = np.zeros(N)

fraud_score += (policy_tenure_days < 30) * 2
fraud_score += (num_previous_claims > 3) * 2
fraud_score += (prior_fraud_flag == 1) * 3
fraud_score += (claim_delay_days > 7) * 2
fraud_score += (police_report_filed == 0) * 2
fraud_score += (witness_present == 0) * 1
fraud_score += (claim_amount > vehicle_market_value) * 2
fraud_score += (accident_type == 1) * 1

# Threshold for fraud
fraud_label = (fraud_score >= 4).astype(int)

# Assemble dataset
df = pd.DataFrame({
    "policy_tenure_days": policy_tenure_days,
    "premium_amount": premium_amount,
    "coverage_amount": coverage_amount,
    "policy_recently_upgraded": policy_recently_upgraded,
    "customer_age": customer_age,
    "num_previous_claims": num_previous_claims,
    "prior_fraud_flag": prior_fraud_flag,
    "late_premium_history": late_premium_history,
    "claim_amount": claim_amount,
    "repair_estimate": repair_estimate,
    "claim_delay_days": claim_delay_days,
    "police_report_filed": police_report_filed,
    "witness_present": witness_present,
    "photos_submitted": photos_submitted,
    "accident_time": accident_time,
    "accident_type": accident_type,
    "weather_condition": weather_condition,
    "vehicle_age": vehicle_age,
    "vehicle_market_value": vehicle_market_value,
    "injury_reported": injury_reported,
    "fraud_label": fraud_label
})

# Save dataset
df.to_csv("synthetic_fraud_dataset.csv", index=False)
print("✅ Synthetic fraud dataset generated with realistic features.")
