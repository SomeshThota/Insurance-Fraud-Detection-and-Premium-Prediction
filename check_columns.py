import pandas as pd

# Load fraud dataset
data = pd.read_csv("fraud_oracle.csv")

# Print all column names
print(data.columns)
