import pandas as pd

# Load fraud dataset
fraud_data = pd.read_csv("fraud_oracle.csv")   # replace with the actual fraud file name
print("Fraud dataset shape:", fraud_data.shape)
print(fraud_data.head())

# Load premium dataset (train.csv is usually the main one)
premium_data = pd.read_csv("aug_train.csv")   # replace with the actual premium file name
print("Premium dataset shape:", premium_data.shape)
print(premium_data.head())
