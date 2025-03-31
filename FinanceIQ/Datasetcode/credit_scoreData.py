import pandas as pd
import numpy as np
import random

# Define number of samples
num_samples = 500

# Generate synthetic data
np.random.seed(42)
random.seed(42)

data = {
    "Age": np.random.randint(21, 65, num_samples),
    "Annual_Income": np.random.randint(20000, 150000, num_samples),
    "Credit_Score": np.random.randint(300, 850, num_samples),
    "Loan_Amount": np.random.randint(5000, 50000, num_samples),
    "Loan_Term": np.random.choice([12, 24, 36, 48, 60], num_samples),
    "Employment_Status": np.random.choice(["Employed", "Self-Employed", "Unemployed"], num_samples),
    "Existing_Loans": np.random.randint(0, 5, num_samples),
    "Debt_to_Income_Ratio": np.round(np.random.uniform(0.1, 0.6, num_samples), 2),
    "Loan_Approval": np.random.choice(["Approved", "Rejected"], num_samples, p=[0.7, 0.3])
}

# Create DataFrame
df = pd.DataFrame(data)

# Save to CSV file
df.to_csv("credit_scoring_dataset.csv", index=False)

print("CSV file 'credit_scoring_dataset.csv' has been created successfully!")
