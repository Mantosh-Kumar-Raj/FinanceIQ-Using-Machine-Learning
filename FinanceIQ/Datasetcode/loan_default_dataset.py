import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Number of samples
num_samples = 1000

# Generate synthetic data
borrower_demographics = np.random.randint(18, 65, num_samples)  # Age of borrower
loan_amount = np.random.randint(5000, 50000, num_samples)  # Loan amount in $
income = np.random.randint(20000, 100000, num_samples)  # Annual income in $
credit_score = np.random.randint(300, 850, num_samples)  # Credit score
repayment_history = np.random.choice(["Good", "Average", "Poor"], num_samples)  # Categorical variable
economic_indicators = np.random.randn(num_samples)  # Random economic factor

# Encode repayment history
repayment_history_encoded = [1 if x == "Good" else 0.5 if x == "Average" else 0 for x in repayment_history]

# Generate target variable (loan default: 1 = Default, 0 = No Default)
loan_default = (
    (credit_score < 600) & (income < 40000) & (np.array(repayment_history_encoded) < 0.5)
).astype(int)

# Create DataFrame
df = pd.DataFrame({
    "borrower_demographics": borrower_demographics,
    "loan_amount": loan_amount,
    "income": income,
    "credit_score": credit_score,
    "repayment_history": repayment_history,
    "economic_indicators": economic_indicators,
    "loan_default": loan_default
})

# Save to CSV
df.to_csv("loan_performance.csv", index=False)

print("Synthetic dataset generated and saved as 'loan_performance.csv'")
