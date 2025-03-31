import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score

# ========================================
# 🔹 Step 1: Generate Synthetic Dataset
# ========================================

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
    "repayment_history": repayment_history_encoded,
    "economic_indicators": economic_indicators,
    "loan_default": loan_default
})

# Save dataset to CSV
dataset_path = "loan_performance.csv"
df.to_csv(dataset_path, index=False)
print(f"✅ Synthetic dataset saved at: {dataset_path}")

# ========================================
# 🔹 Step 2: Train Loan Default Prediction Model
# ========================================

# Load dataset
data = pd.read_csv(dataset_path)

# Feature selection
X = data[['borrower_demographics', 'loan_amount', 'income', 'credit_score', 'repayment_history', 'economic_indicators']]
y = data['loan_default']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = GradientBoostingClassifier()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
print(f'📊 Loan Default Prediction Accuracy: {accuracy:.4f}')

# ========================================
# 🔹 Step 3: Save Model as Pickle (.pkl)
# ========================================

# Save the trained model
model_path = "loan_default.pkl"
with open(model_path, "wb") as f:
    pickle.dump(model, f)

print(f"✅ Model saved at: {model_path}")
