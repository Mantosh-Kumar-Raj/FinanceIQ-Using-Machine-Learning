import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib

# Create a dummy dataset for the purpose of demonstration
data = {
    'age': np.random.randint(18, 70, 100),
    'income': np.random.randint(20000, 100000, 100),
    'current_credit_score': np.random.randint(300, 850, 100),
    'spending_score': np.random.randint(0, 100, 100),
    'gender': np.random.choice(['Male', 'Female'], 100),
    'country': np.random.choice(['USA', 'UK', 'France', 'Germany', 'India'], 100),
    'target': np.random.randint(300, 850, 100)  # Target variable (credit score)
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Convert categorical variables to numerical
df['gender'] = df['gender'].map({'Male': 0, 'Female': 1})
df['country'] = df['country'].astype('category').cat.codes

# Feature selection
X = df[['age', 'income', 'current_credit_score', 'spending_score', 'gender', 'country']]
y = df['target']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest model
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Create the models directory if it doesn't exist
model_dir = 'D:/Bank fraud detection/models'
os.makedirs(model_dir, exist_ok=True)  # Create the directory if it doesn't exist

# Save the model
joblib.dump(model, os.path.join(model_dir, 'credit_scoring.pk1'))

print("Model saved successfully!")
