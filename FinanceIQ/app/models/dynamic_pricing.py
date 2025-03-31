import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder

# Load dataset
csv_filename = "bank_dynamic_pricing.csv"  # Update with the correct path if needed
data = pd.read_csv(csv_filename)

# Encode categorical variable (Customer_Type)
label_encoder = LabelEncoder()
data["Customer_Type"] = label_encoder.fit_transform(data["Customer_Type"])

# Feature selection
X = data[["Customer_Type", "Account_Balance", "Transaction_Amount", "Competitor_Pricing", "Risk_Score"]]
y = data["Optimal_Pricing"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = GradientBoostingRegressor()
model.fit(X_train, y_train)

# Save the model as a .pkl file
pkl_filename = "bank_dynamic_pricing.pkl"  # Update path if needed
with open(pkl_filename, 'wb') as file:
    pickle.dump(model, file)

print(f"Model saved as '{pkl_filename}' successfully!")
