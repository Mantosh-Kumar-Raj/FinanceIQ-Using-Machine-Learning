import pandas as pd
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 🔹 Load dataset
data = pd.read_csv('customer_churn.csv')

# 🔹 One-Hot Encoding for categorical features
data = pd.get_dummies(data, columns=['demographics'], drop_first=True)

# 🔹 Debugging: Print available columns
print("Available columns after one-hot encoding:", data.columns.tolist())

# 🔹 Feature selection
demographic_columns = [col for col in data.columns if 'demographics_' in col]
X = data[['account_activity', 'service_usage', 'customer_feedback'] + demographic_columns]
y = data['churn']

# 🔹 Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🔹 Train classification model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 🔹 Predict on test data
y_pred = model.predict(X_test)

# 🔹 Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Churn Prediction Accuracy: {accuracy:.2f}')

# 🔹 Print feature importances
importances = model.feature_importances_
feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

print("Feature Importances:")
print(feature_importance_df)

# 🔹 Define the correct path for saving the model
model_dir = r"C:\Users\Mohit\OneDrive\Desktop\Bank fraud detection\models"
model_path = os.path.join(model_dir, "churn_model.pkl")

# 🔹 Ensure the directory exists
os.makedirs(model_dir, exist_ok=True)

# 🔹 Save the trained model
with open(model_path, 'wb') as f:
    pickle.dump(model, f)

print(f"✅ Model saved successfully at: {model_path}")
