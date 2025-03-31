import pandas as pd
import pickle
import os
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

# ✅ 1. Load dataset
file_path = "transactions.csv"  # Ensure this file exists
if not os.path.exists(file_path):
    print(f"❌ Error: {file_path} not found!")
    exit()

data = pd.read_csv(file_path)
print("✅ Dataset loaded successfully!")

# ✅ 2. Feature selection (Ensure these columns exist)
required_columns = ['transaction_amount', 'location', 'merchant_details', 'user_behavior']
for col in required_columns:
    if col not in data.columns:
        print(f"❌ Error: Missing column '{col}' in dataset!")
        exit()

X = data[required_columns]

# ✅ 3. Convert categorical features using Label Encoding
encoder = LabelEncoder()
for col in ['location', 'merchant_details', 'user_behavior']:
    X[col] = encoder.fit_transform(X[col])

print("✅ Categorical features encoded successfully!")

# ✅ 4. Train Isolation Forest model
model = IsolationForest(contamination=0.01, random_state=42)
model.fit(X)

# ✅ 5. Predict anomalies (fraud detection)
fraud_pred = model.predict(X)

# Convert -1 (fraud) to 1, and 1 (not fraud) to 0
data['is_fraud'] = (fraud_pred == -1).astype(int)
print("✅ Fraud predictions generated!")

# ✅ 6. Evaluate model (if true labels exist)
if 'true_labels' in data.columns:
    print("📊 Classification Report:")
    print(classification_report(data['true_labels'], data['is_fraud']))
else:
    print("⚠️ No ground truth labels found. Only predictions are generated.")

# ✅ 7. Save processed dataset with fraud predictions
output_csv = "transactions_with_predictions.csv"
data.to_csv(output_csv, index=False)
print(f"✅ Predictions saved successfully in '{output_csv}'!")

# ✅ 8. Save trained model as a pickle file
model_save_path = r"C:\Users\Mohit\OneDrive\Desktop\fraud_detection_model.pkl"  # Change path if needed

try:
    with open(model_save_path, "wb") as f:
        pickle.dump(model, f)
    print(f"✅ Model saved successfully at: {model_save_path}")
except Exception as e:
    print(f"❌ Error saving model: {e}")
