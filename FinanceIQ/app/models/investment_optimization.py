import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Step 1: Load Dataset
data = pd.read_csv('investment_data.csv')

# Step 2: Feature Selection
X = data[['Risk_Tolerance', 'Economic_Indicators']]  # Independent variables
y = data['Investment_Return']  # Target variable

# Step 3: Split Data into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 5: Make Predictions
y_pred = model.predict(X_test)

# Step 6: Evaluate the Model
mse = mean_squared_error(y_test, y_pred)
print(f'Investment Portfolio Optimization MSE: {mse:.2f}')

# Step 7: Save the Model using Pickle
with open('investment_model.pkl', 'wb') as file:
    pickle.dump(model, file)
print("✅ Model saved as 'investment_model.pkl'")

# Step 8: Load the Model (to test loading)
with open('investment_model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

# Step 9: Test Model Loading
sample_input = [[0.7, 0.85]]  # Example: Risk Tolerance = 0.7, Economic Indicators = 0.85
predicted_return = loaded_model.predict(sample_input)
print(f'Predicted Investment Return: {predicted_return[0]:.2f}')
