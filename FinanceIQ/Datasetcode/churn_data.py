import pandas as pd

# Sample data for customer churn prediction
data = {
    'account_activity': [1, 0, 1, 1, 0],  # 1: active, 0: inactive
    'service_usage': [3, 5, 2, 1, 0],  # Number of services used
    'demographics': ['young', 'middle-aged', 'senior', 'young', 'senior'],  # Age category
    'customer_feedback': [5, 3, 4, 2, 1],  # Rating from 1 to 5
    'churn': [0, 1, 0, 1, 1]  # 0: did not churn, 1: churned
}

# Convert to DataFrame
churn_df = pd.DataFrame(data)

# Save as CSV
churn_df.to_csv('customer_churn.csv', index=False)
 