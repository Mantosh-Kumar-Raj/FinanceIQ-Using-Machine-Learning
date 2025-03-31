import pandas as pd
import random

# Generate sample data
data = []
customer_types = ["Regular", "Premium", "Business"]

for _ in range(50):  # Generating 50 records
    customer_type = random.choice(customer_types)
    account_balance = round(random.uniform(1000, 50000), 2)  # Balance range
    transaction_amount = round(random.uniform(100, 10000), 2)  # Transaction size
    competitor_pricing = round(random.uniform(3.5, 6.5), 2)  # Competitor rate
    risk_score = round(random.uniform(0.1, 1.0), 2)  # Risk score (0-1)

    # Dynamic pricing logic (higher risk → higher price)
    base_price = competitor_pricing + (risk_score * 2) - (account_balance / 100000)
    optimal_pricing = round(max(base_price, competitor_pricing), 2)

    data.append([
        customer_type, account_balance, transaction_amount,
        competitor_pricing, risk_score, optimal_pricing
    ])

# Create DataFrame
df = pd.DataFrame(data, columns=[
    "Customer_Type", "Account_Balance", "Transaction_Amount",
    "Competitor_Pricing", "Risk_Score", "Optimal_Pricing"
])

# Save to CSV
csv_filename = "bank_dynamic_pricing.csv"
df.to_csv(csv_filename, index=False)

print(f"CSV file '{csv_filename}' created successfully!")
