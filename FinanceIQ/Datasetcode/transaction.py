import pandas as pd
import random

# Generate synthetic transaction data
data = []
locations = ["New York", "California", "Texas", "Florida", "Illinois", "Nevada", "Ohio"]
merchants = ["Amazon", "Best Buy", "Walmart", "Apple Store", "Starbucks", "Target", "Car Dealer", 
             "McDonald's", "Luxury Store", "Jewelers", "Gas Station", "Electronics Store", 
             "Hotel Booking", "Online Store", "Crypto Exchange", "Local Market", "Cinema", "Car Rental"]
user_behaviors = ["Frequent Online Purchases", "Rare High-Value Purchase", "Daily Groceries", 
                  "Unusual High Spend", "Daily Coffee", "Seasonal Shopping", "Unexpected Large Purchase", 
                  "Regular Fast Food", "Rare Luxury Purchase", "High-Value Jewelry", "Frequent Travel", 
                  "Entertainment", "Unusual Crypto Activity", "Last-Minute Booking", "Regular Shopping"]

for transaction_id in range(1, 101):  # Generate 100 transactions
    transaction_amount = random.randint(10, 10000)  # Amount between $10 and $10,000
    location = random.choice(locations)
    merchant = random.choice(merchants)
    user_behavior = random.choice(user_behaviors)
    
    # Introduce some fraud patterns
    is_fraud = 1 if (transaction_amount > 3000 and user_behavior in ["Unusual High Spend", 
               "Unexpected Large Purchase", "Rare Luxury Purchase", "Unusual Crypto Activity"]) else 0

    data.append([transaction_id, transaction_amount, location, merchant, user_behavior, is_fraud])

# Create DataFrame
df = pd.DataFrame(data, columns=["transaction_id", "transaction_amount", "location", 
                                 "merchant_details", "user_behavior", "true_labels"])

# Save to CSV
df.to_csv("transactions.csv", index=False)

print("Synthetic transactions.csv file has been created successfully!")
