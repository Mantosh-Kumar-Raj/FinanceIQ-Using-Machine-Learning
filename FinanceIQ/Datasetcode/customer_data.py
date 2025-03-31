import pandas as pd

# Sample customer data
data = {
    "customer_id": range(1, 11),
    "age": [25, 32, 40, 22, 35, 27, 45, 30, 50, 23],
    "income": [30000, 45000, 70000, 28000, 50000, 32000, 80000, 40000, 90000, 31000],
    "spending_score": [60, 40, 20, 75, 50, 65, 10, 55, 5, 70],
    "gender": ["Male", "Female", "Male", "Female", "Male", "Female", "Male", "Female", "Male", "Female"]
}

df = pd.DataFrame(data)
df.to_csv("customer_data.csv", index=False)

print("✅ `customer_data.csv` created successfully!")
