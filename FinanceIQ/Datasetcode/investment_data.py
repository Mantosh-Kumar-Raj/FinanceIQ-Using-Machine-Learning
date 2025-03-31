import pandas as pd

# Create DataFrame
data = {
    "InvestorID": range(1, 21),
    "Risk_Tolerance": [0.7, 0.5, 0.8, 0.3, 0.6, 0.9, 0.4, 0.7, 0.5, 0.8, 0.3, 0.6, 0.9, 0.4, 0.7, 0.5, 0.8, 0.3, 0.6, 0.9],
    "Economic_Indicators": [0.85, 0.90, 0.75, 0.60, 0.80, 0.95, 0.65, 0.78, 0.70, 0.92, 0.55, 0.82, 0.97, 0.63, 0.88, 0.73, 0.89, 0.58, 0.76, 0.94],
    "Investment_Return": [12.5, 10.2, 14.3, 6.8, 11.1, 15.5, 8.3, 12.9, 9.5, 13.7, 5.9, 10.8, 16.2, 7.4, 12.1, 9.8, 14.0, 6.2, 11.4, 15.9]
}

df = pd.DataFrame(data)

# Save as CSV
df.to_csv("investment_data.csv", index=False)
print("✅ investment_data.csv file created successfully!")
