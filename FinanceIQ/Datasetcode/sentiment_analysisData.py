import pandas as pd

# 📌 Sample Banking Sentiment Data
data = pd.DataFrame({
    "feedback": [
        "The bank's customer service is excellent!",
        "I had a terrible experience with the loan application.",
        "The mobile banking app is easy to use.",
        "The bank charges too many hidden fees.",
        "Great experience! The staff was very helpful.",
        "I am not happy with the high-interest rates.",
        "The account opening process was smooth and fast.",
        "Their customer support is unresponsive and slow.",
        "Overall, an average experience, nothing special.",
        "Fast and secure transactions every time.",
        "Long waiting times at the branch.",
        "Loan processing was quick and hassle-free.",
        "The ATM service is very poor.",
        "I love the new investment options!",
        "Too many hidden charges make it frustrating."
    ],
    "sentiment": [
        "Positive", "Negative", "Positive", "Negative", "Positive",
        "Negative", "Positive", "Negative", "Neutral", "Positive",
        "Negative", "Positive", "Negative", "Positive", "Negative"
    ]
})

# 📌 Save as CSV
csv_path = "C:\\Users\\Mohit\\OneDrive\\Desktop\\Bank fraud detection\\models\\sentiment_data.csv"
data.to_csv(csv_path, index=False)

print(f"✅ Sentiment dataset saved at: {csv_path}")
