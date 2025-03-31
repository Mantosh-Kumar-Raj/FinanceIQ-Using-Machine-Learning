import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Sample Sentiment Dataset
data = {
    "feedback": [
        "I love this bank, their service is excellent!",
        "The customer support is terrible.",
        "Great mobile banking app, easy to use.",
        "Too many hidden charges, not happy at all.",
        "Very smooth loan application process.",
        "I will never recommend this bank to anyone!",
        "Decent experience, nothing special.",
        "My card got blocked, and support was not helpful.",
        "Best interest rates in the market!",
        "The website is slow and frustrating."
    ],
    "sentiment": [
        "positive", "negative", "positive", "negative", "positive",
        "negative", "neutral", "negative", "positive", "negative"
    ]
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Feature Extraction
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["feedback"])
y = df["sentiment"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Model
model = LogisticRegression()
model.fit(X_train, y_train)

# Save Model & Vectorizer
model_path = "C:\\Users\\Mohit\\OneDrive\\Desktop\\Bank fraud detection\\sentiment_model.pkl"
vectorizer_path = "C:\\Users\\Mohit\\OneDrive\\Desktop\\Bank fraud detection\\vectorizer.pkl"

with open(model_path, "wb") as file:
    pickle.dump(model, file)

with open(vectorizer_path, "wb") as file:
    pickle.dump(vectorizer, file)

print("✅ Sentiment Analysis Model Saved Successfully!")
