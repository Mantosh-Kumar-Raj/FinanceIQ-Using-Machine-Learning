import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib  # For saving and loading models
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load the dataset
file_name = "customer_data.csv"
 
try:
    data = pd.read_csv(file_name)
    print("✅ Dataset loaded successfully!")
    print("Available columns:", data.columns.tolist())  # Check available columns
except FileNotFoundError:
    print("❌ Error: customer_data.csv not found! Please check the file path.")
    exit()

# Define necessary features for clustering
required_columns = ['age', 'income', 'spending_score']
missing_columns = [col for col in required_columns if col not in data.columns]

if missing_columns:
    print(f"❌ Missing columns: {missing_columns}")
    exit()

# Select features
X = data[required_columns]

# Normalize the features (important for K-Means)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save the scaler for future use
joblib.dump(scaler, "scaler.pkl")

# Determine the optimal number of clusters using the Elbow Method
inertia = []
k_values = range(1, 11)

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

# Plot the Elbow Curve
plt.figure(figsize=(8, 5))
plt.plot(k_values, inertia, marker='o', linestyle='--', color='b')
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia (Within-cluster Sum of Squares)")
plt.title("Elbow Method for Optimal K")
plt.show()

# Apply K-Means with the optimal K (set manually based on elbow method)
optimal_k = 3  # Adjust based on the elbow plot
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
data['Cluster'] = kmeans.fit_predict(X_scaled)

# Save the trained K-Means model
model_filename = "kmeans_model.pkl"
joblib.dump(kmeans, model_filename)
print(f"✅ K-Means model saved as {model_filename}")

# Save clustered data
output_file = "customer_data_with_clusters.csv"
data.to_csv(output_file, index=False)
print(f"✅ Segmentation complete! Results saved to {output_file}")

# Visualize the clusters (Age vs. Spending Score)
plt.figure(figsize=(8, 6))
sns.scatterplot(x=data['age'], y=data['spending_score'], hue=data['Cluster'], palette='viridis', s=100)
plt.xlabel("Age")
plt.ylabel("Spending Score")
plt.title("Customer Segmentation Clusters")
plt.legend(title="Cluster")
plt.show()
