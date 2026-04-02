# train.py
import pickle
import os

from data_preprocessing import load_data, preprocess_data
from model_evaluation import find_optimal_k, train_kmeans

# Step 1: Load data
df = load_data()

# Step 2: Preprocess
scaled_data, scaler, original_df = preprocess_data(df)

# Step 3: Find optimal K (optional - run once)
find_optimal_k(scaled_data)

# Step 4: Train model
kmeans_model, clusters = train_kmeans(scaled_data, k=3)

# Step 5: Attach clusters
original_df['cluster'] = clusters

# Step 6: Identify risky cluster (IMPORTANT LOGIC)
cluster_summary = original_df.groupby('cluster').mean()
print("\nCluster Summary:\n", cluster_summary)

# Example logic: highest cost = risky
risky_cluster = cluster_summary['total_dollars'].idxmax()
print("Risky Cluster:", risky_cluster)

# Step 7: Save models
os.makedirs("models", exist_ok=True)

with open("models/kmeans_invoice_model.pkl", "wb") as f:
    pickle.dump(kmeans_model, f)

with open("models/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# Save risky cluster
with open("models/risky_cluster.pkl", "wb") as f:
    pickle.dump(risky_cluster, f)

print("\n✅ Models saved successfully!")