import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn

from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score
from scipy.cluster.hierarchy import dendrogram, linkage

sns.set_style("whitegrid")


# Start MLflow

mlflow.set_experiment("PatrolIQ_Clustering")
mlflow.start_run()


# Load Data

df = pd.read_csv(r"C:\Users\SAKTHI\Desktop\myproject\PatrolIQ\data\feature_data\feature_dataset.csv")
print("Dataset Shape:", df.shape)

sample_df = df.sample(50000, random_state=42)
print("Sample Shape:", sample_df.shape)

# Log basic info
mlflow.log_param("sample_size", 50000)


# Geographic Features

geo_features = sample_df[
    ["latitude", "longitude", "lat_bin", "lon_bin", "district_cluster"]
]


# Elbow Method

inertia = []
k_range = range(2, 12)

for k in k_range:
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    model.fit(geo_features)
    inertia.append(model.inertia_)

plt.figure(figsize=(8,5))
plt.plot(k_range, inertia, marker="o")
plt.title("Elbow Method")
plt.xlabel("Clusters")
plt.ylabel("Inertia")
plt.close()


# KMeans

kmeans = KMeans(n_clusters=7, random_state=42, n_init=10)
sample_df["geo_cluster_kmeans"] = kmeans.fit_predict(geo_features)

print("KMeans clustering completed")

# Metrics
sil_score = silhouette_score(geo_features, sample_df["geo_cluster_kmeans"])
db_index = davies_bouldin_score(geo_features, sample_df["geo_cluster_kmeans"])

print("Silhouette:", sil_score)
print("DB Index:", db_index)

mlflow.log_param("kmeans_clusters", 7)
mlflow.log_metric("kmeans_silhouette", sil_score)
mlflow.log_metric("kmeans_db_index", db_index)

# Plot
plt.figure(figsize=(8,8))
plt.scatter(
    sample_df["longitude"],
    sample_df["latitude"],
    c=sample_df["geo_cluster_kmeans"],
    cmap="tab10",
    s=2
)

plt.title("KMeans Clusters")
plt.close()

mlflow.sklearn.log_model(kmeans, "kmeans_model")


# DBSCAN

dbscan = DBSCAN(eps=0.5, min_samples=10)
sample_df["geo_cluster_dbscan"] = dbscan.fit_predict(geo_features)

print("DBSCAN clustering completed")

mlflow.log_param("dbscan_eps", 0.5)
mlflow.log_param("dbscan_min_samples", 10)

# Handle noise points

mask = sample_df["geo_cluster_dbscan"] != -1

if len(set(sample_df["geo_cluster_dbscan"][mask])) > 1:
    db_sil = silhouette_score(
        geo_features[mask],
        sample_df["geo_cluster_dbscan"][mask]
    )
    mlflow.log_metric("dbscan_silhouette", db_sil)

# Plot
plt.figure(figsize=(8,8))
plt.scatter(
    sample_df["longitude"],
    sample_df["latitude"],
    c=sample_df["geo_cluster_dbscan"],
    cmap="tab10",
    s=2
)

plt.title("DBSCAN Clusters")
plt.close()


# Hierarchical

hier_sample = geo_features.sample(5000)
linked = linkage(hier_sample, method="ward")

plt.figure(figsize=(10,6))
dendrogram(linked)
plt.title("Hierarchical Dendrogram")
plt.close()


# Temporal Clustering

temporal_features = sample_df[
    [
        "hour",'season_Autumn','season_Spring','season_Summer','season_Winter',
        'day_name_Friday','day_name_Monday','day_name_Saturday',
        'day_name_Sunday','day_name_Thursday','day_name_Tuesday','day_name_Wednesday'
    ]
]

kmeans_temporal = KMeans(n_clusters=4, random_state=42)
sample_df["temporal_cluster"] = kmeans_temporal.fit_predict(temporal_features)

print("Temporal clustering completed")

temp_sil = silhouette_score(
    temporal_features,
    sample_df["temporal_cluster"]
)

mlflow.log_metric("temporal_silhouette", temp_sil)

# Plot
plt.figure(figsize=(8,6))
sns.scatterplot(
    x=sample_df["hour"],
    y=sample_df["temporal_cluster"],
    hue=sample_df["temporal_cluster"],
    palette="tab10"
)

plt.title("Temporal Clusters")
plt.close()

mlflow.sklearn.log_model(kmeans_temporal, "temporal_model")

# -----------------------------
# End MLflow
# -----------------------------
mlflow.end_run()

print("✅ MLflow tracking completed")