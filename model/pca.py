import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


# Start MLflow

mlflow.set_experiment("PatrolIQ_PCA_TSNE")
mlflow.start_run()


# Load Data

df = pd.read_csv(r"C:\Users\SAKTHI\Desktop\myproject\PatrolIQ\data\feature_data\feature_dataset.csv")
print("Dataset Shape:", df.shape)


# Feature Selection

drop_cols = [
    'id','case_number','date','block','iucr',
    'primary_type','description','location_description',
    'updated_on','location'
]

X = df.drop(columns=drop_cols, errors='ignore')
X = X.select_dtypes(include=[np.number])
X = X.fillna(0)

print("Final Feature Shape:", X.shape)

mlflow.log_param("num_features", X.shape[1])


# PCA

pca = PCA(n_components=3)
X_pca = pca.fit_transform(X)

explained_var = pca.explained_variance_ratio_
total_var = np.sum(explained_var)

print("\nExplained Variance:", explained_var)
print("Total Variance:", total_var)

# Log PCA metrics

mlflow.log_param("pca_components", 3)
mlflow.log_metric("pca_total_variance", total_var)
mlflow.log_metric("pca_pc1", explained_var[0])
mlflow.log_metric("pca_pc2", explained_var[1])
mlflow.log_metric("pca_pc3", explained_var[2])


# Scree Plot

plt.figure()
plt.plot(range(1, len(explained_var)+1), np.cumsum(explained_var), marker='o')
plt.xlabel("Components")
plt.ylabel("Cumulative Variance")
plt.title("PCA Scree Plot")
plt.close()


# PCA Scatter

plt.figure()
plt.scatter(X_pca[:, 0], X_pca[:, 1], s=5, alpha=0.5)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA Projection")
plt.close()


# Feature Importance

importance = np.abs(pca.components_).sum(axis=0)

top_features = pd.Series(
    importance,
    index=X.columns
).sort_values(ascending=False)

print("\nTop 5 Features:")
print(top_features.head(5))


# Log PCA model
mlflow.sklearn.log_model(pca, "pca_model")


# t-SNE

sample_size = 5000
X_sample = X.iloc[:sample_size]

tsne = TSNE(
    n_components=2,
    perplexity=40,
    learning_rate='auto',
    random_state=42
)

X_tsne = tsne.fit_transform(X_sample)

mlflow.log_param("tsne_sample_size", sample_size)
mlflow.log_param("tsne_perplexity", 40)


# t-SNE Plot

plt.figure()
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], s=5, alpha=0.6)
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.title("t-SNE Clusters")
plt.close()


# End MLflow

mlflow.end_run()

print("✅ MLflow tracking completed")