# pip install pandas numpy sentence-transformers chromadb sklearn scipy matplotlib seaborn umap-learn evidently

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from chromadb import Client
from chromadb.config import Settings
from sklearn.decomposition import PCA
from scipy.spatial import distance
from scipy.stats import entropy, wasserstein_distance
from numpy.linalg import inv
import matplotlib.pyplot as plt
import seaborn as sns
import umap.umap_ as umap
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from alibi_detect.cd import MMDDrift

###############################################################################
# Create sample dataset
###############################################################################
# Replace this with loading your own dataset (e.g., df = pd.read_csv("your_data.csv"))
df = pd.DataFrame({
    "mrch_nm_raw": [
        "SAFEWAY #2841",
        "SHELL OIL 1000492406",
        "SUNOCO 091017800",
        "POPEYES 11562",
        "TEXAS ROADHOUSE FR #7005",
        "HARBOR FREIGHT TOOLS 233",
        "COSTCO WHOLESALE",
        "WALMART SUPERCENTER 1234",
        "TARGET T-123",
        "AMAZON MKTPLACE PMTS",
        "MCDONALD'S F23987",
        "BURGER KING #0081"
    ]
})

# Split into old and new sets for demonstration
mid = len(df)//2
old_df = df.iloc[:mid]
new_df = df.iloc[mid:]

###############################################################################
# Embedding Model
###############################################################################
model_name = "sentence-transformers/all-mpnet-base-v2"
model = SentenceTransformer(model_name)

old_embeddings = model.encode(old_df["mrch_nm_raw"].tolist(), convert_to_numpy=True)
new_embeddings = model.encode(new_df["mrch_nm_raw"].tolist(), convert_to_numpy=True)

###############################################################################
# Store embeddings in Chroma DB
###############################################################################
client = Client(
    Settings(
        chroma_db_impl="duckdb+parquet",
        persist_directory="chroma_db"
    )
)

collection = client.get_or_create_collection(name="merchant_embeddings")

collection.add(
    documents=old_df["mrch_nm_raw"].tolist(),
    embeddings=old_embeddings.tolist(),
    ids=[f"old_{i}" for i in range(len(old_df))]
)

collection.add(
    documents=new_df["mrch_nm_raw"].tolist(),
    embeddings=new_embeddings.tolist(),
    ids=[f"new_{i}" for i in range(len(new_df))]
)

###############################################################################
# Compute Drift Metrics (Traditional Methods)
###############################################################################
old_centroid = np.mean(old_embeddings, axis=0)
new_centroid = np.mean(new_embeddings, axis=0)

# Cosine Similarity (traditional)
cosine_sim = 1 - distance.cosine(old_centroid, new_centroid)

# Euclidean Distance (traditional)
euclidean_dist = np.linalg.norm(old_centroid - new_centroid)

# JSD (traditional)
def jensen_shannon_divergence(p, q):
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5*(p+q)
    return 0.5*entropy(p, m) + 0.5*entropy(q, m)

old_hist, bins = np.histogram(old_embeddings.flatten(), bins=50, density=True)
new_hist, _ = np.histogram(new_embeddings.flatten(), bins=bins, density=True)
jsd_value = jensen_shannon_divergence(old_hist, new_hist)

# KL Divergence (Symmetric)
def safe_kl_div(p, q):
    p = p / p.sum()
    q = q / q.sum()
    eps = 1e-10
    return np.sum(p * np.log((p+eps)/(q+eps)))

kl_old_new = safe_kl_div(old_hist, new_hist)
kl_new_old = safe_kl_div(new_hist, old_hist)
kl_sym = 0.5*(kl_old_new + kl_new_old)

# Wasserstein Distance (traditional)
old_vals = old_embeddings.flatten()
new_vals = new_embeddings.flatten()
w_distance = wasserstein_distance(old_vals, new_vals)

# Mahalanobis Distance (traditional)
cov = np.cov(old_embeddings, rowvar=False)
inv_cov = inv(cov + np.eye(cov.shape[0])*1e-10)
diff = new_centroid - old_centroid
mahalanobis_dist = np.sqrt(diff.T.dot(inv_cov).dot(diff))

# Covariance Distance (traditional)
new_cov = np.cov(new_embeddings, rowvar=False)
cov_dist = np.linalg.norm(cov - new_cov, ord='fro')

# PCA Shift (traditional)
explained_variance_old = PCA(n_components=2).fit(old_embeddings).explained_variance_ratio_
explained_variance_new = PCA(n_components=2).fit(new_embeddings).explained_variance_ratio_
pca_shift = np.linalg.norm(explained_variance_old - explained_variance_new)

###############################################################################
# Using Alibi Detect for a More Robust Drift Metric (MMD)
###############################################################################
# MMD Drift Detector: A robust method from alibi-detect for distribution drift.
# We'll set up the detector using default RBF kernel parameters for demonstration.
cd = MMDDrift(
    x_ref=old_embeddings,  # reference data (old embeddings)
    p_val=0.05,            # significance level
    preprocess_x=None,     # no preprocessing needed
    n_permutations=100,    # number of permutations for p-value estimation
    chunk_size=None
)

# Test for drift
mmd_result = cd.predict(new_embeddings)
mmd_driftscore = mmd_result['data']['distance']
mmd_pvalue = mmd_result['data']['p_val']
mmd_is_drift = bool(mmd_result['data']['is_drift'])

print(f"MMD Drift Score: {mmd_driftscore}")
print(f"MMD p-value: {mmd_pvalue}")
print(f"Is drift detected by MMD test? {'Yes' if mmd_is_drift else 'No'}")

###############################################################################
# Generate Evidently Report
###############################################################################
old_df_evidently = old_df.copy()
new_df_evidently = new_df.copy()
old_df_evidently["mrch_length"] = old_df_evidently["mrch_nm_raw"].apply(len)
new_df_evidently["mrch_length"] = new_df_evidently["mrch_nm_raw"].apply(len)

report = Report(metrics=[DataDriftPreset()])
report.run(reference_data=old_df_evidently, current_data=new_df_evidently)
report.save_html("data_drift_report.html")
print("Evidently data drift report saved as data_drift_report.html")

###############################################################################
# Plotting / Visualization
###############################################################################

# 1. Histogram of Old vs New embedding distribution
plt.figure(figsize=(10, 6))
sns.histplot(old_vals, bins=50, color='blue', alpha=0.5, stat='density', label='Old')
sns.histplot(new_vals, bins=50, color='red', alpha=0.5, stat='density', label='New')
plt.title("Distribution of Flattened Embeddings")
plt.xlabel("Embedding Value")
plt.ylabel("Density")
plt.legend()
plt.show()

# 2. PCA 2D Scatter Plot
all_data = np.vstack([old_embeddings, new_embeddings])
pca_2d = PCA(n_components=2).fit(all_data)
old_2d = pca_2d.transform(old_embeddings)
new_2d = pca_2d.transform(new_embeddings)

plt.figure(figsize=(8, 6))
plt.scatter(old_2d[:,0], old_2d[:,1], color='blue', alpha=0.7, label='Old Embeddings')
plt.scatter(new_2d[:,0], new_2d[:,1], color='red', alpha=0.7, label='New Embeddings')
plt.title("PCA Projection of Old vs New Embeddings")
plt.xlabel("PC 1")
plt.ylabel("PC 2")
plt.legend()
plt.show()

# 3. Bar plot of traditional metrics
metrics = {
    "Cosine Similarity": cosine_sim,
    "Euclidean Distance": euclidean_dist,
    "JSD": jsd_value,
    "Sym KL Div": kl_sym,
    "Wasserstein": w_distance,
    "Mahalanobis": mahalanobis_dist,
    "Cov Dist": cov_dist,
    "PCA Shift": pca_shift,
    "MMD Drift Score": mmd_driftscore
}

plt.figure(figsize=(10, 6))
sns.barplot(x=list(metrics.keys()), y=list(metrics.values()))
plt.xticks(rotation=45, ha='right')
plt.title("Drift Metrics Comparison (Including MMD)")
plt.ylabel("Metric Value")
plt.tight_layout()
plt.show()

# 4. UMAP Visualization
umap_reducer = umap.UMAP(n_components=2, random_state=42)
umap_embedding = umap_reducer.fit_transform(all_data)

old_umap = umap_embedding[:len(old_embeddings)]
new_umap = umap_embedding[len(old_embeddings):]

plt.figure(figsize=(8, 6))
plt.scatter(old_umap[:,0], old_umap[:,1], color='blue', alpha=0.7, label='Old Embeddings')
plt.scatter(new_umap[:,0], new_umap[:,1], color='red', alpha=0.7, label='New Embeddings')
plt.title("UMAP Projection of Old vs New Embeddings")
plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")
plt.legend()
plt.show()
