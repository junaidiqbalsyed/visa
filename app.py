# pip install pandas numpy sentence-transformers chromadb scikit-learn scipy matplotlib seaborn evidently ace_tools_open tqdm torch


import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from chromadb import Client
from chromadb.config import Settings
from sklearn.decomposition import PCA
from scipy.spatial import distance
from scipy.stats import entropy, wasserstein_distance, ks_2samp
from numpy.linalg import inv
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from tqdm import tqdm

###############################################################################
# Load Data
###############################################################################
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

mid = len(df)//2
old_df = df.iloc[:mid]
new_df = df.iloc[mid:]

###############################################################################
# Model and Device Setup
###############################################################################
model_name = "sentence-transformers/all-mpnet-base-v2"
model = SentenceTransformer(model_name)

# Move model to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# If you need to enforce GPU memory limits (some methods):
# Note: Not all environments allow this. Uncomment if supported.
# torch.cuda.set_per_process_memory_fraction(0.9)  # Use 90% of available GPU memory

###############################################################################
# Batch Embedding Function
###############################################################################
def batch_encode(texts, model, batch_size=16):
    """
    Encode texts in batches with tqdm progress bar.
    """
    embeddings_list = []
    for start_idx in tqdm(range(0, len(texts), batch_size), desc="Encoding Batches"):
        end_idx = start_idx + batch_size
        batch_texts = texts[start_idx:end_idx]
        # Encode on the GPU (if available)
        with torch.no_grad():
            batch_embeddings = model.encode(batch_texts, convert_to_numpy=True, show_progress_bar=False, batch_size=batch_size)
        embeddings_list.append(batch_embeddings)
    return np.vstack(embeddings_list)

###############################################################################
# Generate Embeddings with Batch Processing
###############################################################################
old_texts = old_df["mrch_nm_raw"].tolist()
new_texts = new_df["mrch_nm_raw"].tolist()

# Adjust batch_size based on memory constraints
# Smaller batch_size => less memory, but longer runtime
old_embeddings = batch_encode(old_texts, model, batch_size=16)
new_embeddings = batch_encode(new_texts, model, batch_size=16)

###############################################################################
# Store Embeddings in Chroma DB
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
# Traditional Metrics
###############################################################################
old_centroid = np.mean(old_embeddings, axis=0)
new_centroid = np.mean(new_embeddings, axis=0)

cosine_sim = 1 - distance.cosine(old_centroid, new_centroid)
euclidean_dist = np.linalg.norm(old_centroid - new_centroid)

def jensen_shannon_divergence(p, q):
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5*(p+q)
    return 0.5*entropy(p, m) + 0.5*entropy(q, m)

old_hist, bins = np.histogram(old_embeddings.flatten(), bins=50, density=True)
new_hist, _ = np.histogram(new_embeddings.flatten(), bins=bins, density=True)
jsd_value = jensen_shannon_divergence(old_hist, new_hist)

def safe_kl_div(p, q):
    p = p / p.sum()
    q = q / q.sum()
    eps = 1e-10
    return np.sum(p * np.log((p+eps)/(q+eps)))

kl_old_new = safe_kl_div(old_hist, new_hist)
kl_new_old = safe_kl_div(new_hist, old_hist)
kl_sym = 0.5*(kl_old_new + kl_new_old)

old_vals = old_embeddings.flatten()
new_vals = new_embeddings.flatten()
w_distance = wasserstein_distance(old_vals, new_vals)

cov = np.cov(old_embeddings, rowvar=False)
inv_cov = inv(cov + np.eye(cov.shape[0])*1e-10)
diff = new_centroid - old_centroid
mahalanobis_dist = np.sqrt(diff.T.dot(inv_cov).dot(diff))

new_cov = np.cov(new_embeddings, rowvar=False)
cov_dist = np.linalg.norm(cov - new_cov, ord='fro')

explained_variance_old = PCA(n_components=2).fit(old_embeddings).explained_variance_ratio_
explained_variance_new = PCA(n_components=2).fit(new_embeddings).explained_variance_ratio_
pca_shift = np.linalg.norm(explained_variance_old - explained_variance_new)

ks_statistic, ks_pvalue = ks_2samp(old_vals, new_vals)

###############################################################################
# Evidently Report
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
# Visualizations
###############################################################################
# Histogram
plt.figure(figsize=(10, 6))
sns.histplot(old_vals, bins=50, color='blue', alpha=0.5, stat='density', label='Old')
sns.histplot(new_vals, bins=50, color='red', alpha=0.5, stat='density', label='New')
plt.title("Distribution of Flattened Embeddings")
plt.xlabel("Embedding Value")
plt.ylabel("Density")
plt.legend()
plt.show()

# t-SNE Visualization
all_data = np.vstack([old_embeddings, new_embeddings])
tsne = TSNE(n_components=2, random_state=42, perplexity=30, init='random')
tsne_embedding = tsne.fit_transform(all_data)
old_tsne = tsne_embedding[:len(old_embeddings)]
new_tsne = tsne_embedding[len(old_embeddings):]

plt.figure(figsize=(8, 6))
plt.scatter(old_tsne[:,0], old_tsne[:,1], color='blue', alpha=0.7, label='Old Embeddings')
plt.scatter(new_tsne[:,0], new_tsne[:,1], color='red', alpha=0.7, label='New Embeddings')
plt.title("t-SNE Projection of Old vs New Embeddings")
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.legend()
plt.show()

# Bar plot of metrics
metrics = {
    "Cosine Similarity": cosine_sim,
    "Euclidean Distance": euclidean_dist,
    "JSD": jsd_value,
    "Sym KL Div": kl_sym,
    "Wasserstein": w_distance,
    "Mahalanobis": mahalanobis_dist,
    "Cov Dist": cov_dist,
    "PCA Shift": pca_shift,
    "KS Statistic": ks_statistic,
    "KS p-value": ks_pvalue
}

plt.figure(figsize=(10, 6))
sns.barplot(x=list(metrics.keys()), y=list(metrics.values()))
plt.xticks(rotation=45, ha='right')
plt.title("Drift Metrics Comparison")
plt.ylabel("Metric Value")
plt.tight_layout()
plt.show()

##############################################################################################################################################################################################

# import pandas as pd
# import numpy as np
# from sentence_transformers import SentenceTransformer
# from chromadb import Client
# from chromadb.config import Settings
# from sklearn.decomposition import PCA
# from scipy.spatial import distance
# from scipy.stats import entropy, wasserstein_distance, ks_2samp
# from numpy.linalg import inv
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.manifold import TSNE
# from evidently.report import Report
# from evidently.metric_preset import DataDriftPreset

# ###############################################################################
# # Load or create dataset
# ###############################################################################
# df = pd.DataFrame({
#     "mrch_nm_raw": [
#         "SAFEWAY #2841",
#         "SHELL OIL 1000492406",
#         "SUNOCO 091017800",
#         "POPEYES 11562",
#         "TEXAS ROADHOUSE FR #7005",
#         "HARBOR FREIGHT TOOLS 233",
#         "COSTCO WHOLESALE",
#         "WALMART SUPERCENTER 1234",
#         "TARGET T-123",
#         "AMAZON MKTPLACE PMTS",
#         "MCDONALD'S F23987",
#         "BURGER KING #0081"
#     ]
# })

# mid = len(df)//2
# old_df = df.iloc[:mid]
# new_df = df.iloc[mid:]

# model_name = "sentence-transformers/all-mpnet-base-v2"
# model = SentenceTransformer(model_name)

# old_embeddings = model.encode(old_df["mrch_nm_raw"].tolist(), convert_to_numpy=True)
# new_embeddings = model.encode(new_df["mrch_nm_raw"].tolist(), convert_to_numpy=True)

# ###############################################################################
# # Chroma DB storage
# ###############################################################################
# client = Client(
#     Settings(
#         chroma_db_impl="duckdb+parquet",
#         persist_directory="chroma_db"
#     )
# )

# collection = client.get_or_create_collection(name="merchant_embeddings")
# collection.add(
#     documents=old_df["mrch_nm_raw"].tolist(),
#     embeddings=old_embeddings.tolist(),
#     ids=[f"old_{i}" for i in range(len(old_df))]
# )
# collection.add(
#     documents=new_df["mrch_nm_raw"].tolist(),
#     embeddings=new_embeddings.tolist(),
#     ids=[f"new_{i}" for i in range(len(new_df))]
# )

# ###############################################################################
# # Traditional Metrics
# ###############################################################################
# old_centroid = np.mean(old_embeddings, axis=0)
# new_centroid = np.mean(new_embeddings, axis=0)

# cosine_sim = 1 - distance.cosine(old_centroid, new_centroid)
# euclidean_dist = np.linalg.norm(old_centroid - new_centroid)

# def jensen_shannon_divergence(p, q):
#     p = p / p.sum()
#     q = q / q.sum()
#     m = 0.5*(p+q)
#     return 0.5*entropy(p, m) + 0.5*entropy(q, m)

# old_hist, bins = np.histogram(old_embeddings.flatten(), bins=50, density=True)
# new_hist, _ = np.histogram(new_embeddings.flatten(), bins=bins, density=True)
# jsd_value = jensen_shannon_divergence(old_hist, new_hist)

# def safe_kl_div(p, q):
#     p = p / p.sum()
#     q = q / q.sum()
#     eps = 1e-10
#     return np.sum(p * np.log((p+eps)/(q+eps)))

# kl_old_new = safe_kl_div(old_hist, new_hist)
# kl_new_old = safe_kl_div(new_hist, old_hist)
# kl_sym = 0.5*(kl_old_new + kl_new_old)

# old_vals = old_embeddings.flatten()
# new_vals = new_embeddings.flatten()
# w_distance = wasserstein_distance(old_vals, new_vals)

# cov = np.cov(old_embeddings, rowvar=False)
# inv_cov = inv(cov + np.eye(cov.shape[0])*1e-10)
# diff = new_centroid - old_centroid
# mahalanobis_dist = np.sqrt(diff.T.dot(inv_cov).dot(diff))

# new_cov = np.cov(new_embeddings, rowvar=False)
# cov_dist = np.linalg.norm(cov - new_cov, ord='fro')

# explained_variance_old = PCA(n_components=2).fit(old_embeddings).explained_variance_ratio_
# explained_variance_new = PCA(n_components=2).fit(new_embeddings).explained_variance_ratio_
# pca_shift = np.linalg.norm(explained_variance_old - explained_variance_new)

# # Kolmogorov-Smirnov Test as a robust statistical test of distribution difference
# ks_statistic, ks_pvalue = ks_2samp(old_vals, new_vals)

# ###############################################################################
# # Evidently Report
# ###############################################################################
# old_df_evidently = old_df.copy()
# new_df_evidently = new_df.copy()
# old_df_evidently["mrch_length"] = old_df_evidently["mrch_nm_raw"].apply(len)
# new_df_evidently["mrch_length"] = new_df_evidently["mrch_nm_raw"].apply(len)

# report = Report(metrics=[DataDriftPreset()])
# report.run(reference_data=old_df_evidently, current_data=new_df_evidently)
# report.save_html("data_drift_report.html")
# print("Evidently data drift report saved as data_drift_report.html")

# ###############################################################################
# # Visualizations
# ###############################################################################

# # Histogram
# plt.figure(figsize=(10, 6))
# sns.histplot(old_vals, bins=50, color='blue', alpha=0.5, stat='density', label='Old')
# sns.histplot(new_vals, bins=50, color='red', alpha=0.5, stat='density', label='New')
# plt.title("Distribution of Flattened Embeddings")
# plt.xlabel("Embedding Value")
# plt.ylabel("Density")
# plt.legend()
# plt.show()

# # t-SNE Visualization
# all_data = np.vstack([old_embeddings, new_embeddings])
# tsne = TSNE(n_components=2, random_state=42, perplexity=30, init='random')
# tsne_embedding = tsne.fit_transform(all_data)
# old_tsne = tsne_embedding[:len(old_embeddings)]
# new_tsne = tsne_embedding[len(old_embeddings):]

# plt.figure(figsize=(8, 6))
# plt.scatter(old_tsne[:,0], old_tsne[:,1], color='blue', alpha=0.7, label='Old Embeddings')
# plt.scatter(new_tsne[:,0], new_tsne[:,1], color='red', alpha=0.7, label='New Embeddings')
# plt.title("t-SNE Projection of Old vs New Embeddings")
# plt.xlabel("t-SNE 1")
# plt.ylabel("t-SNE 2")
# plt.legend()
# plt.show()

# # Bar plot of metrics
# metrics = {
#     "Cosine Similarity": cosine_sim,
#     "Euclidean Distance": euclidean_dist,
#     "JSD": jsd_value,
#     "Sym KL Div": kl_sym,
#     "Wasserstein": w_distance,
#     "Mahalanobis": mahalanobis_dist,
#     "Cov Dist": cov_dist,
#     "PCA Shift": pca_shift,
#     "KS Statistic": ks_statistic,
#     "KS p-value": ks_pvalue
# }

# plt.figure(figsize=(10, 6))
# sns.barplot(x=list(metrics.keys()), y=list(metrics.values()))
# plt.xticks(rotation=45, ha='right')
# plt.title("Drift Metrics Comparison")
# plt.ylabel("Metric Value")
# plt.tight_layout()
# plt.show()


