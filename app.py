# pip install pandas numpy sentence-transformers chromadb scikit-learn scipy matplotlib seaborn evidently ace_tools_open tqdm torch
# pip install pandas numpy sentence-transformers langchain  tiktoken huggingface-hub 


import pandas as pd
import math
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn.functional as F
from torch import tensor
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from sklearn.manifold import TSNE
from scipy.stats import ks_2samp
from torch.linalg import inv as torch_inv, svd as torch_svd

###############################################################################
# Device Setup
###############################################################################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

###############################################################################
# Sample Data (Replace with your own data loading code)
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

mid = len(df) // 2
old_df = df.iloc[:mid]
new_df = df.iloc[mid:]

old_texts = old_df["mrch_nm_raw"].tolist()
new_texts = new_df["mrch_nm_raw"].tolist()

###############################################################################
# Embeddings Setup (Consider a smaller model if GPU memory is tight)
###############################################################################
embedding_fn = HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2')

###############################################################################
# Initialize Chroma Vector Store
###############################################################################
vectordb = Chroma(
    collection_name="merchant_embeddings",
    persist_directory="chroma_db",
    embedding_function=embedding_fn
)

###############################################################################
# Batch Insertion with GPU Embeddings
###############################################################################
def add_texts_in_batches(vdb, texts, batch_size=2000, sub_batch_size=256):
    """
    Add documents to Chroma in GPU-accelerated batches.
    Adjust batch_size and sub_batch_size to fit GPU memory.
    """
    num_batches = math.ceil(len(texts) / batch_size)
    all_embeddings = []

    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = start_idx + batch_size
        batch_texts = texts[start_idx:end_idx]

        embeddings = []
        with tqdm(total=len(batch_texts), desc=f"Embedding batch {i+1}/{num_batches}", unit="doc") as pbar:
            for sb_start in range(0, len(batch_texts), sub_batch_size):
                sb_end = sb_start + sub_batch_size
                sub_batch_texts = batch_texts[sb_start:sb_end]
                # embed_documents returns a list of np arrays
                sub_embeddings = embedding_fn.embed_documents(sub_batch_texts)
                # Convert to torch and move to device
                sub_embeddings = torch.tensor(sub_embeddings, device=device, dtype=torch.float32)
                embeddings.append(sub_embeddings)
                pbar.update(len(sub_batch_texts))
        embeddings = torch.cat(embeddings, dim=0)

        # Move embeddings back to CPU for Chroma insertion
        # Chroma expects lists of embeddings on CPU
        embeddings_cpu = embeddings.cpu().numpy()
        with tqdm(total=1, desc=f"Adding batch {i+1}/{num_batches} to Chroma", unit="batch") as pbar_add:
            vdb.add_texts(texts=batch_texts, embeddings=embeddings_cpu)
            pbar_add.update(1)

        all_embeddings.append(embeddings)

    if all_embeddings:
        return torch.cat(all_embeddings, dim=0)
    else:
        # Empty case
        return torch.empty((0, embedding_fn.client.get_sentence_embedding_dimension()), device=device)

###############################################################################
# Add Old and New Data and Retrieve Embeddings
###############################################################################
old_embeddings = add_texts_in_batches(vectordb, old_texts, batch_size=2000, sub_batch_size=256)
new_embeddings = add_texts_in_batches(vectordb, new_texts, batch_size=2000, sub_batch_size=256)

vectordb.persist()
print("Chroma DB persistence done.")

###############################################################################
# Metric Computations on GPU with Torch
###############################################################################
# Compute centroids
old_centroid = old_embeddings.mean(dim=0)
new_centroid = new_embeddings.mean(dim=0)

# Cosine Similarity
cosine_sim = F.cosine_similarity(old_centroid, new_centroid, dim=0).item()

# Euclidean Distance
euclidean_dist = torch.norm(old_centroid - new_centroid).item()

# Histogram-based metrics: use torch.histc
all_vals = torch.cat([old_embeddings, new_embeddings], dim=0).flatten()
val_min = torch.min(all_vals)
val_max = torch.max(all_vals)

bins = 50
old_hist = torch.histc(old_embeddings.flatten(), bins=bins, min=val_min.item(), max=val_max.item())
new_hist = torch.histc(new_embeddings.flatten(), bins=bins, min=val_min.item(), max=val_max.item())

# Normalize hist to get probability distribution
old_hist = old_hist / old_hist.sum()
new_hist = new_hist / new_hist.sum()

def kl_div(p, q):
    eps = 1e-10
    p_ = p + eps
    q_ = q + eps
    return torch.sum(p_ * torch.log(p_ / q_))

def jsd(p, q):
    m = 0.5 * (p + q)
    return 0.5 * kl_div(p, m) + 0.5 * kl_div(q, m)

jsd_value = jsd(old_hist, new_hist).item()

kl_old_new = kl_div(old_hist, new_hist).item()
kl_new_old = kl_div(new_hist, old_hist).item()
kl_sym = 0.5*(kl_old_new + kl_new_old)

# Wasserstein Distance: 
# For wasserstein_distance, we can approximate using CPU scipy since it’s a one-liner.
# If we want GPU version, we must implement it. Let's just move back to CPU.
old_vals_cpu = old_embeddings.flatten().cpu().numpy()
new_vals_cpu = new_embeddings.flatten().cpu().numpy()
w_distance = wasserstein_distance(old_vals_cpu, new_vals_cpu)

# Mahalanobis Distance
# Compute covariance on GPU
# torch.cov requires at least PyTorch 1.9+
cov_old = torch.cov(old_embeddings.T)
inv_cov = torch_inv(cov_old + torch.eye(cov_old.shape[0], device=device)*1e-10)
diff = new_centroid - old_centroid
mahalanobis_dist = torch.sqrt(diff @ inv_cov @ diff).item()

# Covariance Distance (Frobenius norm on GPU)
cov_new = torch.cov(new_embeddings.T)
cov_dist = torch.norm(cov_old - cov_new, p='fro').item()

# PCA Shift using SVD on GPU
def pca_variance_ratio(x, n_components=2):
    x_centered = x - x.mean(dim=0)
    U,S,Vt = torch_svd(x_centered, full_matrices=False)
    explained_variance = (S**2)/(x.shape[0]-1)
    evr = explained_variance / explained_variance.sum()
    return evr[:n_components]

explained_variance_old = pca_variance_ratio(old_embeddings)
explained_variance_new = pca_variance_ratio(new_embeddings)
pca_shift = torch.norm(explained_variance_old - explained_variance_new).item()

# KS Test on CPU
ks_statistic, ks_pvalue = ks_2samp(old_vals_cpu, new_vals_cpu)

###############################################################################
# Evidently Report (CPU)
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
# Visualizations (CPU)
###############################################################################
import numpy as np
# Histogram of Old vs New
old_vals_cpu = old_vals_cpu  # already on CPU
new_vals_cpu = new_vals_cpu

plt.figure(figsize=(10, 6))
sns.histplot(old_vals_cpu, bins=50, color='blue', alpha=0.5, stat='density', label='Old')
sns.histplot(new_vals_cpu, bins=50, color='red', alpha=0.5, stat='density', label='New')
plt.title("Distribution of Flattened Embeddings")
plt.xlabel("Embedding Value")
plt.ylabel("Density")
plt.legend()
plt.show()

# t-SNE Visualization (CPU)
all_data_cpu = torch.cat([old_embeddings, new_embeddings], dim=0).cpu().numpy()
tsne = TSNE(n_components=2, random_state=42, perplexity=30, init='random', verbose=1)
tsne_embedding = tsne.fit_transform(all_data_cpu)
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

# Bar plot of Metrics
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


###########################################################################################################

import pandas as pd
import numpy as np
import math
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.spatial import distance
from scipy.stats import entropy, wasserstein_distance, ks_2samp
from numpy.linalg import inv
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

###############################################################################
# Data Setup
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

mid = len(df) // 2
old_df = df.iloc[:mid]
new_df = df.iloc[mid:]

old_texts = old_df["mrch_nm_raw"].tolist()
new_texts = new_df["mrch_nm_raw"].tolist()

###############################################################################
# Embeddings Setup
###############################################################################
embedding_fn = HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2')

###############################################################################
# Initialize Chroma Vector Store
###############################################################################
vectordb = Chroma(
    collection_name="merchant_embeddings",
    persist_directory="chroma_db",
    embedding_function=embedding_fn
)

###############################################################################
# Batch Insertion with Progress Bars using tqdm
###############################################################################
def add_texts_in_batches(vdb, texts, batch_size=20000, sub_batch_size=512):
    """
    Add documents and their embeddings to Chroma in batches and sub-batches.
    Uses tqdm progress bars to show progress during embedding and insertion.
    """
    num_batches = math.ceil(len(texts) / batch_size)
    all_embeddings = []

    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = start_idx + batch_size
        batch_texts = texts[start_idx:end_idx]

        embeddings = []
        # Embedding in sub-batches to avoid memory issues
        with tqdm(total=len(batch_texts), desc=f"Embedding batch {i+1}/{num_batches}", unit="doc") as pbar:
            for sb_start in range(0, len(batch_texts), sub_batch_size):
                sb_end = sb_start + sub_batch_size
                sub_batch_texts = batch_texts[sb_start:sb_end]
                sub_embeddings = embedding_fn.embed_documents(sub_batch_texts)
                embeddings.extend(sub_embeddings)
                pbar.update(len(sub_batch_texts))

        # Add this batch to Chroma
        with tqdm(total=1, desc=f"Adding batch {i+1}/{num_batches} to Chroma", unit="batch") as pbar_add:
            vdb.add_texts(texts=batch_texts, embeddings=embeddings)
            pbar_add.update(1)

        all_embeddings.append(np.array(embeddings))

    if all_embeddings:
        return np.vstack(all_embeddings)
    else:
        return np.empty((0, embedding_fn.client.get_sentence_embedding_dimension()))

###############################################################################
# Add Old and New Data and Retrieve Embeddings
###############################################################################
old_embeddings = add_texts_in_batches(vectordb, old_texts, batch_size=20000, sub_batch_size=512)
new_embeddings = add_texts_in_batches(vectordb, new_texts, batch_size=20000, sub_batch_size=512)

# Persist changes
vectordb.persist()
print("Chroma DB persistence done.")

###############################################################################
# Compute Drift Metrics
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

# Histogram of Old vs New
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
tsne = TSNE(n_components=2, random_state=42, perplexity=30, init='random', verbose=1)
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

# Bar plot of Metrics
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


#########################################################################################################################

# import pandas as pd
# import numpy as np
# import torch
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
# from tqdm import tqdm

# ###############################################################################
# # Load Data
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

# ###############################################################################
# # Model and Device Setup
# ###############################################################################
# model_name = "sentence-transformers/all-mpnet-base-v2"
# model = SentenceTransformer(model_name)

# # Move model to GPU if available
# device = "cuda" if torch.cuda.is_available() else "cpu"
# model.to(device)

# # If you need to enforce GPU memory limits (some methods):
# # Note: Not all environments allow this. Uncomment if supported.
# # torch.cuda.set_per_process_memory_fraction(0.9)  # Use 90% of available GPU memory

# ###############################################################################
# # Batch Embedding Function
# ###############################################################################
# def batch_encode(texts, model, batch_size=16):
#     """
#     Encode texts in batches with tqdm progress bar.
#     """
#     embeddings_list = []
#     for start_idx in tqdm(range(0, len(texts), batch_size), desc="Encoding Batches"):
#         end_idx = start_idx + batch_size
#         batch_texts = texts[start_idx:end_idx]
#         # Encode on the GPU (if available)
#         with torch.no_grad():
#             batch_embeddings = model.encode(batch_texts, convert_to_numpy=True, show_progress_bar=False, batch_size=batch_size)
#         embeddings_list.append(batch_embeddings)
#     return np.vstack(embeddings_list)

# ###############################################################################
# # Generate Embeddings with Batch Processing
# ###############################################################################
# old_texts = old_df["mrch_nm_raw"].tolist()
# new_texts = new_df["mrch_nm_raw"].tolist()

# # Adjust batch_size based on memory constraints
# # Smaller batch_size => less memory, but longer runtime
# old_embeddings = batch_encode(old_texts, model, batch_size=16)
# new_embeddings = batch_encode(new_texts, model, batch_size=16)

# ###############################################################################
# # Store Embeddings in Chroma DB
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


