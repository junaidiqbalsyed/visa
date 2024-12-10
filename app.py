import os
import logging
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils import embedding_functions
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from time import time

# -------------------------
# Logger Setup
# -------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("drift_detection.log")]
)
logger = logging.getLogger(__name__)

# -------------------------
# 1. Custom PyTorch Dataset
# -------------------------
class MerchantDataset(Dataset):
    """
    PyTorch Dataset to read merchant data directly from a CSV file.
    """
    def __init__(self, file_path):
        self.file_path = file_path
        self.lines = self._load_data()

    def _load_data(self):
        with open(self.file_path, "r") as file:
            data = file.readlines()[1:]  # Skip header
        logger.info(f"Loaded {len(data)} rows from {self.file_path}.")
        return data

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        row = self.lines[idx].strip().split(",")
        mrch_nm_raw = row[0]  # Assuming the first column is 'mrch_nm_raw'
        metadata = {"row": row[1:]}  # Remaining columns as metadata
        return mrch_nm_raw, metadata

# -------------------------
# 2. SentenceTransformer Model Class
# -------------------------
class EmbeddingModel:
    def __init__(self, model_path="model/all-MiniLM-L6-v2"):
        """
        Load SentenceTransformer model from local path.
        """
        try:
            self.model = SentenceTransformer(model_path, device="cuda" if torch.cuda.is_available() else "cpu")
            logger.info(f"Embedding model loaded successfully from path: {model_path}")
        except Exception as e:
            logger.error(f"Error loading SentenceTransformer model: {str(e)}")
            raise

    def generate_embeddings(self, texts):
        """
        Generate embeddings using the SentenceTransformer model.
        """
        try:
            embeddings = self.model.encode(texts, batch_size=256, show_progress_bar=False, normalize_embeddings=True)
            return embeddings
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            raise

# -------------------------
# 3. Chroma DB Setup with Persistence
# -------------------------
def setup_chroma_db(embedding_name="v1"):
    save_path = f"./embeddings/{embedding_name}"
    os.makedirs(save_path, exist_ok=True)
    client = chromadb.PersistentClient(path=save_path)
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction()
    collection = client.get_or_create_collection(name="merchant_embeddings", embedding_function=embed_fn)
    logger.info(f"Chroma DB initialized and will be saved in {save_path}.")
    return collection

def insert_to_chroma_db(collection, embeddings, metadata_list):
    ids = [str(i) for i in range(len(metadata_list))]
    collection.add(embeddings=embeddings.tolist(), metadatas=metadata_list, ids=ids)
    logger.info(f"Inserted batch of {len(metadata_list)} embeddings into Chroma DB.")

# -------------------------
# 4. Batch Processing with Multi-Processing
# -------------------------
def process_batch(batch, embedding_model, chroma_collection):
    """
    Function to process each batch: generate embeddings and store in Chroma DB.
    """
    start_time = time()
    raw_names, metadata_list = zip(*batch)
    embeddings = embedding_model.generate_embeddings(list(raw_names))
    insert_to_chroma_db(chroma_collection, embeddings, metadata_list)
    logger.info(f"Processed batch of {len(batch)} in {time() - start_time:.2f} seconds.")

def process_data(file_path, embedding_model, chroma_collection, batch_size=1024, num_workers=4):
    """
    Process the entire dataset in batches with multiprocessing for embedding generation.
    """
    dataset = MerchantDataset(file_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    logger.info("Starting batch processing with multi-processing.")
    with ProcessPoolExecutor(max_workers=num_workers) as process_executor:
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Processing Batches", dynamic_ncols=True)):
            process_executor.submit(process_batch, batch, embedding_model, chroma_collection)

# -------------------------
# 5. Main Function
# -------------------------
def main():
    try:
        file_path = "path/to/your/data.csv"  # Replace with your CSV path
        embedding_name = "v1"  # Embedding version name

        # Initialize Embedding Model
        embedding_model = EmbeddingModel(model_path="model/all-MiniLM-L6-v2")

        # Setup Chroma DB with persistence
        chroma_collection = setup_chroma_db(embedding_name)

        # Process Data
        process_data(file_path, embedding_model, chroma_collection, batch_size=1024, num_workers=4)

        logger.info("All embeddings generated and stored successfully in Chroma DB.")
    except Exception as e:
        logger.critical(f"Critical error: {str(e)}")
        raise

if __name__ == "__main__":
    main()
