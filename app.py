import os
import logging
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import chromadb
from chromadb.utils import embedding_functions
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

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
# 1. Load Data using Pandas
# -------------------------
def load_data(file_path, chunksize=100000):
    """
    Load data in chunks to avoid memory issues.
    """
    try:
        logger.info(f"Loading data from {file_path} in chunks of {chunksize} rows.")
        for chunk in pd.read_csv(file_path, chunksize=chunksize):
            yield chunk
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

# -------------------------
# 2. Embedding Model Class
# -------------------------
class EmbeddingModel:
    def __init__(self, model_name="nvidia/NV-Embed-v2"):
        try:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name).to(self.device, dtype=self.dtype)
            logger.info("Embedding model loaded successfully in half precision.")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {str(e)}")
            raise

    def generate_embeddings(self, texts, batch_size=256):
        """
        Generate embeddings in batches for given texts.
        """
        embeddings = []
        try:
            for i in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings"):
                batch = texts[i:i + batch_size]
                inputs = self.tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    outputs = self.model(**inputs).last_hidden_state[:, 0, :]
                embeddings.extend(outputs.cpu().numpy())
            return embeddings
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            raise

# -------------------------
# 3. Chroma DB Setup
# -------------------------
def setup_chroma_db():
    try:
        client = chromadb.Client()
        embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction()
        collection = client.get_or_create_collection(name="merchant_embeddings", embedding_function=embed_fn)
        logger.info("Chroma DB initialized.")
        return collection
    except Exception as e:
        logger.error(f"Error setting up Chroma DB: {str(e)}")
        raise

def insert_to_chroma_db(collection, embeddings, metadata_list):
    try:
        ids = [str(i) for i in range(len(metadata_list))]
        collection.add(embeddings=embeddings, metadatas=metadata_list, ids=ids)
        logger.info(f"Batch of {len(metadata_list)} inserted into Chroma DB.")
    except Exception as e:
        logger.error(f"Error inserting into Chroma DB: {str(e)}")

# -------------------------
# 4. Multi-Processing for Embeddings
# -------------------------
def process_chunk(chunk, embedding_model, chroma_collection):
    try:
        logger.info(f"Processing chunk with {len(chunk)} rows.")
        texts = chunk['mrch_nm_raw'].tolist()
        embeddings = embedding_model.generate_embeddings(texts)
        metadata_list = chunk.to_dict(orient="records")
        insert_to_chroma_db(chroma_collection, embeddings, metadata_list)
        logger.info("Chunk processed and stored successfully.")
    except Exception as e:
        logger.error(f"Error processing chunk: {str(e)}")

# -------------------------
# 5. Process Data with Multi-Processing
# -------------------------
def process_data(file_path, embedding_model, chroma_collection, chunksize=100000, max_workers=4):
    try:
        with ProcessPoolExecutor(max_workers=max_workers) as process_executor:
            for idx, chunk in enumerate(load_data(file_path, chunksize)):
                logger.info(f"Submitting chunk {idx + 1} to process pool.")
                process_executor.submit(process_chunk, chunk, embedding_model, chroma_collection)
    except Exception as e:
        logger.error(f"Error processing data: {str(e)}")
        raise

# -------------------------
# 6. Main Function
# -------------------------
def main():
    try:
        # File Path
        file_path = "path/to/your/data.csv"  # Replace with actual file path

        # Load Embedding Model
        embedding_model = EmbeddingModel()

        # Setup Chroma DB
        chroma_collection = setup_chroma_db()

        # Process Data
        process_data(file_path, embedding_model, chroma_collection, chunksize=100000, max_workers=4)

        logger.info("All embeddings generated and stored successfully in Chroma DB.")
    except Exception as e:
        logger.critical(f"Critical error in execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()
