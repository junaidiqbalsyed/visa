import os
import logging
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
import torch
from torch.utils.data import Dataset
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
# 1. Spark Initialization
# -------------------------
def init_spark(app_name="DataDriftEmbeddingSystem"):
    spark = SparkSession.builder \
        .appName(app_name) \
        .config("spark.executor.memory", "16g") \
        .config("spark.driver.memory", "16g") \
        .getOrCreate()
    logger.info("Spark session initialized.")
    return spark

# -------------------------
# 2. Load Data from Hadoop
# -------------------------
def load_data_from_hadoop(spark, hadoop_path):
    df = spark.read.format("parquet").load(hadoop_path)
    logger.info("Data loaded successfully from Hadoop.")
    return df.select("mrch_nm_raw", *[col(c) for c in df.columns if c != "mrch_nm_raw"])

# -------------------------
# 3. Embedding Model Class
# -------------------------
class EmbeddingModel:
    def __init__(self, model_name="nvidia/NV-Embed-v2"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device, dtype=self.dtype)
        logger.info("Embedding model loaded successfully.")

    def generate_embeddings(self, texts):
        try:
            inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs).last_hidden_state[:, 0, :]
            return outputs.cpu().numpy()
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            return None

# -------------------------
# 4. Chroma DB Setup
# -------------------------
def setup_chroma_db():
    client = chromadb.Client()
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction()
    collection = client.get_or_create_collection(name="merchant_embeddings", embedding_function=embed_fn)
    logger.info("Chroma DB initialized.")
    return collection

def insert_to_chroma_db(collection, embeddings, metadata_list):
    try:
        ids = [str(i) for i in range(len(metadata_list))]
        collection.add(embeddings=embeddings, metadatas=metadata_list, ids=ids)
        logger.info("Batch inserted into Chroma DB successfully.")
    except Exception as e:
        logger.error(f"Error inserting into Chroma DB: {str(e)}")

# -------------------------
# 5. Multi-Processing for Embeddings
# -------------------------
def process_batch(batch, embedding_model):
    texts = [row["mrch_nm_raw"] for row in batch]
    embeddings = embedding_model.generate_embeddings(texts)
    return embeddings, batch

def process_data_in_batches(df, embedding_model, chroma_collection, batch_size=10000, max_workers=4):
    df_batches = df.rdd.mapPartitions(lambda x: [list(x)]).toLocalIterator()
    with ProcessPoolExecutor(max_workers=max_workers) as process_executor:
        for idx, batch in enumerate(df_batches):
            logger.info(f"Processing batch {idx + 1}")
            embeddings, batch_data = process_executor.submit(process_batch, batch, embedding_model).result()

            if embeddings is not None:
                metadata_list = [row.asDict() for row in batch_data]
                # Use threading for Chroma DB insertion
                with ThreadPoolExecutor(max_workers=2) as thread_executor:
                    thread_executor.submit(insert_to_chroma_db, chroma_collection, embeddings.tolist(), metadata_list)

# -------------------------
# 6. Main Function
# -------------------------
def main():
    try:
        # Initialize Spark
        spark = init_spark()

        # Load Data
        hadoop_path = "hdfs://path/to/hadoop/data"  # Replace with actual path
        df = load_data_from_hadoop(spark, hadoop_path)

        # Initialize Embedding Model
        embedding_model = EmbeddingModel()

        # Setup Chroma DB
        chroma_collection = setup_chroma_db()

        # Process Data in Batches with Parallelization
        process_data_in_batches(df, embedding_model, chroma_collection, batch_size=10000, max_workers=4)

        logger.info("All embeddings generated and stored successfully.")
    except Exception as e:
        logger.critical(f"Critical error in execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()
