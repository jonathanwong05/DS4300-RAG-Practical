import ollama
import redis
import numpy as np
from redis.commands.search.query import Query
import os
import fitz
import time
import psutil
import re

# Initialize Redis connection
redis_client = redis.Redis(host="localhost", port=6380, db=0)

VECTOR_DIM = 768
INDEX_NAME = "embedding_index"
DOC_PREFIX = "doc:"
DISTANCE_METRIC = "COSINE"


# Preprocessing function
def clean_text(text, remove_punctuation=False, remove_whitespace=False):
    if remove_punctuation:
        text = re.sub(r'[^\w\s]', '', text)
    if remove_whitespace:
        text = ' '.join(text.split())
    return text

# used to clear the redis vector store
def clear_redis_store():
    print("Clearing existing Redis store...")
    redis_client.flushdb()
    print("Redis store cleared.")


# Create an HNSW index in Redis
def create_hnsw_index():
    try:
        redis_client.execute_command(f"FT.DROPINDEX {INDEX_NAME} DD")
    except redis.exceptions.ResponseError:
        pass

    redis_client.execute_command(
        f"""
        FT.CREATE {INDEX_NAME} ON HASH PREFIX 1 {DOC_PREFIX}
        SCHEMA text TEXT
        embedding VECTOR HNSW 6 DIM {VECTOR_DIM} TYPE FLOAT32 DISTANCE_METRIC {DISTANCE_METRIC}
        """
    )
    print("Index created successfully.")


# Generate an embedding using nomic-embed-text
def get_embedding(text: str, model: str = "nomic-embed-text") -> list:

    response = ollama.embeddings(model=model, prompt=text)
    return response["embedding"]


# store the embedding in Redis
def store_embedding(file: str, page: str, chunk: str, embedding: list):
    
    max_length = 50
    shortened_chunk = (chunk[:max_length] + '...') if len(chunk) > max_length else chunk

    key = f"{DOC_PREFIX}:{file}_page_{page}_chunk_{shortened_chunk}"
    redis_client.hset(
        key,
        mapping={
            "file": file,
            "page": page,
            "chunk": chunk,
            "embedding": np.array(
                embedding, dtype=np.float32
            ).tobytes(),  # Store as byte array
        },
    )
    print(f"Stored embedding for: {shortened_chunk}")


# extract the text from a PDF by page
def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    doc = fitz.open(pdf_path)
    text_by_page = []
    for page_num, page in enumerate(doc):
        text_by_page.append((page_num, page.get_text()))
    return text_by_page


# split the text into chunks with overlap
def split_text_into_chunks(text, chunk_size=500, overlap=50):
    """Split text into chunks of approximately chunk_size words with overlap."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i : i + chunk_size])
        chunks.append(chunk)
    return chunks


# Process all PDF files in a given directory
def process_pdfs(data_dir):

    total_files = 0
    total_pages = 0
    total_chunks = 0

    for file_name in os.listdir(data_dir):
        if file_name.endswith(".pdf"):
            total_files += 1
            file_start = time.time()

            pdf_path = os.path.join(data_dir, file_name)
            text_by_page = extract_text_from_pdf(pdf_path)
            file_page_count = len(text_by_page)
            file_chunk_count = 0

            for page_num, text in text_by_page:
                cleaned_text = clean_text(text, remove_punctuation=False, remove_whitespace=False)
                chunks = split_text_into_chunks(cleaned_text)
                file_chunk_count += len(chunks)
                for chunk_index, chunk in enumerate(chunks):
                    embedding = get_embedding(chunk)
                    store_embedding(
                        file=file_name,
                        page=str(page_num),
                        # chunk=str(chunk_index),
                        chunk=str(chunk),
                        embedding=embedding,
                    )
            file_end = time.time()
            file_time = file_end - file_start

            print(f" -----> Processed {file_name}: {file_page_count} pages, {file_chunk_count} chunks in {file_time:.2f} seconds")
            total_pages += file_page_count
            total_chunks += file_chunk_count

    return total_files, total_pages, total_chunks

def query_redis(query_text: str):
    q = (
        Query("*=>[KNN 5 @embedding $vec AS vector_distance]")
        .sort_by("vector_distance")
        .return_fields("id", "vector_distance")
        .dialect(2)
    )
    query_text = "Efficient search in vector databases"
    embedding = get_embedding(query_text)

    query_start = time.time()
    res = redis_client.ft(INDEX_NAME).search(
        q, query_params={"vec": np.array(embedding, dtype=np.float32).tobytes()}
    )
    query_end = time.time()
    query_time = query_end - query_start

    similarities = []
    print("\n" + "=" * 50)
    print("QUERY SUMMARY")
    print("=" * 50)
    print(f"Query: {query_text}")
    print(f"Query execution time: {query_time:.2f} seconds")
    print("Results:")
    for doc in res.docs:
        print(f"Document ID: {doc.id} | Similarity: {doc.vector_distance}")
    print("=" * 50 + "\n")


def main():
    clear_redis_store()
    create_hnsw_index()

    # Log memory before ingestion
    process_obj = psutil.Process(os.getpid())
    mem_before = process_obj.memory_info().rss / (1024 * 1024)

    ingestion_start = time.time()
    total_files, total_pages, total_chunks = process_pdfs("../data/")
    print("\n---Done processing PDFs---\n")
    ingestion_end = time.time()
    ingestion_time = ingestion_end - ingestion_start

    # Log memory after ingestion
    mem_after = process_obj.memory_info().rss / (1024 * 1024)

    print("\n" + "=" * 50)
    print("INGESTION SUMMARY")
    print("=" * 50)
    print(f"Total files processed  : {total_files}")
    print(f"Total pages processed  : {total_pages}")
    print(f"Total chunks generated : {total_chunks}")
    print(f"Total ingestion time   : {ingestion_time:.2f} seconds")
    print(f"Memory usage before    : {mem_before:.2f} MB")
    print(f"Memory usage after     : {mem_after:.2f} MB")
    print("=" * 50 + "\n")

    query_redis("What is the capital of France?")


if __name__ == "__main__":
    main()
