import ollama
import redis
import numpy as np
from redis.commands.search.query import Query
import os
import fitz
import time
import psutil
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class RedisIngestor:
    def __init__(self, data_dir="../data/"):
        self.data_dir = data_dir
        self.redis_client = redis.Redis(host="localhost", port=6379, db=0)
        self.embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.similarities = []
        self.VECTOR_DIM = 384
        self.INDEX_NAME = "embedding_index"
        self.DOC_PREFIX = "doc:"

    def clean_text(self, text):
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def extract_text_from_pdf(self, pdf_path):
        doc = fitz.open(pdf_path)
        text_by_page = []
        for page_num, page in enumerate(doc):
            text = page.get_text()
            if text.strip():
                text_by_page.append((page_num, text))
        return text_by_page

    def split_text_into_chunks(self, text, chunk_size=500, overlap=150):
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i: i + chunk_size])
            chunks.append(chunk)
        return chunks

    def calculate_similarity(self, embeddings):
        if len(embeddings) < 2:
            return []
        sim_matrix = cosine_similarity(embeddings)
        np.fill_diagonal(sim_matrix, 0)  # Ignore self-similarity
        return sim_matrix.max(axis=1)  # Get max similarity for each chunk

    def clear_redis_store(self):
        print("Clearing existing Redis store...")
        self.redis_client.flushdb()
        print("Redis store cleared.")

    def create_hnsw_index(self):
        try:
            self.redis_client.execute_command(f"FT.DROPINDEX {self.INDEX_NAME} DD")
        except redis.exceptions.ResponseError:
            pass

        self.redis_client.execute_command(
            f"""
            FT.CREATE {self.INDEX_NAME} ON HASH PREFIX 1 {self.DOC_PREFIX}
            SCHEMA text TEXT
            embedding VECTOR HNSW 6 DIM {self.VECTOR_DIM} TYPE FLOAT32 DISTANCE_METRIC COSINE
            """
        )
        print("Index created successfully.")

    def store_embeddings(self, file_name, page_num, chunks):
        if not chunks:
            return

        embeddings = self.embedding_model.encode(chunks)
        chunk_similarities = self.calculate_similarity(embeddings)
        self.similarities.extend(chunk_similarities)

        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            key = f"{self.DOC_PREFIX}:{file_name}_page_{page_num}_chunk_{i}"
            self.redis_client.hset(
                key,
                mapping={
                    "file": file_name,
                    "page": str(page_num),
                    "chunk": chunk,
                    "embedding": np.array(embedding, dtype=np.float32).tobytes(),
                    "similarity": str(chunk_similarities[i]) if i < len(chunk_similarities) else "0.0"
                }
            )

    def process_pdfs(self):
        total_files = 0
        total_pages = 0
        total_chunks = 0
        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss / (1024 * 1024)
        start_time = time.time()

        self.clear_redis_store()
        self.create_hnsw_index()

        for file_name in os.listdir(self.data_dir):
            if file_name.endswith(".pdf"):
                total_files += 1
                file_start = time.time()
                pdf_path = os.path.join(self.data_dir, file_name)

                try:
                    text_by_page = self.extract_text_from_pdf(pdf_path)
                    file_page_count = len(text_by_page)
                    file_chunk_count = 0

                    for page_num, text in text_by_page:
                        cleaned_text = self.clean_text(text)
                        chunks = self.split_text_into_chunks(cleaned_text)
                        if chunks:
                            self.store_embeddings(file_name, page_num, chunks)
                            file_chunk_count += len(chunks)

                    file_end = time.time()
                    file_time = file_end - file_start
                    print(
                        f" -----> Processed {file_name}: {file_page_count} pages, {file_chunk_count} chunks in {file_time:.2f} seconds")
                    total_pages += file_page_count
                    total_chunks += file_chunk_count

                except Exception as e:
                    print(f"Error processing {file_name}: {str(e)}")

        mem_after = process.memory_info().rss / (1024 * 1024)
        ingestion_time = time.time() - start_time

        # Calculate similarity statistics
        avg_similarity = np.mean(self.similarities) if self.similarities else 0
        max_similarity = np.max(self.similarities) if self.similarities else 0
        min_similarity = np.min(self.similarities) if self.similarities else 0

        print("\n" + "=" * 50)
        print("INGESTION SUMMARY")
        print("=" * 50)
        print(f"Total files processed  : {total_files}")
        print(f"Total pages processed  : {total_pages}")
        print(f"Total chunks generated : {total_chunks}")
        print(f"Total ingestion time   : {ingestion_time:.2f} seconds")
        print(f"Memory usage before    : {mem_before:.2f} MB")
        print(f"Memory usage after     : {mem_after:.2f} MB")
        print(f"Average chunk similarity: {avg_similarity:.4f}")
        print(f"Maximum chunk similarity: {max_similarity:.4f}")
        print(f"Minimum chunk similarity: {min_similarity:.4f}")
        print("=" * 50 + "\n")


if __name__ == "__main__":
    ingestor = RedisIngestor()
    ingestor.process_pdfs()