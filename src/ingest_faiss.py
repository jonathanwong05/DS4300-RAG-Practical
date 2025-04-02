import os
import fitz  # PyMuPDF
import numpy as np
import faiss
import pickle
from sentence_transformers import SentenceTransformer
import time
import psutil
import re
from sklearn.metrics.pairwise import cosine_similarity


class FAISSIngestor:
    def __init__(self, data_dir="../data/"):
        self.data_dir = data_dir
        self.embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.index = None
        self.metadata = []
        self.embedding_dim = 384
        self.similarities = []

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
        """Calculate pairwise cosine similarities between chunks"""
        if len(embeddings) < 2:
            return []
        sim_matrix = cosine_similarity(embeddings)
        np.fill_diagonal(sim_matrix, 0)  # Ignore self-similarity
        return sim_matrix.max(axis=1)  # Get max similarity for each chunk

    def create_faiss_index(self):
        self.index = faiss.IndexFlatIP(self.embedding_dim)

    def process_pdfs(self):
        total_files = 0
        total_pages = 0
        total_chunks = 0
        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss / (1024 * 1024)
        start_time = time.time()

        self.create_faiss_index()
        all_embeddings = []

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
                            chunk_embeddings = self.embedding_model.encode(chunks)

                            # Calculate and store similarities
                            chunk_similarities = self.calculate_similarity(chunk_embeddings)
                            self.similarities.extend(chunk_similarities)

                            all_embeddings.append(chunk_embeddings)

                            for i, chunk in enumerate(chunks):
                                self.metadata.append({
                                    "file": file_name,
                                    "page": page_num,
                                    "chunk": chunk,
                                    "similarity": float(chunk_similarities[i]) if i < len(chunk_similarities) else 0.0
                                })

                            file_chunk_count += len(chunks)

                    file_end = time.time()
                    file_time = file_end - file_start
                    print(
                        f" -----> Processed {file_name}: {file_page_count} pages, {file_chunk_count} chunks in {file_time:.2f} seconds")
                    total_pages += file_page_count
                    total_chunks += file_chunk_count

                except Exception as e:
                    print(f"Error processing {file_name}: {str(e)}")

        if all_embeddings:
            all_embeddings = np.vstack(all_embeddings).astype('float32')
            faiss.normalize_L2(all_embeddings)
            self.index.add(all_embeddings)

            faiss.write_index(self.index, "faiss_index.index")
            with open("faiss_metadata.pkl", "wb") as f:
                pickle.dump(self.metadata, f)

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
    ingestor = FAISSIngestor()
    ingestor.process_pdfs()