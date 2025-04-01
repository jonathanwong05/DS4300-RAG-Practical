import os
import fitz  # PyMuPDF
import numpy as np
import faiss
import pickle
from sentence_transformers import SentenceTransformer
import time
import psutil
import re


class FAISSIngestor:
    def __init__(self, data_dir="../data/"):
        self.data_dir = data_dir
        self.embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.index = None
        self.metadata = []
        self.embedding_dim = 384  # Match MiniLM-L6 dimension

    def clean_text(self, text):
        """Basic text cleaning"""
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def extract_text_from_pdf(self, pdf_path):
        """Extract text with page numbers"""
        doc = fitz.open(pdf_path)
        text_by_page = []
        for page_num, page in enumerate(doc):
            text = page.get_text()
            if text.strip():
                text_by_page.append((page_num, text))
        return text_by_page

    def split_text_into_chunks(self, text, chunk_size=500, overlap=150):
        """Split text into overlapping chunks"""
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i: i + chunk_size])
            chunks.append(chunk)
        return chunks

    def create_faiss_index(self):
        """Initialize FAISS index"""
        self.index = faiss.IndexFlatIP(self.embedding_dim)  # Using Inner Product for cosine similarity

    def process_pdfs(self):
        """Main processing pipeline"""
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
                    for page_num, text in text_by_page:
                        cleaned_text = self.clean_text(text)
                        chunks = self.split_text_into_chunks(cleaned_text)

                        if chunks:
                            # Batch encode chunks
                            chunk_embeddings = self.embedding_model.encode(chunks)

                            # Store metadata for each chunk
                            for i, chunk in enumerate(chunks):
                                self.metadata.append({
                                    "file": file_name,
                                    "page": page_num,
                                    "chunk": chunk,
                                    "chunk_idx": i
                                })

                            all_embeddings.append(chunk_embeddings)
                            total_chunks += len(chunks)

                    total_pages += len(text_by_page)
                    print(f"Processed {file_name} ({len(text_by_page)} pages, {len(chunks)} chunks)")

                except Exception as e:
                    print(f"Error processing {file_name}: {str(e)}")

        # Combine all embeddings and add to index
        if all_embeddings:
            all_embeddings = np.vstack(all_embeddings).astype('float32')

            # Normalize embeddings before adding to index
            faiss.normalize_L2(all_embeddings)
            self.index.add(all_embeddings)

            # Save index and metadata
            faiss.write_index(self.index, "faiss_index.index")
            with open("faiss_metadata.pkl", "wb") as f:
                pickle.dump(self.metadata, f)

        mem_after = process.memory_info().rss / (1024 * 1024)
        total_time = time.time() - start_time

        print("\n" + "=" * 50)
        print("INGESTION SUMMARY")
        print("=" * 50)
        print(f"Total files processed: {total_files}")
        print(f"Total pages processed: {total_pages}")
        print(f"Total chunks generated: {total_chunks}")
        print(f"Total ingestion time: {total_time:.2f} seconds")
        print(f"Memory usage before: {mem_before:.2f} MB")
        print(f"Memory usage after: {mem_after:.2f} MB")
        print(f"FAISS index size: {self.index.ntotal if self.index else 0} vectors")
        print("=" * 50)


if __name__ == "__main__":
    ingestor = FAISSIngestor()
    ingestor.process_pdfs()