import os
import time
import fitz  # PyMuPDF
import re
import chromadb
from sentence_transformers import SentenceTransformer
import psutil


class ChromaDBIngestor:
    def __init__(self, data_dir="../data/", db_path="./chroma_db"):
        self.data_dir = data_dir
        self.chroma_client = chromadb.PersistentClient(path=db_path)
        self.collection = self.chroma_client.get_or_create_collection(
            name="pdf_embeddings",
            metadata={"hnsw:space": "cosine"}
        )
        self.embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    def clean_text(self, text):
        # Basic cleaning - preserve original structure
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def extract_text_from_pdf(self, pdf_path):
        doc = fitz.open(pdf_path)
        text_by_page = []
        for page_num, page in enumerate(doc):
            text = page.get_text()
            if text.strip():  # Only add non-empty pages
                text_by_page.append((page_num, text))
        return text_by_page

    def split_text_into_chunks(self, text, chunk_size=500, overlap=150):
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i: i + chunk_size])
            chunks.append(chunk)
        return chunks

    def store_embeddings(self, file_name, page_num, chunks):
        if not chunks:
            return

        embeddings = self.embedding_model.encode(chunks)
        documents = chunks  # Store the actual text as documents
        ids = [f"{file_name}_p{page_num}_c{i}" for i in range(len(chunks))]
        metadatas = [{
            "file": file_name,
            "page": str(page_num),
            "chunk_idx": str(i)
        } for i in range(len(chunks))]

        # Store both documents and embeddings
        self.collection.add(
            ids=ids,
            embeddings=embeddings.tolist(),
            metadatas=metadatas,
            documents=documents  # This is the critical addition
        )
        print(f"Stored {len(chunks)} chunks from {file_name} page {page_num}")

    def process_pdfs(self):
        total_files = 0
        total_pages = 0
        total_chunks = 0

        # Clear existing collection
        self.chroma_client.delete_collection("pdf_embeddings")
        self.collection = self.chroma_client.get_or_create_collection(
            name="pdf_embeddings",
            metadata={"hnsw:space": "cosine"}
        )

        for file_name in os.listdir(self.data_dir):
            if file_name.endswith(".pdf"):
                total_files += 1
                pdf_path = os.path.join(self.data_dir, file_name)

                try:
                    text_by_page = self.extract_text_from_pdf(pdf_path)
                    for page_num, text in text_by_page:
                        cleaned_text = self.clean_text(text)
                        chunks = self.split_text_into_chunks(cleaned_text)
                        if chunks:
                            self.store_embeddings(file_name, page_num, chunks)
                            total_chunks += len(chunks)
                    total_pages += len(text_by_page)
                    print(f"Processed {file_name} ({len(text_by_page)} pages)")
                except Exception as e:
                    print(f"Error processing {file_name}: {str(e)}")

        print(f"\nIngestion complete. Processed {total_files} files, {total_pages} pages, {total_chunks} chunks.")


if __name__ == "__main__":
    ingestor = ChromaDBIngestor()
    ingestor.process_pdfs()