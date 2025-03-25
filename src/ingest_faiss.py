import os
import time
import fitz  # PyMuPDF
import re
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Initialize the embedding model
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Faiss index setup (384 is the dimensionality of the 'all-MiniLM-L6-v2' embeddings)
embedding_dim = 384
index = faiss.IndexFlatL2(embedding_dim)

# Metadata storage (to be used in the search function)
metadata = []


# Preprocessing function
def clean_text(text, remove_punctuation=False, remove_whitespace=False):
    if remove_punctuation:
        text = re.sub(r'[^\w\s]', '', text)
    if remove_whitespace:
        text = ' '.join(text.split())
    return text


# Extract text from a PDF by page
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text_by_page = []
    for page_num, page in enumerate(doc):
        text_by_page.append((page_num, page.get_text()))
    return text_by_page


# Split text into chunks with overlap
def split_text_into_chunks(text, chunk_size=500, overlap=150):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i: i + chunk_size])
        chunks.append(chunk)
    return chunks


# Store embedding in Faiss and metadata
def store_embedding(file, page, chunk, embedding):
    # Convert embedding to float32 and normalize it
    embedding = np.array(embedding, dtype=np.float32).reshape(1, -1)
    faiss.normalize_L2(embedding)  # Normalize the embedding

    # Add embedding to Faiss index
    index.add(embedding)

    # Store metadata
    metadata.append({"file": file, "page": page, "chunk": chunk})

    print(f"Stored embedding for: {file}, Page {page}")


# Process all PDFs in a directory
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
                for chunk in chunks:
                    embedding = embedding_model.encode(chunk)
                    store_embedding(file_name, page_num, chunk, embedding)

            file_end = time.time()
            print(
                f"Processed {file_name}: {file_page_count} pages, {file_chunk_count} chunks in {file_end - file_start:.2f} seconds")
            total_pages += file_page_count
            total_chunks += file_chunk_count

    print(f"\nProcessed {total_files} files, {total_pages} pages, {total_chunks} chunks.")


if __name__ == "__main__":
    data_dir = "../data/"  # Update to your actual data directory
    process_pdfs(data_dir)
