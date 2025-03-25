import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

# Load the embedding model
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Faiss index setup (same as ingestion)
embedding_dim = 384
index = faiss.IndexFlatL2(embedding_dim)

# Metadata storage (this should be loaded from wherever it's being stored)
metadata = []

# Debugging: print the size of the Faiss index and metadata
print(f"Faiss index size: {index.ntotal}")
print(f"Metadata size: {len(metadata)}")


# Search function to retrieve documents
def search(query, top_k=3):
    query_embedding = embedding_model.encode([query])[0].astype(np.float32).reshape(1, -1)
    faiss.normalize_L2(query_embedding)  # Normalize the query embedding

    print(f"Query embedding: {query_embedding[:10]}...")  # Debugging output for the query embedding

    # Perform the search in Faiss
    distances, indices = index.search(query_embedding, top_k)

    # Debugging: Check the returned indices
    print(f"Distances: {distances}")
    print(f"Indices: {indices}")

    # Check if results exist
    if len(indices[0]) == 0:
        print("No matching documents found.")
        return

    # Retrieve and display the top-k results
    for i, idx in enumerate(indices[0]):
        # Ensure the index is within the metadata bounds
        if 0 <= idx < len(metadata):
            file_name = metadata[idx].get("file", "No file provided")
            chunk_text = metadata[idx].get("chunk", "No chunk provided")
            similarity = round(1 - distances[0][i], 4)  # Convert distance to similarity
            print(f"{i + 1}. File: {file_name} (Similarity: {similarity})\n   Snippet: {chunk_text[:300]}...\n")
        else:
            print(f"Warning: Index {idx} out of range for metadata.")


if __name__ == "__main__":
    query = input("Enter your search query: ")
    search(query)

