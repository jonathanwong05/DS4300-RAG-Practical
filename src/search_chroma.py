from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer

db_path = "./chroma_db"
collection_name = "pdf_embeddings"  # Ensure this matches the collection name used in ingestion
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def search(query, top_k=3):
    client = PersistentClient(path=db_path)
    collection = client.get_collection(name=collection_name)

    query_embedding = model.encode([query]).tolist()[0]
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)

    if not results["metadatas"]:
        print("No matching documents found.")
        return

    seen_files = set()  # Track files that have already been printed

    for i, (metadata_list, distances) in enumerate(zip(results["metadatas"], results["distances"])):
        distance = distances[0] if isinstance(distances, list) else distances

        for metadata in metadata_list:
            file_name = metadata.get("file", "No file provided")
            if file_name not in seen_files:  # Check if file is already printed
                seen_files.add(file_name)
                chunk_text = metadata.get("chunk", "No chunk text available")
                similarity = round(1 - distance, 4)  # Convert distance to similarity

                print(f"{i+1}. File: {file_name} (Similarity: {similarity})\n   Snippet: {chunk_text[:300]}...\n")

if __name__ == "__main__":
    query = input("Enter your search query: ")
    search(query)