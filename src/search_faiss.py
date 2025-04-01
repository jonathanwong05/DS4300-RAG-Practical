import numpy as np
import faiss
import pickle
from sentence_transformers import SentenceTransformer
import ollama


class FAISSSearcher:
    def __init__(self):
        self.embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.embedding_dim = 384

        try:
            # Load FAISS index
            self.index = faiss.read_index("faiss_index.index")

            # Load metadata
            with open("faiss_metadata.pkl", "rb") as f:
                self.metadata = pickle.load(f)

            print(f"Loaded FAISS index with {self.index.ntotal} vectors and {len(self.metadata)} metadata entries")

        except Exception as e:
            raise Exception(f"Failed to load FAISS index: {str(e)}. Please run ingestion first.")

    def search(self, query, top_k=3):
        """Search for similar chunks"""
        # Encode and normalize query
        query_embedding = self.embedding_model.encode([query])[0].astype('float32')
        query_embedding = np.expand_dims(query_embedding, axis=0)
        faiss.normalize_L2(query_embedding)

        # Search FAISS index
        distances, indices = self.index.search(query_embedding, top_k)

        # Prepare results
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if 0 <= idx < len(self.metadata):
                metadata = self.metadata[idx]
                results.append({
                    "file": metadata["file"],
                    "page": metadata["page"],
                    "chunk": metadata["chunk"],
                    "similarity": 1 - distance  # Convert distance to similarity
                })

        return results

    def generate_response(self, query, context_results):
        """Generate RAG response"""
        if not context_results:
            return "I couldn't find any relevant information to answer your question."

        context_str = "\n\n".join(
            f"From {res['file']} (Page {res['page']}):\n"
            f"{res['chunk']}\n"
            f"(Similarity: {res['similarity']:.3f})"
            for res in context_results
        )

        prompt = f"""You are a technical assistant. Answer the question using only the provided context.
If the answer isn't in the context, say you don't know.

Question: {query}

Context:
{context_str}

Answer in detail:"""

        try:
            response = ollama.chat(
                model="mistral",
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.3}
            )
            return response['message']['content']
        except Exception as e:
            return f"Error generating response: {str(e)}"

    def interactive_search(self):
        """Interactive search interface"""
        print("🔍 FAISS RAG Search Interface")
        print("Type 'exit' to quit\n")

        while True:
            query = input("\nEnter your search query: ").strip()
            if query.lower() in ('exit', 'quit'):
                break

            try:
                print("\nSearching...")
                results = self.search(query)

                if not results:
                    print("\nNo relevant documents found.")
                    continue

                print("\nTop matches:")
                for i, res in enumerate(results, 1):
                    preview = res['chunk'][:200].replace('\n', ' ')
                    print(f"{i}. {res['file']} (Page {res['page']}, Similarity: {res['similarity']:.3f})")
                    print(f"   {preview}...\n")

                print("Generating response...")
                response = self.generate_response(query, results)
                print("\n" + "=" * 50)
                print("AI Response:")
                print("=" * 50)
                print(response)
                print("=" * 50)

            except Exception as e:
                print(f"\nError: {str(e)}")


if __name__ == "__main__":
    try:
        searcher = FAISSSearcher()
        searcher.interactive_search()
    except Exception as e:
        print(f"Error: {str(e)}")
        print("Please ensure you've run the ingestion process first.")