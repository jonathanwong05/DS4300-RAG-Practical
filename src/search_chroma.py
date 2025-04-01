import chromadb
from sentence_transformers import SentenceTransformer
import ollama
from typing import List, Dict, Any


class ChromaDBSearcher:
    def __init__(self, db_path="./chroma_db"):
        self.client = chromadb.PersistentClient(path=db_path)
        try:
            self.collection = self.client.get_collection("pdf_embeddings")
        except ValueError:
            raise Exception("Collection not found. Please run ingestion first.")
        self.embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.ensure_ollama_model()

    def ensure_ollama_model(self):
        """Check if Mistral model is available, pull if needed"""
        try:
            ollama.show("mistral")
        except Exception:
            print("Mistral model not found. Pulling it now...")
            ollama.pull("mistral")
            print("Mistral model ready!")

    def search_embeddings(self, query: str, top_k: int = 3) -> Dict[str, Any]:
        query_embedding = self.embedding_model.encode(query).tolist()
        return self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["metadatas", "distances", "documents"]
        )

    def format_results(self, results: Dict[str, Any]) -> List[Dict[str, Any]]:
        formatted = []
        for metadata, document, distance in zip(
                results['metadatas'][0],
                results['documents'][0],
                results['distances'][0]
        ):
            if document:
                formatted.append({
                    "file": metadata.get('file', 'Unknown'),
                    "page": metadata.get('page', '0'),
                    "content": document,
                    "similarity": 1 - float(distance)
                })
        return formatted

    def generate_response(self, query: str, context: List[Dict[str, Any]]) -> str:
        if not context:
            return "I couldn't find any relevant information in the documents."

        context_str = "\n\n".join(
            f"From {res['file']} (Page {res['page']}):\n"
            f"{res['content']}\n"
            for res in context
        )

        prompt = f"""You are a technical assistant. Answer the question using only the provided context.
If the answer isn't in the context, say you don't know.

Context:
{context_str}

Question: {query}

Answer in detail, focusing on the technical aspects:"""

        try:
            response = ollama.chat(
                model="mistral",
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.3}  # More focused responses
            )
            return response['message']['content']
        except Exception as e:
            return f"Error: Could not generate response. Please try again later.\n{str(e)}"

    def interactive_search(self):
        print("🔍 ChromaDB RAG Search Interface")
        print("Type 'exit' to quit\n")

        while True:
            query = input("\nEnter your search query: ").strip()
            if query.lower() in ('exit', 'quit'):
                break

            try:
                print("\nSearching...")
                raw_results = self.search_embeddings(query)
                results = self.format_results(raw_results)

                if not results:
                    print("\nNo relevant documents found.")
                    continue

                print("\nTop matches:")
                for i, res in enumerate(results, 1):
                    preview = res['content'][:200].replace('\n', ' ')
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
        searcher = ChromaDBSearcher()
        searcher.interactive_search()
    except Exception as e:
        print(f"Error: {str(e)}")
        print("Possible solutions:")
        print("1. Run ingest_chroma.py first to create the document collection")
        print("2. Ensure Ollama is running (run 'ollama serve' in another terminal)")
        print("3. Install required models with 'ollama pull mistral'")