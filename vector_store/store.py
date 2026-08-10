import os
import numpy as np

# Try importing ChromaDB
try:
    import chromadb
    from chromadb.config import Settings
except ImportError:
    chromadb = None


class ChromaVectorStore:
    """
    Manages the ChromaDB client configuration, index persistence,
    document insertion, and semantic similarity searching.
    """

    def __init__(self, persist_dir: str = "./db/chroma_db"):
        self.persist_dir = persist_dir
        os.makedirs(os.path.dirname(persist_dir), exist_ok=True)
        self.model = None  # Lazy-loaded on first embedding generation

        # Load Chroma Client
        if chromadb:
            # Pass a dummy embedding function to prevent Chroma from instantiating its default models
            class DummyEmbeddingFunction:
                def __call__(self, input):
                    return []

                def name(self):
                    return "default"

            dummy_ef = DummyEmbeddingFunction()
            self.client = chromadb.PersistentClient(path=persist_dir)
            self.collection = self.client.get_or_create_collection(
                name="health_insurance_kb",
                metadata={"hnsw:space": "cosine"},
                embedding_function=dummy_ef,
            )
        else:
            self.client = None
            self.collection = None
            self.fallback_db = []  # In-memory list backup
            print("Warning: ChromaDB is not installed. Running in-memory lookup mode.")

    def has_document(self, record_id: str) -> bool:
        """
        Checks if a document exists by ID without using or calculating embeddings.
        """
        if self.collection:
            try:
                res = self.collection.get(ids=[record_id])
                return res and len(res.get("ids", [])) > 0
            except Exception as e:
                print(f"Error checking document presence for {record_id}: {e}")
                return False
        else:
            return any(item["id"] == record_id for item in self.fallback_db)

    def get_embedding(self, text: str) -> list[float]:
        """
        Generates standard 384-dimensional dense vectors.
        """
        if self.model is None:
            # Optionally collect lightweight memory diagnostics when loading the model
            try:
                import psutil
            except Exception:
                psutil = None

            try:
                from sentence_transformers import SentenceTransformer

                if psutil:
                    proc = psutil.Process()
                    before = proc.memory_info().rss / (1024 * 1024)
                    print(f"Memory before embedding model load: {before:.1f} MB")

                print(
                    "Loading SentenceTransformer model 'all-MiniLM-L6-v2' on CPU (lazy)..."
                )
                # Force CPU execution to prevent GPU/CUDA initialization
                self.model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

                if psutil:
                    after = proc.memory_info().rss / (1024 * 1024)
                    print(f"Memory after embedding model load: {after:.1f} MB")
            except ImportError:
                print(
                    "Warning: SentenceTransformers is not installed. running in mock vector mode."
                )
                self.model = "fallback"

        if self.model != "fallback":
            return self.model.encode(text).tolist()
        else:
            # Deterministic mock hash vectors for fallback compatibility
            np.random.seed(abs(hash(text)) % (2**32 - 1))
            mock_vec = np.random.randn(384)
            norm = np.linalg.norm(mock_vec)
            if norm > 0:
                mock_vec = mock_vec / norm
            return mock_vec.tolist()

    def add_documents(self, documents: list[dict]):
        """
        Saves parsed chunks and associated metadata into the database index.
        """
        ids = []
        embeddings = []
        texts = []
        metadatas = []

        for idx, doc in enumerate(documents):
            chunk_id = doc.get("record_id", f"chunk_{idx}")
            content = doc.get("content", "")

            # Generate embedding vector
            vector = self.get_embedding(content)

            # Filter and prepare metadata values
            meta = {
                "title": str(doc.get("title", "Unknown")),
                "category": str(doc.get("category", "General")),
                "source": str(doc.get("source", "Unknown")),
                "page": str(doc.get("page", "1")),
                "section": str(doc.get("section", "General")),
                "url": str(doc.get("url", "")),
                "version": str(doc.get("version", "1.0")),
                "timestamp": str(doc.get("timestamp", "")),
            }

            ids.append(chunk_id)
            embeddings.append(vector)
            texts.append(content)
            metadatas.append(meta)

            # Save to fallback in case Chroma is absent
            if not self.collection:
                self.fallback_db.append(
                    {
                        "id": chunk_id,
                        "content": content,
                        "embedding": vector,
                        "metadata": meta,
                    }
                )

        if self.collection:
            self.collection.add(
                ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas
            )
        print(f"Ingested {len(documents)} document chunks into vector database.")

    def query(self, query_text: str, limit: int = 5) -> list[dict]:
        """
        Queries the vector index for similar text chunks.
        """
        query_vector = self.get_embedding(query_text)
        results = []

        if self.collection:
            res = self.collection.query(
                query_embeddings=[query_vector], n_results=limit
            )
            # Format results
            if res and res["documents"]:
                for i in range(len(res["documents"][0])):
                    # Convert distance to similarity score
                    dist = res["distances"][0][i] if res["distances"] else 1.0
                    similarity = 1.0 - float(dist)

                    results.append(
                        {
                            "id": res["ids"][0][i],
                            "content": res["documents"][0][i],
                            "metadata": res["metadatas"][0][i],
                            "score": similarity,
                        }
                    )
        else:
            # Fallback memory cosine similarity calculation
            candidates = []
            for item in self.fallback_db:
                # Cosine similarity calculation
                dot_product = np.dot(query_vector, item["embedding"])
                candidates.append((dot_product, item))

            # Sort by dot product score
            candidates.sort(key=lambda x: x[0], reverse=True)

            for score, item in candidates[:limit]:
                results.append(
                    {
                        "id": item["id"],
                        "content": item["content"],
                        "metadata": item["metadata"],
                        "score": float(score),
                    }
                )

        return results
