import os
import numpy as np

# Try importing ChromaDB and SentenceTransformers
try:
    import chromadb
    from chromadb.config import Settings
except ImportError:
    chromadb = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

class ChromaVectorStore:
    """
    Manages the ChromaDB client configuration, index persistence,
    document insertion, and semantic similarity searching.
    """
    def __init__(self, persist_dir: str = "./db/chroma_db"):
        self.persist_dir = persist_dir
        os.makedirs(os.path.dirname(persist_dir), exist_ok=True)
        
        # Load Sentence Transformers model
        if SentenceTransformer:
            self.model = SentenceTransformer("all-MiniLM-L6-v2")
        else:
            self.model = None
            print("Warning: SentenceTransformers is not installed. running in mock vector mode.")
            
        # Load Chroma Client
        if chromadb:
            self.client = chromadb.PersistentClient(path=persist_dir)
            self.collection = self.client.get_or_create_collection(
                name="health_insurance_kb",
                metadata={"hnsw:space": "cosine"}
            )
        else:
            self.client = None
            self.collection = None
            self.fallback_db = [] # In-memory list backup
            print("Warning: ChromaDB is not installed. Running in-memory lookup mode.")

    def get_embedding(self, text: str) -> list[float]:
        """
        Generates standard 384-dimensional dense vectors.
        """
        if self.model:
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
                "timestamp": str(doc.get("timestamp", ""))
            }
            
            ids.append(chunk_id)
            embeddings.append(vector)
            texts.append(content)
            metadatas.append(meta)
            
            # Save to fallback in case Chroma is absent
            if not self.collection:
                self.fallback_db.append({
                    "id": chunk_id,
                    "content": content,
                    "embedding": vector,
                    "metadata": meta
                })

        if self.collection:
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas
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
                query_embeddings=[query_vector],
                n_results=limit
            )
            # Format results
            if res and res["documents"]:
                for i in range(len(res["documents"][0])):
                    # Convert distance to similarity score
                    dist = res["distances"][0][i] if res["distances"] else 1.0
                    similarity = 1.0 - float(dist)
                    
                    results.append({
                        "id": res["ids"][0][i],
                        "content": res["documents"][0][i],
                        "metadata": res["metadatas"][0][i],
                        "score": similarity
                    })
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
                results.append({
                    "id": item["id"],
                    "content": item["content"],
                    "metadata": item["metadata"],
                    "score": float(score)
                })
                
        return results
