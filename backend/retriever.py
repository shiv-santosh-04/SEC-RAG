import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from backend.config import CHROMA_PERSIST_DIRECTORY, COLLECTION_NAME, EMBEDDING_MODEL_NAME

class Retriever:
    def __init__(self):
        # Load the embedding model once at startup
        print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}...")
        self.model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        
        # Initialize persistent ChromaDB client
        self.client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIRECTORY)
        
        # Get or create the collection
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )

    def get_embedding(self, text):
        return self.model.encode(text).tolist()

    def add_documents(self, documents, metadatas, ids):
        embeddings = self.model.encode(documents).tolist()
        self.collection.add(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )

    def query(self, question, top_k=5):
        query_embedding = self.get_embedding(question)
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        return results

# Singleton instance
retriever_instance = None

def get_retriever():
    global retriever_instance
    if retriever_instance is None:
        retriever_instance = Retriever()
    return retriever_instance
