import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Configuration constants
CHROMA_PERSIST_DIRECTORY = "./chroma_store"
COLLECTION_NAME = "sec_documents"
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1"
GEMINI_MODEL_NAME = "gemini-2.0-flash" # Use the latest available name
