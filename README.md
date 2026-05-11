# SEC RAG Application

A production-ready RAG application for analyzing SEC documents using FastAPI, Streamlit, ChromaDB, and Gemini 2.5 Flash.

## Project Structure
- `backend/`: FastAPI server for ingestion and querying.
- `frontend/`: Streamlit UI for the end-user.
- `SEC-data/`: Source documents (PDF and TXT).
- `chroma_store/`: Persistent vector database.

## Setup Instructions

### 1. Environment Setup
Create and activate a virtual environment:
```powershell
python -m venv venv
.\venv\Scripts\activate
```

### 2. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 3. Configuration
Copy `.env.example` to `.env` and add your Google AI Studio API key:
```
GOOGLE_API_KEY=your_actual_api_key_here
```

### 4. Run the Application
Start the backend (FastAPI):
```powershell
uvicorn backend.main:app --reload --port 8000
```

In a new terminal, start the frontend (Streamlit):
```powershell
streamlit run frontend/app.py
```

## Features
- **Page-wise Ingestion**: PDFs are parsed and embedded page-by-page.
- **Persistent Storage**: ChromaDB stores embeddings locally in `./chroma_store`.
- **Gemini 2.5 Flash**: High-performance LLM for grounded answer generation.
- **Source Attribution**: Every answer includes links back to the original file and page.
