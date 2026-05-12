# SEC RAG Application

A production-ready RAG application for analyzing SEC documents using FastAPI, Streamlit, ChromaDB, and Gemma-4-26B.

## Project Structure
- `backend/`: FastAPI server for ingestion and querying.
- `frontend/`: Streamlit UI for the end-user.
- `SEC-data/`: Source documents (PDF and TXT).
- `chroma_store/`: Persistent vector database.

## Model
This application utilizes the **Gemma-4-26B** model for generating answers, providing high-quality reasoning and structured outputs.

## API Endpoints
The backend provides the following endpoints:

- **`GET /health`**: Returns the system status and the number of documents currently stored in the vector database.
- **`POST /ingest`**: Triggers the ingestion process. It scans the `SEC-data/` directory, extracts text from PDF and TXT files, and stores their embeddings in ChromaDB.
- **`POST /query`**: Accepts a JSON payload with a `question` and optional `top_k` value. It retrieves relevant document chunks and generates an answer grounded in the context.

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
- **Gemma-4-26B Integration**: Leverages advanced model capabilities for grounded answer generation.
- **Source Attribution**: Every answer includes links back to the original file and page.
