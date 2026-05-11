import time
import google.generativeai as genai
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from backend.config import GOOGLE_API_KEY, GEMINI_MODEL_NAME
from backend.retriever import get_retriever
from backend.ingest import ingest_documents
import google.api_core.exceptions

# Initialize FastAPI
app = FastAPI(title="SEC RAG API")

# Initialize Gemini
if not GOOGLE_API_KEY or "your_google_ai_studio" in GOOGLE_API_KEY:
    print("WARNING: GOOGLE_API_KEY is not set correctly in .env")
else:
    genai.configure(api_key=GOOGLE_API_KEY)

class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = 5

class Source(BaseModel):
    file: str
    page: int

class QueryResponse(BaseModel):
    answer: str
    sources: List[Source]

@app.on_event("startup")
async def startup_event():
    # Load retriever and model at startup
    get_retriever()

@app.post("/ingest")
async def ingest():
    try:
        docs, chunks = ingest_documents()
        return {
            "status": "success",
            "documents_ingested": docs,
            "chunks_created": chunks
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    retriever = get_retriever()
    
    # 1. Retrieve chunks
    results = retriever.query(request.question, top_k=request.top_k)
    
    if not results or not results['documents'][0]:
        return QueryResponse(answer="No relevant documents found.", sources=[])

    # 2. Build context
    context_parts = []
    sources = []
    for i in range(len(results['documents'][0])):
        text = results['documents'][0][i]
        meta = results['metadatas'][0][i]
        source_info = f"Source: {meta['source']}, Page: {meta['page']}"
        context_parts.append(f"--- {source_info} ---\n{text}")
        sources.append(Source(file=meta['source'], page=meta['page']))
    
    context_str = "\n\n".join(context_parts)
    
    # 3. Call Gemini
    prompt = f"""You are a financial analyst assistant. Answer based only on the provided context. 
If the answer is not in the context, say so clearly.

Context:
{context_str}

Question: {request.question}

Answer:"""

    model = genai.GenerativeModel(GEMINI_MODEL_NAME)
    
    # Retry logic for rate limiting
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = model.generate_content(prompt)
            return QueryResponse(answer=response.text, sources=sources)
        except google.api_core.exceptions.ResourceExhausted:
            if attempt < max_retries - 1:
                print(f"Rate limit exceeded. Retrying in 1s... (Attempt {attempt+1})")
                time.sleep(1)
            else:
                raise HTTPException(status_code=429, detail="Gemini API rate limit exceeded. Please try again later.")
        except Exception as e:
            print(f"Gemini API Error: {e}")
            raise HTTPException(status_code=500, detail=f"LLM Error: {str(e)}")

@app.get("/health")
async def health():
    retriever = get_retriever()
    count = retriever.collection.count()
    return {
        "status": "ok",
        "chroma_collection_count": count
    }
