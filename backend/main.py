import time
from google import genai
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from backend.config import GOOGLE_API_KEY, GEMINI_MODEL_NAME
from backend.retriever import get_retriever
from backend.ingest import ingest_documents

# Initialize FastAPI
app = FastAPI(title="SEC RAG API")

# Initialize Gemini Client
client = None
if not GOOGLE_API_KEY or "your_google_ai_studio" in GOOGLE_API_KEY:
    print("WARNING: GOOGLE_API_KEY is not set correctly in .env")
else:
    client = genai.Client(api_key=GOOGLE_API_KEY)

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
    if client is None:
        raise HTTPException(status_code=500, detail="Gemini client not initialized. Check your GOOGLE_API_KEY.")

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
    prompt = f"""# Role
You are an expert Financial Analyst Assistant specializing in SEC filings and financial reports. Your goal is to provide accurate, concise, and well-justified answers based on the provided documents.

# Instructions
1. **Analyze the Context**: Carefully read the provided document excerpts to find information relevant to the question.
2. **Answer Strictly from Context**: Provide your answer based ONLY on the provided context. Do not use outside knowledge.
3. **Handle MCQs**: If the question is a Multiple Choice Question (MCQ):
   - State the correct option (e.g., "Correct Option: B") clearly.
   - Provide a concise justification referencing specific details from the context.
   - If a choice is clearly contradicted by the context, briefly explain why.
4. **Missing Information**: If the answer is not contained within the context, state: "The provided context does not contain enough information to answer this question."
5. **Formatting**: Use clean Markdown formatting for your response.

# Context
{context_str}

# Question
{request.question}

# Answer:"""

    # Retry logic for rate limiting
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL_NAME,
                contents=prompt
            )
            return QueryResponse(answer=response.text, sources=sources)
        except Exception as e:
            # Check for rate limit error (429)
            error_msg = str(e).lower()
            if "429" in error_msg or "resource_exhausted" in error_msg:
                if attempt < max_retries - 1:
                    print(f"Rate limit exceeded. Retrying in 1s... (Attempt {attempt+1})")
                    time.sleep(1)
                    continue
                else:
                    raise HTTPException(status_code=429, detail="Gemini API rate limit exceeded. Please try again later.")
            
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
