import os
import pdfplumber
from backend.retriever import get_retriever

def ingest_documents(data_dir="./SEC-data"):
    retriever = get_retriever()
    documents_ingested = 0
    chunks_created = 0

    for root, dirs, files in os.walk(data_dir):
        for file in files:
            file_path = os.path.join(root, file)
            filename = os.path.relpath(file_path, data_dir)
            
            if file.endswith(".pdf"):
                print(f"Ingesting PDF: {file_path}")
                try:
                    with pdfplumber.open(file_path) as pdf:
                        for page_num, page in enumerate(pdf.pages, start=1):
                            text = page.extract_text()
                            if text and text.strip():
                                chunk_id = f"{filename}_page_{page_num}"
                                retriever.add_documents(
                                    documents=[text],
                                    metadatas=[{"source": filename, "page": page_num}],
                                    ids=[chunk_id]
                                )
                                chunks_created += 1
                    documents_ingested += 1
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")

            elif file.endswith(".txt"):
                print(f"Ingesting TXT: {file_path}")
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        text = f.read()
                        if text and text.strip():
                            chunk_id = f"{filename}_page_1"
                            retriever.add_documents(
                                documents=[text],
                                metadatas=[{"source": filename, "page": 1}],
                                ids=[chunk_id]
                            )
                            chunks_created += 1
                    documents_ingested += 1
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")

    return documents_ingested, chunks_created
