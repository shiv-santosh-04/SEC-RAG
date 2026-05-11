import streamlit as st
import requests
import json

# Configuration
BACKEND_URL = "http://localhost:8000"

st.set_page_config(page_title="SEC Document Research Assistant", layout="wide")

st.title("SEC Document Research Assistant")

# Sidebar
st.sidebar.header("Management")

if st.sidebar.button("Ingest SEC Documents"):
    with st.spinner("Ingesting documents... This may take a while."):
        try:
            response = requests.post(f"{BACKEND_URL}/ingest")
            if response.status_code == 200:
                data = response.json()
                st.sidebar.success(f"Ingested {data['documents_ingested']} documents!")
                st.sidebar.info(f"Chunks created: {data['chunks_created']}")
            else:
                st.sidebar.error(f"Error: {response.status_code} - {response.text}")
        except Exception as e:
            st.sidebar.error(f"Failed to connect to backend: {e}")

st.sidebar.divider()
st.sidebar.header("System Health")

try:
    health_resp = requests.get(f"{BACKEND_URL}/health")
    if health_resp.status_code == 200:
        health_data = health_resp.json()
        st.sidebar.write(f"Status: ✅ {health_data['status']}")
        st.sidebar.write(f"Documents in store: {health_data['chroma_collection_count']}")
    else:
        st.sidebar.write("Status: ❌ Error")
except:
    st.sidebar.write("Status: ❌ Offline (Backend not running)")

# Main Area
st.subheader("Ask a question about the SEC filings...")

col1, col2 = st.columns([4, 1])

with col1:
    question = st.text_input("Question:", placeholder="e.g., What are the risk factors mentioned in the report?")

with col2:
    top_k = st.slider("Number of sources (top_k)", 1, 10, 5)

if st.button("Search"):
    if not question:
        st.warning("Please enter a question.")
    else:
        with st.spinner("Searching and generating answer..."):
            try:
                payload = {"question": question, "top_k": top_k}
                response = requests.post(f"{BACKEND_URL}/query", json=payload)
                
                if response.status_code == 200:
                    result = response.json()
                    
                    st.markdown("### Answer")
                    st.info(result['answer'])
                    
                    if result['sources']:
                        with st.expander("Sources"):
                            for idx, source in enumerate(result['sources'], start=1):
                                st.write(f"**{idx}. {source['file']}** (Page {source['page']})")
                elif response.status_code == 429:
                    st.error("Rate limit exceeded. Please wait a moment and try again.")
                else:
                    st.error(f"Error: {response.status_code} - {response.text}")
            except Exception as e:
                st.error(f"Failed to connect to backend: {e}")
