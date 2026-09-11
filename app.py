import streamlit as st
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import httpx
import os

# 1. Page Setup
st.set_page_config(page_title="Tax Buddy Pakistan", layout="centered")
st.title("🇵🇰 Tax Buddy Pakistan")
st.caption("AI assistant for Pakistani tax law. Answers with citations from FBR documents.")

# 2. Securely fetch the API Key (from Hugging Face Secrets)
GROQ_KEY = st.secrets.get("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY")

# 3. Load Database and Model once (Cache it so it doesn't reload every click)
@st.cache_resource
def load_brain():
    with open("bm25_index.pkl", "rb") as f:
        data = pickle.load(f)
        bm25_index = data["bm25"]
        chunks = data["chunks"]
    
    embeddings = np.load("embeddings.npy")
    model = SentenceTransformer('BAAI/bge-m3')
    return bm25_index, chunks, embeddings, model

# Try loading, show an error on the screen if files are missing
try:
    bm25_index, all_master_chunks, embeddings, embedding_model = load_brain()
except Exception as e:
    st.error("Database files not found! Make sure bm25_index.pkl and embeddings.npy are uploaded to Hugging Face.")
    st.stop()

# 4. Search Logic
def search_documents(query, k=4):
    q_emb = embedding_model.encode([query], normalize_embeddings=True)[0]
    vec_scores = embeddings @ q_emb
    bm_scores = np.array(bm25_index.get_scores(query.lower().split()))
    
    if bm_scores.max() > 0:
        bm_norm = (bm_scores - bm_scores.min()) / (bm_scores.max() - bm_scores.min())
    else:
        bm_norm = bm_scores
        
    fused = 0.5 * vec_scores + 0.5 * bm_norm
    top_idx = np.argsort(-fused)[:k]
    return [all_master_chunks[i] for i in top_idx]

# 5. UI Interaction
query = st.text_input("Ask a tax question:", placeholder="e.g., What is the penalty for failing to furnish a tax return?")

if st.button("Ask") and query:
    if not GROQ_KEY:
        st.error("GROQ API Key is missing! Please add it in Hugging Face Space Settings -> Variables and secrets.")
        st.stop()
        
    with st.spinner("Searching FBR documents and analyzing..."):
        try:
            # Step A: Search for the best chunks
            results = search_documents(query)
            context = "\n\n".join([f"Source: {c['doc']} (Page {c['page']})\n{c['text']}" for c in results])
            
            # Step B: Give rules to the AI
            prompt = f"""You are a Pakistani tax advisor. Answer ONLY using the context below. 
            CRITICAL RULES:
            1. Always cite the source document and page number in your answer.
            2. If the answer is not in the context, say exactly: "I cannot find this in the FBR documents provided."
            
            Context:
            {context}
            
            Question: {query}"""
            
            # Step C: Ask Groq
            resp = httpx.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {GROQ_KEY}"},
                json={
                    "model": "llama3-8b-8192", 
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.1 
                },
                timeout=60
            )
            
            # Step D: Display the output
            answer = resp.json()["choices"][0]["message"]["content"]
            st.markdown("### Answer")
            st.write(answer)
            
            # Display the source text in a drop-down menu
            with st.expander("📄 View Source Documents"):
                for c in results:
                    st.markdown(f"**{c['doc']} — Page {c['page']}**")
                    st.text(c['text'][:300] + "...")
                    
        except Exception as e:
            st.error(f"An error occurred: {e}")
