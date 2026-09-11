import streamlit as st
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import httpx
import os

# Set up the page
st.set_page_config(page_title="Tax Buddy Pakistan", layout="centered")
st.title("🇵🇰 Tax Buddy Pakistan")
st.caption("AI assistant for Pakistani tax law. Answers with citations from FBR documents.")

# (We will add the rest of the search and chat logic here in the next step)
st.write("Welcome to the Tax Buddy interface! The UI is ready to be connected to the database.")

query = st.text_input("Ask a tax question:", placeholder="e.g., What is the sales tax rate?")
if st.button("Ask"):
    st.info("I am ready to search for: " + query)
