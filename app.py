import os
import urllib.request
import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import streamlit as st
import re

# ---- Download large files from Google Drive ---- #
FAISS_FILE_ID = "1znp6SRpyaS6HYELxkXNwNXsq7xNNIT2R"  # replace this
NPY_FILE_ID = "1s5SDmuJU0fwzBu6SdLs2j990eXmnpwoz"      # replace this

FAISS_URL = f"https://drive.google.com/uc?export=download&id={FAISS_FILE_ID}"
NPY_URL = f"https://drive.google.com/uc?export=download&id={NPY_FILE_ID}"

FAISS_FILE = "cpsolvers_index.faiss"
NPY_FILE = "chunk_vectors.npy"

if not os.path.exists(FAISS_FILE):
    st.info("Downloading FAISS index from Google Drive...")
    urllib.request.urlretrieve(FAISS_URL, FAISS_FILE)

if not os.path.exists(NPY_FILE):
    st.info("Downloading vectors from Google Drive...")
    urllib.request.urlretrieve(NPY_URL, NPY_FILE)

# ---- Load data ---- #
df = pd.read_csv("chunk_metadata.csv")
chunks = df.to_dict(orient="records")
index = faiss.read_index(FAISS_FILE)
vectors = np.load(NPY_FILE)
model = SentenceTransformer("all-MiniLM-L6-v2")

# ---- UI + Boolean Search Logic ---- #
st.title("🧠 Clinical Problem Solvers Semantic Search")
st.markdown("Use **AND**, **OR**, **NOT**. Example: `fever and night sweats and not TB`")

query = st.text_input("🔍 Enter query:")

def parse_query(query):
    query = query.lower()
    not_terms = re.findall(r'\bnot\s+([\w\s]+)', query)
    query = re.sub(r'\bnot\s+[\w\s]+', '', query)
    or_groups = [part.strip() for part in query.split(' or ')]
    parsed = {"or_groups": [], "not_terms": [t.strip() for t in not_terms if t.strip()]}
    for group in or_groups:
        and_terms = [term.strip() for term in group.split(' and ') if term.strip()]
        if and_terms:
            parsed["or_groups"].append(and_terms)
    return parsed

if query:
    parsed = parse_query(query)
    all_terms = [term for group in parsed["or_groups"] for term in group]
    term_embeddings = {term: model.encode(term) for term in all_terms}
    query_vector = np.mean(list(term_embeddings.values()), axis=0).reshape(1, -1)
    D, I = index.search(query_vector, k=30)

    shown = 0
    for idx in I[0]:
        chunk = chunks[idx]
        text = chunk["text"].lower()

        if any(term in text for term in parsed["not_terms"]):
            continue

        match = False
        for group in parsed["or_groups"]:
            if all(term in text for term in group):
                match = True
                break

        if match:
            link = f"https://www.youtube.com/watch?v={chunk['video_id']}&t={chunk['start']}s"
            st.markdown(f"### [🔗 Jump to Segment]({link})")
            st.write(chunk["text"][:400] + "...")
            st.markdown("---")
            shown += 1

        if shown >= 5:
            break
