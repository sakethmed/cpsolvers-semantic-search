import streamlit as st
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import re

# Load data
df = pd.read_csv("chunk_metadata.csv")
chunks = df.to_dict(orient="records")
index = faiss.read_index("cpsolvers_index.faiss")
model = SentenceTransformer("all-MiniLM-L6-v2")

# UI
st.title("🧠 Clinical Problem Solvers Semantic Search")
st.markdown("Enter queries like `fever and night sweats and not malaria` using **AND**, **OR**, and **NOT**.")

query = st.text_input("🔍 Enter your query:")

def parse_boolean_query(query):
    query = query.lower()

    # Extract NOT terms
    not_terms = re.findall(r'\bnot\s+([\w\s]+)', query)
    query = re.sub(r'\bnot\s+[\w\s]+', '', query)

    # Split into OR groups
    or_groups = [part.strip() for part in query.split(' or ')]

    parsed = {
        "or_groups": [],
        "not_terms": [term.strip() for term in not_terms if term.strip()]
    }

    for group in or_groups:
        and_terms = [term.strip() for term in group.split(' and ') if term.strip()]
        if and_terms:
            parsed["or_groups"].append(and_terms)

    return parsed

if query:
    parsed = parse_boolean_query(query)

    # Embed all terms
    all_terms = [term for group in parsed["or_groups"] for term in group]
    term_embeddings = {term: model.encode(term) for term in all_terms}

    # Use average vector
    query_vector = np.mean(list(term_embeddings.values()), axis=0).reshape(1, -1)
    D, I = index.search(query_vector, k=30)

    shown = 0
    for idx in I[0]:
        chunk = chunks[idx]
        text = chunk["text"].lower()

        # Exclude NOT terms
        if any(term in text for term in parsed["not_terms"]):
            continue

        # Match OR groups (each OR group must match all its AND terms)
        match = False
        for group in parsed["or_groups"]:
            if all(term in text for term in group):
                match = True
                break

        if match:
            video_id = chunk["video_id"]
            start = chunk["start"]
            link = f"https://www.youtube.com/watch?v={video_id}&t={start}s"

            st.markdown(f"### [🔗 Segment Link]({link})")
            st.write(chunk["text"][:400] + "...")
            st.markdown("---")
            shown += 1

        if shown >= 5:
            break
