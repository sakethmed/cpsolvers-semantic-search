from youtube_transcript_api import YouTubeTranscriptApi
from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
import faiss
import os
import time
import random

def safe_get_transcript(video_id, retries=5):
    for attempt in range(retries):
        try:
            return YouTubeTranscriptApi.get_transcript(video_id)
        except Exception as e:
            if "429" in str(e):
                wait = 30 * (attempt + 1)
                print(f"⚠️ 429 error. Waiting {wait}s...")
                time.sleep(wait)
            else:
                raise e
    raise Exception("Too many retries for video: " + video_id)

def chunk_transcript(transcript, video_id, max_duration=45):
    chunks = []
    temp_chunk = ""
    chunk_start = transcript[0]["start"]

    for segment in transcript:
        temp_chunk += " " + segment["text"]
        if segment["start"] - chunk_start >= max_duration:
            chunks.append({
                "start": int(chunk_start),
                "text": temp_chunk.strip(),
                "video_id": video_id
            })
            chunk_start = segment["start"]
            temp_chunk = ""

    if temp_chunk.strip():
        chunks.append({
            "start": int(chunk_start),
            "text": temp_chunk.strip(),
            "video_id": video_id
        })

    return chunks

def embed_chunks(chunks, model):
    return [model.encode(chunk["text"]) for chunk in chunks]

def append_to_csv(chunks, filepath):
    df = pd.DataFrame(chunks)
    df.to_csv(filepath, mode='a', header=not os.path.exists(filepath), index=False)

def append_to_processed(video_id, filepath):
    with open(filepath, "a") as f:
        f.write(f"{video_id}\n")

def save_vectors(vectors, filepath):
    if os.path.exists(filepath):
        existing = np.load(filepath)
        vectors = np.concatenate([existing, vectors])
    np.save(filepath, vectors)

# Load video IDs
with open("video_ids.txt") as f:
    all_video_ids = [line.strip() for line in f if line.strip()]

processed_file = "processed_ids.csv"
if os.path.exists(processed_file):
    done_ids = set(pd.read_csv(processed_file)["video_id"])
else:
    done_ids = set()

remaining_ids = [vid for vid in all_video_ids if vid not in done_ids]
model = SentenceTransformer("all-MiniLM-L6-v2")

temp_chunks = []
temp_vectors = []

for count, vid in enumerate(remaining_ids):
    try:
        transcript = safe_get_transcript(vid)
        chunks = chunk_transcript(transcript, vid)
        vectors = embed_chunks(chunks, model)

        temp_chunks.extend(chunks)
        temp_vectors.extend(vectors)

        print(f"✅ Processed {vid} — {len(chunks)} chunks")

        if count % 5 == 0 or count == len(remaining_ids) - 1:
            append_to_csv(temp_chunks, "chunk_metadata.csv")
            save_vectors(np.array(temp_vectors), "chunk_vectors.npy")
            for chunk in temp_chunks:
                append_to_processed(chunk["video_id"], processed_file)
            temp_chunks, temp_vectors = [], []
            print(f"💾 Saved batch at count {count}")

        time.sleep(random.uniform(2.5, 4.5))

    except Exception as e:
        print(f"❌ Error with {vid}: {e}")

# Final FAISS index
vectors = np.load("chunk_vectors.npy")
dimension = vectors.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(vectors)
faiss.write_index(index, "cpsolvers_index.faiss")
print("✅ All done. FAISS index saved!")
