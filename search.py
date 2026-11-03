import os, json
import numpy as np
import faiss
from google import genai
from google.genai import types

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
index = faiss.read_index("hr.faiss")
chunks = json.load(open("hr_chunks.json", encoding="utf-8"))

while True:
    question = input("\nAsk a question (or type q to quit): ")
    if question.lower() == "q":
        break

    q = client.models.embed_content(
        model="gemini-embedding-001",
        contents=[question],
        config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
    )
    q_vec = np.array([q.embeddings[0].values], dtype="float32")
    faiss.normalize_L2(q_vec)

    scores, ids = index.search(q_vec, 5)
    for rank, (s, i) in enumerate(zip(scores[0], ids[0]), 1):
        print(f"\n--- Result {rank} (score {s:.2f}) ---")
        print(chunks[i][:200], "...")