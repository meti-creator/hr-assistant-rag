import os, json
import numpy as np
import faiss
from google import genai
from google.genai import types

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
index = faiss.read_index("hr.faiss")
chunks = json.load(open("hr_chunks.json", encoding="utf-8"))

question = "How many days of annual leave do I get?"

q = client.models.embed_content(
    model="gemini-embedding-001",
    contents=[question],
    config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
)
q_vec = np.array([q.embeddings[0].values], dtype="float32")
faiss.normalize_L2(q_vec)

scores, ids = index.search(q_vec, 2)
for rank, (s, i) in enumerate(zip(scores[0], ids[0]), 1):
    print(f"--- Result {rank} (score {s:.2f}) ---")
    print(chunks[i])