import json
import os

import faiss
import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY is not set. Add it to your environment or a .env file before running this app."
    )

client = genai.Client(api_key=GEMINI_API_KEY)
if not os.path.exists("hr.faiss"):
    raise FileNotFoundError("hr.faiss was not found. Generate the index first with build_index_gemini.py.")

index = faiss.read_index("hr.faiss")
with open("hr_chunks.json", encoding="utf-8") as f:
    chunks = json.load(f)


def main():
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


if __name__ == "__main__":
    main()