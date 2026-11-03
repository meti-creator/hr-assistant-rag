import json
import os

import faiss
import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import types
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY is not set. Add it to your environment or a .env file before running this app."
    )


def main():
    client = genai.Client(api_key=GEMINI_API_KEY)

    with open("hr_policy.txt", encoding="utf-8") as f:
        text = f.read()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(text)
    print(f"{len(chunks)} chunks")

    result = client.models.embed_content(
        model="gemini-embedding-001",
        contents=chunks,
        config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
    )
    vectors = np.array([e.values for e in result.embeddings], dtype="float32")
    print("each chunk became", vectors.shape[1], "numbers")

    faiss.normalize_L2(vectors)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    faiss.write_index(index, "hr.faiss")
    with open("hr_chunks.json", "w", encoding="utf-8") as f:
        json.dump(chunks, f)
    print("saved hr.faiss and hr_chunks.json")


if __name__ == "__main__":
    main()