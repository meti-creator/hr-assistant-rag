import os, json
import numpy as np
import faiss
from google import genai
from google.genai import types
from langchain_text_splitters import RecursiveCharacterTextSplitter

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

# 1. Load and chunk (same as before)
text = open("hr_policy.txt", encoding="utf-8").read()
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_text(text)
print(f"{len(chunks)} chunks")

# 2. Turn every chunk into coordinates with Gemini
result = client.models.embed_content(
    model="gemini-embedding-001",
    contents=chunks,
    config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
)
vectors = np.array([e.values for e in result.embeddings], dtype="float32")
print("each chunk became", vectors.shape[1], "numbers")

# 3. Put them in FAISS (normalizing makes "inner product" behave like cosine similarity)
faiss.normalize_L2(vectors)
index = faiss.IndexFlatIP(vectors.shape[1])
index.add(vectors)

# 4. Save the index and the original chunk text
faiss.write_index(index, "hr.faiss")
json.dump(chunks, open("hr_chunks.json", "w", encoding="utf-8"))
print("saved hr.faiss and hr_chunks.json")