import os, json
import numpy as np
import faiss
from google import genai
from google.genai import types

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
index = faiss.read_index("hr.faiss")
chunks = json.load(open("hr_chunks.json", encoding="utf-8"))

not_found_rule = """If the excerpts do not clearly answer the question, do NOT guess and do NOT invent a rule.
Say: "The policy doesn't cover this directly." Then, if any excerpt is related, briefly mention what it does say and cite the section.
Finally, suggest contacting HR at hr@abebatrading.example for a definite answer."""

RULES = f"""You are an HR assistant for Abeba Trading PLC.
Answer ONLY using the policy excerpts provided. Do not use outside knowledge.
Mention the section number you used, like (Section 2.1).
{not_found_rule}
Keep answers short and clear."""

while True:
    question = input("\nAsk a question (or type q to quit): ")
    if question.lower() == "q":
        break

    # 1. Find the closest chunks
    q = client.models.embed_content(
        model="gemini-embedding-001",
        contents=[question],
        config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
    )
    q_vec = np.array([q.embeddings[0].values], dtype="float32")
    faiss.normalize_L2(q_vec)
    scores, ids = index.search(q_vec, 5)

    context = "\n\n---\n\n".join(chunks[i] for i in ids[0])

    # 2. Ask Gemini to answer from those chunks
    response = client.models.generate_content(
        model="gemini-3.8-flash",
        contents=f"Policy excerpts:\n{context}\n\nQuestion: {question}",
        config=types.GenerateContentConfig(system_instruction=RULES, temperature=0.2),
    )
    print("\n" + response.text)
    