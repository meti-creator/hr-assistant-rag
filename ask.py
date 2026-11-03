import json
import os
import time

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

MODELS = ["gemini-3.8-flash", "gemini-flash-latest"]
RETRYABLE_STATUS_CODES = {408, 429, 500, 502, 503, 504}


def generate_with_retry(contents, config):
    failures = {}
    for attempt in range(4):
        for model in MODELS:
            try:
                return client.models.generate_content(
                    model=model,
                    contents=contents,
                    config=config,
                )
            except Exception as exc:
                failures[model] = exc
                print(
                    f"{model} failed ({type(exc).__name__}: {exc}); "
                    "trying the next model..."
                )
                status_code = getattr(exc, "code", None)
                if status_code is not None and status_code not in RETRYABLE_STATUS_CODES:
                    raise RuntimeError(
                        f"Gemini request failed for {model}: {exc}"
                    ) from exc
        time.sleep(2 ** attempt)
    details = "; ".join(
        f"{model}: {type(error).__name__}: {error}"
        for model, error in failures.items()
    )
    raise RuntimeError(
        f"Gemini generation failed after retries. Errors: {details}"
    ) from next(reversed(failures.values()))


not_found_rule = """If the excerpts do not clearly answer the question, do NOT guess and do NOT invent a rule.
Say: "The policy doesn't cover this directly." Then, if any excerpt is related, briefly mention what it does say and cite the section.
Finally, suggest contacting HR at hr@abebatrading.example for a definite answer."""

RULES = f"""You are an HR assistant for Abeba Trading PLC.
Answer ONLY using the policy excerpts provided. Do not use outside knowledge.
Mention the section number you used, like (Section 2.1).
{not_found_rule}
Keep answers short and clear."""


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

        context = "\n\n---\n\n".join(chunks[i] for i in ids[0])

        response = generate_with_retry(
            contents=f"Policy excerpts:\n{context}\n\nQuestion: {question}",
            config=types.GenerateContentConfig(system_instruction=RULES, temperature=0.2),
        )
        print("\n" + response.text)


if __name__ == "__main__":
    main()
