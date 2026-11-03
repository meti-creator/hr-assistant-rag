from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.vectorstores import FAISS

# 1. Load and chunk (same as Step 1)
docs = TextLoader("hr_policy.txt", encoding="utf-8").load()
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)

# 2. The embedding model (free, runs on your own computer)
embeddings = FastEmbedEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 3. Build the FAISS index and save it
db = FAISS.from_documents(chunks, embeddings)
db.save_local("hr_index")

# 4. Quick test: which chunks are closest to this question?
question = "How many days of annual leave do I get?"
results = db.similarity_search(question, k=2)

for i, r in enumerate(results, 1):
    print(f"--- Result {i} ---")
    print(r.page_content)