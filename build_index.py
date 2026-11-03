from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter


def main():
    docs = TextLoader("hr_policy.txt", encoding="utf-8").load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    embeddings = FastEmbedEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.from_documents(chunks, embeddings)
    db.save_local("hr_index")

    question = "How many days of annual leave do I get?"
    results = db.similarity_search(question, k=2)

    for i, r in enumerate(results, 1):
        print(f"--- Result {i} ---")
        print(r.page_content)


if __name__ == "__main__":
    main()