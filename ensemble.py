import os

from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import ChatGoogleGenerativeAI

from rag_chain import make_rag_chain
from splitter import split_documents
from vector_store import create_vector_db
from vector_store import load_vector_db 
from dotenv import load_dotenv
from memory import create_memory_chain

def get_model():
    return ChatGoogleGenerativeAI(
        model=os.environ.get("GEMINI_MODEL", "gemini-2-flash"),
        api_key=os.environ.get("GOOGLE_API_KEY"),
        temperature=0
    )

def ensemble_retriever_from_docs(docs, embeddings=None):

    texts = split_documents(docs)

    vs = create_vector_db(texts, embeddings)

    vs_retriever = vs.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 8,
            "fetch_k": 20,
            "lambda_mult": 0.5
        },
    )

    bm25_retriever = BM25Retriever.from_texts(
        [t.page_content for t in texts]
    )
    bm25_retriever.k = 8

    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vs_retriever],
        weights=[0.5, 0.5]
    )

    return ensemble_retriever

def load_all_pdfs(data_path="./data"):

    docs = []

    for file in os.listdir(data_path):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(data_path, file))
            docs.extend(loader.load())

    return docs

def get_retriever():

    vector_db = load_vector_db()

    return vector_db.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 4,
            "fetch_k": 20,
            "lambda_mult": 0.5
        },
    )

def main():

    load_dotenv()

    docs = load_all_pdfs("./data")

    ensemble_retriever = ensemble_retriever_from_docs(docs)

    model = get_model()

    rag_chain = make_rag_chain(model, ensemble_retriever)
    memory_chain = create_memory_chain(model, rag_chain)

    chain = memory_chain

    session_id = "user_1"

    while True:
        q = input("You: ")

        result = chain.invoke(
            {"question": q},
            config={"configurable": {"session_id": session_id}}
        )

    print("Bot:", result)

if __name__ == "__main__":
    main()


