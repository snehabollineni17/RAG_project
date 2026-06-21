import os
import logging

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

def create_vector_db(texts, embeddings=None, collection_name="chroma"):

    if not texts:
        logging.warning("Empty texts passed to vector DB")

    embeddings = embeddings or get_embeddings()

    db = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=os.path.join("store", collection_name),
    )

    batch_size = 150
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        db.add_documents(batch)

    db.persist()

    return db

def load_vector_db(collection_name="chroma"):

    embeddings = get_embeddings()

    db = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=os.path.join("store", collection_name),
    )

    return db