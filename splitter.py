
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


def split_documents(docs):

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )

    if not docs:
        texts = []
    elif isinstance(docs[0], Document):
        texts = text_splitter.split_documents(docs)
    else:
        texts = text_splitter.create_documents(docs)

    n_chunks = len(texts)
    print(f"Split into {n_chunks} chunks")
    return texts
