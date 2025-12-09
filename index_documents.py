"""Index source documents and persist in vector embedding database."""

import os
from dotenv import load_dotenv

# Load the environment variables from the .env file
load_dotenv()

from transformers import AutoTokenizer
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.vectorstores import OpenSearchVectorSearch
from langchain.vectorstores.pgvector import PGVector

SOURCE_DOCUMENTS = ["source_documents/tsla-20230930.pdf"]
COLLECTION_NAME = "doc_index"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def main():
    print("Ingesting...")
    all_docs = ingest_docs(SOURCE_DOCUMENTS)
    print("Persisting...")
    db = generate_embed_index(all_docs)
    print("Done.")


def ingest_docs(source_documents):
    all_docs = []
    for source_doc in source_documents:
        print(source_doc)
        docs = pdf_to_chunks(source_doc)
        all_docs = all_docs + docs
    return all_docs


def pdf_to_chunks(pdf_file):
    # Use the tokenizer from the embedding model to determine the chunk size
    # so that chunks don't get truncated.
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer,
        separators=["\n \n", "\n\n", "\n", " ", ""],
        chunk_size=512,
        chunk_overlap=0,
    )
    loader = PyPDFLoader(pdf_file)
    docs = loader.load_and_split(text_splitter)
    return docs


def generate_embed_index(docs):
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    chroma_persist_dir = os.getenv("CHROMA_PERSIST_DIR")
    db = create_index_chroma(docs, embeddings, chroma_persist_dir)
    return db


def create_index_chroma(docs, embeddings, persist_dir):
    db = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=persist_dir,
    )
    db.persist()
    return db


if __name__ == "__main__":
    main()
