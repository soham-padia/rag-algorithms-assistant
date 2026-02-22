"""Build and persist a FAISS index from PDF files."""

from __future__ import annotations

from pathlib import Path

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config


def load_documents(pdf_dir: Path):
    loader = DirectoryLoader(
        str(pdf_dir),
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
        show_progress=True,
    )
    docs = loader.load()
    return docs


def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(documents)


def build_vectorstore(chunks):
    embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)
    return FAISS.from_documents(chunks, embeddings)


def main() -> None:
    if not config.DATA_DIR.exists():
        raise FileNotFoundError(f"PDF directory not found: {config.DATA_DIR}")

    docs = load_documents(config.DATA_DIR)
    if not docs:
        raise ValueError(f"No PDF files found in {config.DATA_DIR}")

    chunks = split_documents(docs)
    vectorstore = build_vectorstore(chunks)

    config.VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(config.VECTORSTORE_DIR))

    print(f"Indexed {len(docs)} documents into {len(chunks)} chunks.")
    print(f"Saved FAISS index to: {config.VECTORSTORE_DIR}")


if __name__ == "__main__":
    main()
