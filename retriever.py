"""Retriever utilities for loading and querying the FAISS index."""

from __future__ import annotations

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

import config


def load_vectorstore() -> FAISS:
    if not config.VECTORSTORE_DIR.exists():
        raise FileNotFoundError(
            f"Vectorstore not found at {config.VECTORSTORE_DIR}. Run ingest.py first."
        )

    embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)
    return FAISS.load_local(
        str(config.VECTORSTORE_DIR),
        embeddings,
        allow_dangerous_deserialization=True,
    )


def get_retriever(k: int | None = None):
    vectorstore = load_vectorstore()
    return vectorstore.as_retriever(search_kwargs={"k": k or config.TOP_K})
