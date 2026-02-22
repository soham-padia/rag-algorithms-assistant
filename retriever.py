"""Retriever utilities for loading and querying the FAISS index."""

from __future__ import annotations

from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

import config


def _index_files_present() -> bool:
    return (
        (Path(config.VECTORSTORE_DIR) / "index.faiss").exists()
        and (Path(config.VECTORSTORE_DIR) / "index.pkl").exists()
    )


def startup_health_message() -> str:
    if _index_files_present():
        return "Startup health: vector index found. Ready for fast responses."

    pdf_count = 0
    for root in [Path(config.DATA_DIR), Path(config.PROFESSOR_DATA_DIR)]:
        if root.exists():
            pdf_count += len(list(root.rglob("*.pdf")))

    if pdf_count == 0:
        return (
            "Startup health: no vector index and no PDFs found. "
            "Add PDFs to data folders, then send a query."
        )

    return (
        f"Startup health: vector index missing. "
        f"Will auto-index {pdf_count} PDF(s) on first query."
    )


def ensure_vectorstore() -> None:
    if _index_files_present():
        return

    from ingest import build_vectorstore, collect_documents, split_documents

    docs, failures = collect_documents()
    if not docs:
        raise FileNotFoundError(
            "No vectorstore found and no PDFs available to build one. "
            f"Add PDFs under {config.DATA_DIR} and/or {config.PROFESSOR_DATA_DIR}."
        )

    chunks = split_documents(docs)
    vectorstore = build_vectorstore(chunks)

    Path(config.VECTORSTORE_DIR).mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(config.VECTORSTORE_DIR))

    if failures:
        print("Skipped unreadable PDFs while auto-building vectorstore:")
        for path, error in failures:
            print(f"- {path}: {error}")

    print(f"Auto-built vectorstore at: {config.VECTORSTORE_DIR}")


def load_vectorstore() -> FAISS:
    ensure_vectorstore()

    embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)
    return FAISS.load_local(
        str(config.VECTORSTORE_DIR),
        embeddings,
        allow_dangerous_deserialization=True,
    )


def get_retriever(k: int | None = None):
    vectorstore = load_vectorstore()
    return vectorstore.as_retriever(search_kwargs={"k": k or config.TOP_K})
