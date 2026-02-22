"""Build and persist a FAISS index from PDF files."""

from __future__ import annotations

from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config


def load_documents(pdf_dir: Path) -> tuple[list, list[tuple[Path, str]]]:
    docs = []
    failures: list[tuple[Path, str]] = []

    for pdf_path in sorted(pdf_dir.rglob("*.pdf")):
        try:
            loader = PyPDFLoader(str(pdf_path))
            docs.extend(loader.load())
        except Exception as exc:
            failures.append((pdf_path, str(exc)))

    return docs, failures


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


def collect_documents() -> tuple[list, list[tuple[Path, str]]]:
    input_dirs = [config.DATA_DIR, config.PROFESSOR_DATA_DIR]
    docs = []
    failures: list[tuple[Path, str]] = []

    for directory in input_dirs:
        if directory.exists():
            current_docs, current_failures = load_documents(directory)
            docs.extend(current_docs)
            failures.extend(current_failures)

    return docs, failures


def main() -> None:
    if not config.DATA_DIR.exists() and not config.PROFESSOR_DATA_DIR.exists():
        raise FileNotFoundError(
            "No input directories found. Expected at least one of: "
            f"{config.DATA_DIR} or {config.PROFESSOR_DATA_DIR}"
        )

    docs, failures = collect_documents()
    if not docs:
        raise ValueError(
            "No PDF files found. Add files under: "
            f"{config.DATA_DIR} and/or {config.PROFESSOR_DATA_DIR}"
        )

    chunks = split_documents(docs)
    vectorstore = build_vectorstore(chunks)

    config.VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(config.VECTORSTORE_DIR))

    if failures:
        print("Skipped unreadable PDFs:")
        for path, error in failures:
            print(f"- {path}: {error}")

    print(f"Indexed {len(docs)} pages into {len(chunks)} chunks.")
    print(f"Saved FAISS index to: {config.VECTORSTORE_DIR}")


if __name__ == "__main__":
    main()
