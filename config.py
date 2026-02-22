"""Centralized configuration for the RAG study assistant."""

from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "pdfs"
PROFESSOR_DATA_DIR = BASE_DIR / "data" / "professor_materials"
VECTORSTORE_DIR = BASE_DIR / "vectorstore"

# Chunking
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64

# Retrieval
TOP_K = 5

# Models
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
MAX_NEW_TOKENS = 512
TEMPERATURE = 0.0

# Prompting
SYSTEM_PROMPT = (
    "You are an algorithms study assistant. Use only the provided context to answer. "
    "If the context is insufficient, say: 'I don't have enough context.' "
    "Write in clean Markdown. For math, use LaTeX with $...$ for inline math and $$...$$ "
    "for display math. Explain symbols when needed. "
    "Use inline citation ids like [1], [2] in sentences, but do not output a separate "
    "'Sources' section."
)
