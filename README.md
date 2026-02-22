# RAG-Based Algorithms Study Assistant

A retrieval-augmented generation (RAG) system for contextualized reasoning over algorithms textbooks. The system lets users ask natural language questions about algorithms and data structures, retrieves relevant passages from 500+ pages of source material, and generates grounded answers using LLM reasoning over the retrieved context.

Built as a study tool for CS5800 (Algorithms) at Northeastern University.

## Demo

<!-- Add a screenshot or GIF of the Gradio interface here -->
<!-- ![demo](assets/demo.png) -->

## How It Works

```text
User Query
    |
    v
+----------------+
| Embedding      |  Sentence-Transformers
| Model          |
+--------+-------+
         |
         | query vector
         v
+----------------+
| FAISS Index    |  Vector similarity search
| (500+ pages)   |
+--------+-------+
         |
         | top-k chunks
         v
+----------------+
| Prompt         |  Context-injected template
| Construction   |
+--------+-------+
         |
         v
+----------------+
| LLM via        |  LangChain
| LangChain      |
+--------+-------+
         |
         v
Answer (with source references)
```

### Pipeline Stages

1. **Document Ingestion**: PDF parsing and text extraction from algorithms textbooks (CLRS, Kleinberg-Tardos).
2. **Chunking**: Documents are split into overlapping chunks. Multiple chunking strategies were tested (fixed-size, recursive, semantic) to balance retrieval precision with context completeness.
3. **Embedding**: Each chunk is embedded using a Sentence-Transformers model. Multiple embedding models were evaluated for retrieval quality on algorithms-specific queries.
4. **Indexing**: Embeddings are stored in a FAISS vector index for fast approximate nearest neighbor search.
5. **Retrieval**: Given a user query, the system embeds the query and retrieves the top-k most similar chunks from the FAISS index.
6. **Generation**: Retrieved chunks are injected into a prompt template designed to ground the LLM response in the source material and reduce hallucination. The LLM generates an answer via LangChain.
7. **Math Repair Pass**: If math is detected, a second formatting pass repairs Markdown/LaTeX delimiters to improve render quality in the UI.
8. **Visual Summary**: The app extracts essential points from the final answer and renders an SVG summary snapshot card.

## Features

- **Contextual Q&A** over 500+ pages of algorithms content (CLRS, Kleinberg-Tardos)
- **Document chunking pipeline** with configurable chunk size, overlap, and strategy
- **FAISS vector search** for fast retrieval over embedded passages
- **Context-injected prompt templates** to reduce hallucination and improve answer grounding
- **Chat mode** with conversation context for follow-up questions
- **Enter-to-send + button send** interaction in Gradio
- **Streaming responses** with live rendering
- **Loading/status indicator** (`Loading retrieved context...`, `Generating response...`, `Ready.`)
- **Math-aware output** with Markdown + LaTeX delimiters and second-pass formatting repair
- **SVG summary renderer** that generates an "Essential Summary" card per response
- **Modular design**: swap embedding models, chunking strategies, or LLMs independently

## Project Structure

```text
rag-algorithms-assistant/
|-- app.py                  # Gradio interface and main entry point
|-- ingest.py               # Document loading, chunking, and embedding pipeline
|-- retriever.py            # FAISS index building and query retrieval
|-- chain.py                # LangChain prompt templates and LLM chain
|-- config.py               # Centralized configuration (paths, model names, parameters)
|-- requirements.txt        # Python dependencies
|-- .gitignore              # Ignores local index + private class materials
|-- data/
|   |-- pdfs/               # Source textbook PDFs (not included in repo)
|   `-- professor_materials/ # Private class/professor content (gitignored)
|-- vectorstore/            # Persisted FAISS index (generated after ingestion)
`-- README.md
```

## Setup

### Prerequisites

- Python 3.10+
- Apple Silicon Mac supported (M-series; uses PyTorch MPS when available)
- Optional: a Hugging Face token (`HF_TOKEN`) only if you switch to a gated model

### Installation

```bash
git clone https://github.com/soham-padia/rag-algorithms-assistant.git
cd rag-algorithms-assistant
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Configuration

Default generation model is `Qwen/Qwen2.5-3B-Instruct` (free/public on Hugging Face), so no token is required by default.

If you choose a gated/private Hugging Face model, set:

```bash
export HF_TOKEN="your-huggingface-token"
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

Adjust parameters in `config.py`:

```python
# config.py
CHUNK_SIZE = 512          # characters per chunk
CHUNK_OVERLAP = 64        # overlap between consecutive chunks
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "Qwen/Qwen2.5-3B-Instruct"
TOP_K = 5                 # number of chunks to retrieve per query
DATA_DIR = "data/pdfs"
PROFESSOR_DATA_DIR = "data/professor_materials"
VECTORSTORE_DIR = "vectorstore"
```

### Ingestion

Place your PDF files in `data/pdfs/` and/or `data/professor_materials/`, then run:

```bash
python ingest.py
```

This parses the PDFs, chunks the text, computes embeddings, and saves the FAISS index to `vectorstore/`.

### Running the App

```bash
python app.py
```

This launches the Gradio interface at `http://localhost:7860`.

UI behavior:
- Press **Enter** in the message box to submit
- Or click **Send**
- Use **Clear** to reset chat and summary snapshot

## Usage Examples

**Example queries:**

- "Explain the master theorem and when each case applies."
- "What is the difference between Dijkstra's and Bellman-Ford?"
- "Walk me through the proof that comparison-based sorting is Omega(n log n)."
- "How does dynamic programming apply to the knapsack problem?"

The system retrieves relevant passages from the source textbooks and generates answers grounded in the retrieved context.
The app then appends canonical source paths/pages and generates an SVG summary of key points.

## Design Decisions and Iterations

### Chunking Strategy

Fixed-size chunking with overlap was used as the baseline. Recursive chunking (splitting on paragraph and sentence boundaries) improved retrieval quality for proof-heavy content where splitting mid-sentence degraded context. A chunk size of 512 characters with 64-character overlap was selected after testing sizes from 256 to 1024.

### Embedding Model Selection

Several Sentence-Transformers models were evaluated:

- `all-MiniLM-L6-v2`: Good balance of speed and quality; selected as default.
- `all-mpnet-base-v2`: Slightly better retrieval quality but slower embedding time.

Model choice was evaluated by manually checking retrieval relevance on a set of 20+ test queries spanning graph algorithms, dynamic programming, sorting, and complexity theory.

### Prompt Design

Context-injected prompt templates instruct the LLM to:

- Answer based only on the provided passages
- Cite which passage(s) support the answer
- Say "I don't have enough context" rather than hallucinate

This significantly reduced hallucination on out-of-scope queries (e.g., questions about topics not covered in the source PDFs).

## Tech Stack

| Component     | Tool                  |
|---------------|-----------------------|
| Orchestration | LangChain             |
| Embeddings    | Sentence-Transformers |
| Vector Store  | FAISS                 |
| LLM           | Hugging Face Transformers (local, via LangChain) |
| Interface     | Gradio                |
| PDF Parsing   | PyPDFLoader (LangChain)|
| Language      | Python                |

## Requirements

```text
langchain
langchain-huggingface
faiss-cpu
sentence-transformers
transformers
torch
gradio
pypdf
```

## Privacy Notes

- `data/professor_materials/` is intended for private class content.
- Files in `data/professor_materials/` are ignored via `.gitignore` so they are not committed.
- `vectorstore/` is also ignored to avoid committing generated local index artifacts.

## Future Work

- Add evaluation metrics (retrieval precision@k, answer faithfulness) using RAGAS or similar
- Experiment with reranking retrieved passages before generation
- Support additional textbooks and lecture notes
- Add conversation memory for multi-turn follow-up questions
- Fine-tune embedding model on algorithms-specific vocabulary

## License

MIT

## Author

**Soham Padia**
- Website: [soham-padia.github.io](https://soham-padia.github.io)
- LinkedIn: [linkedin.com/in/soham-padia](https://www.linkedin.com/in/soham-padia/)
- Email: padia.so@northeastern.edu
