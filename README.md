# 3GPP RAG Assistant

A retrieval-augmented generation system designed to answer telecom technical questions grounded in 3GPP specifications.

---

## Overview

The system processes 3GPP technical specifications (such as TS 23.501, TS 24.301, and TS 24.501), indexes structured text and tables into a vector database, and executes a multi-stage retrieval and verification pipeline to generate grounded answers to technical questions.

---

## Architecture

The system consists of four primary components:

1. **Ingestion & Indexing Pipeline (`src/ingestion/`)**:
   - **Document Parsing (`parse.py`)**: Extracts text, headings, and table structures from specification PDFs using Docling.
   - **Hierarchy Builder (`tree_builder.py`)**: Parses clause numbers and section hierarchy to build structured clause trees with complete ancestral lineage.
   - **Clause-Aware Chunker (`chunker.py`)**: Chunks text with soft and hard token limits while preserving clause metadata and keeping tables intact.
   - **Vector Indexer (`indexer.py`)**: Computes dense vector embeddings using `BAAI/bge-small-en-v1.5` via FastEmbed and indexes vectors and payloads into a Qdrant vector database.

2. **Dual-Path Retrieval Engine (`src/retrieval/retriever.py`)**:
   - **Query Routing**: Uses regex parsing to detect explicit specification numbers (e.g., TS 23.501) or clause references (e.g., clause 5.5.1).
   - **Path A (Semantic Search)**: Executes global dense semantic retrieval across the entire indexed corpus in Qdrant when no explicit target is present.
   - **Path B (Targeted Metadata Filter)**: Applies Qdrant payload filters matching the identified `spec_id` and `clause_id` hierarchy.
   - **Cross-Encoder Reranking**: Reranks the top 20 retrieved chunks using `cross-encoder/ms-marco-MiniLM-L-6-v2` to select the top 5 chunks.

3. **Generation & Verification Graph (`src/generation/`)**:
   - **Orchestration (`graph.py`)**: Implements a LangGraph state machine coordinating generation, claim verification, and conditional retry loops.
   - **Generation Node (`nodes.py`)**: Prompts the LLM to generate domain-expert answers grounded strictly in retrieved context.
   - **Verification Node (`nodes.py`)**: Evaluates generated claims against source chunks via structured JSON output, identifying unsupported assertions.
   - **Correction Loop**: Feeds unsupported claim feedback back to the generator for up to 3 revision attempts before outputting the final response.

4. **Interface & Evaluation (`app.py`, `eval/`)**:
   - **Streamlit App (`app.py`)**: Interactive chat interface displaying answers, grounded status badges, retries, and retrieved source chunks, alongside an evaluation dashboard.
   - **Evaluation Suite (`eval/run_eval.py`)**: Evaluates queries against an evaluation set (`eval/eval_set.json`), recording verdicts and tracking verification metrics.

---

## Design Decisions

- **Modular State Graph**: Generation and verification are isolated into distinct LangGraph nodes, allowing independent prompt tuning, model configuration, and deterministic retry routing.
- **Table Preservation**: Tables are preserved in full grid structure without splitting across chunks to maintain relational context across technical data.
- **Lineage-Enriched Context**: Chunks include complete hierarchical lineage headers (e.g., Spec, Clause ID, Clause Title), ensuring unambiguous context during embedding and LLM generation.
- **Two-Stage Retrieval**: Dense vector retrieval fetches the top 20 retrieved chunks from Qdrant, followed by cross-encoder reranking to improve relevance for complex technical procedures.
- **Query Targeting**: Queries containing specific specification or clause identifiers route directly to filtered subset queries, narrowing the search space prior to semantic scoring.

---

## Hallucination Mitigation

- **Strict Judge Verification**: A dedicated verification step fact-checks each claim against the retrieved chunks, failing any response that contains claims not entailed by the context.
- **Iterative Feedback Loop**: Unsupported claims identified by the verifier are provided directly back to the generator as revision feedback.
- **Constrained Prompts**: System prompts instruct the LLM to refuse fabrication, avoid assuming details not present in context, and state explicitly when a technical detail is not covered by the specifications.
- **Leak Prevention**: Post-processing strips internal reasoning tags and checks for leaked retrieval artifacts.

---

## Setup

### Prerequisites

- Python 3.9 or higher
- Docker (for running Qdrant)
- OpenAI-compatible API endpoint (e.g., Groq) or local LLM server (e.g., Ollama)

### 1. Environment Setup

Create and activate a virtual environment, then install dependencies:

```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Configuration

Copy the example environment file and configure the variables:

```bash
cp .env.example .env
```

Set the required variables in `.env`:

```env
USE_REMOTE_LLM=true
OPENAI_API_KEY=your_api_key_here
OPENAI_BASE_URL=https://api.groq.com/openai/v1
MODEL_NAME=qwen/qwen3.6-27b
```

### 3. Vector Database

Start a local Qdrant instance:

```bash
docker run -d -p 6333:6333 -p 6334:6334 qdrant/qdrant
```

Index the specifications into Qdrant:

```bash
python src/ingestion/indexer.py
```

### 4. Running the Application

Launch the Streamlit web interface:

```bash
streamlit run app.py
```

### 5. Running the Evaluation Suite

Run the evaluation script against the test benchmark:

```bash
python eval/run_eval.py
```

---

## Limitations

- **Chunk Boundary Splits**: Multi-page procedural flows spanning multiple clauses may be separated across chunk boundaries.
- **General-Purpose Embeddings**: Out-of-the-box embedding and reranking models may misinterpret niche telecom acronyms that occur across different functional layers.
- **Aggressive Verification**: The strict verification judge may occasionally flag valid technical deductions as unsupported if the exact wording is absent from the retrieved chunks.
