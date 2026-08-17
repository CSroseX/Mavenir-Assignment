# 3GPP RAG Assistant

A retrieval-augmented generation system designed to accurately answer telecom questions using 3GPP specifications.

## Tech Stack

* Python for core logic
* Streamlit for the chat interface
* Qdrant for vector storage and semantic search
* FastEmbed (BAAI/bge-small-en-v1.5) for fast local embeddings
* Sentence-Transformers (ms-marco-MiniLM-L-6-v2) for cross-encoder reranking
* LangGraph for orchestrating the generation and verification workflow
* Groq for high-speed LLM inference

## Decisions Made

We adopted a modular agent architecture using LangGraph. This allowed us to separate the generation logic from the verification logic. We also implemented a dual-path retrieval system. If a user explicitly mentions a specification or clause in their query, the system uses Qdrant metadata filters to narrow the search space before falling back to semantic search. 

We used a two-stage retrieval pipeline. The initial retrieval fetches a wide net of 20 candidates using dense embeddings. These are then reranked locally using a cross-encoder to select the final top 5 chunks. This massively improves context relevance for complex telecom procedures.

## Solutions Employed to Minimize Hallucinations

We implemented a strict verification node that acts as an impartial judge. It evaluates the generated answer against the retrieved chunks. If any claim is unsupported by the text, the system fails the verification. The system will then retry generation with explicit feedback about what claims failed. 

We also disabled the reasoning effort on the LLM to prevent it from leaking internal thought processes into the final output. The system prompt strictly prohibits the LLM from mentioning that it is a retrieval system or using citation brackets. If the context is missing, the LLM is instructed to state plainly that the specifications do not cover the detail.

## How to Run

### Prerequisites
* Python 3.9 or higher
* Docker Desktop (for running Qdrant)
* Groq API Key

### 1. Setup the Environment
Create and activate a virtual environment. Install the dependencies.

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure Environment Variables
Create a `.env` file in the root directory and add your API keys.

```env
USE_REMOTE_LLM=true
OPENAI_API_KEY=your_groq_api_key_here
OPENAI_BASE_URL=https://api.groq.com/openai/v1
MODEL_NAME=qwen/qwen3.6-27b
```

### 3. Start Qdrant
Run the Qdrant vector database locally using Docker.

```bash
docker run -p 6333:6333 qdrant/qdrant
```

### 4. Run the Evaluation Suite
You can run the automated evaluation script to test the system against the predefined dataset. 

```bash
python eval/run_eval.py
```

### 5. Launch the Application
Start the Streamlit interface to interact with the assistant.

```bash
streamlit run app.py
```

## Limitations

The semantic search relies on a general purpose embedding model. This model might struggle to differentiate between highly specific telecom acronyms that appear in similar contexts. The system is also limited by the chunk size. Very long procedures that span multiple pages might be split across chunks. This makes it difficult for the LLM to piece together the full workflow. Finally, the strict verification judge can sometimes be overly aggressive and flag valid technical deductions as unsupported claims.
