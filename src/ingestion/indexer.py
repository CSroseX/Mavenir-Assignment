import os
import json
import glob
import sys
import time
import requests
from uuid import uuid4
from tenacity import retry, wait_exponential, stop_after_attempt

# Check dependencies
try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct
except ImportError:
    print("Missing dependencies. Please run: pip install qdrant-client tenacity requests")
    sys.exit(1)

def format_table_for_embedding(grid):
    # Convert a 2D grid to a markdown-like string for the embedding model
    if not grid:
        return ""
        
    lines = []
    for i, row in enumerate(grid):
        lines.append(" | ".join([str(cell) for cell in row]))
        if i == 0:
            lines.append(" | ".join(["---"] * len(row)))
            
    return "\n".join(lines)

def run_indexer():
    # 1. Connect to Qdrant
    print("Connecting to Qdrant at localhost:6333...")
    try:
        q_client = QdrantClient(url="http://localhost:6333", check_compatibility=False)
        # Test connection
        q_client.get_collections()
    except Exception as e:
        print(f"FAILED to connect to Qdrant: {e}")
        print("Please ensure your Qdrant docker container is running (e.g., docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant)")
        sys.exit(1)
        
    COLLECTION_NAME = "3gpp_specs"
    
    # Re-create collection for 3072 dimensions
    collections = [c.name for c in q_client.get_collections().collections]
    if COLLECTION_NAME in collections:
        print(f"Deleting existing collection '{COLLECTION_NAME}' to update dimensions...")
        q_client.delete_collection(collection_name=COLLECTION_NAME)
        
    print(f"Creating collection '{COLLECTION_NAME}' (size: 3072)...")
    q_client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=3072, distance=Distance.COSINE),
    )

    # 2. Check API Key
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("FAILED: GEMINI_API_KEY environment variable is not set.")
        sys.exit(1)
    
    # STRIP THE ROGUE WHITESPACE!
    api_key = api_key.strip()

    # 3. Load Chunks
    chunk_files = glob.glob("data/chunks/*_chunks.json")
    if not chunk_files:
        print("No chunk files found in data/chunks/")
        sys.exit(1)
        
    all_chunks = []
    for f in chunk_files:
        print(f"Loading {f}...")
        with open(f, 'r', encoding='utf-8') as file:
            all_chunks.extend(json.load(file))
            
    print(f"Loaded {len(all_chunks)} total chunks.")

    # 4. Batch Embed and Index
    BATCH_SIZE = 100
    points_indexed = 0
    
    # Add auto-retry for 429 Quota Exhausted limits
    @retry(wait=wait_exponential(multiplier=2, min=4, max=60), stop=stop_after_attempt(7))
    def embed_batch_with_retry(texts):
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-2:batchEmbedContents?key={api_key}"
        payload = {
            "requests": [
                {
                    "model": "models/gemini-embedding-2",
                    "content": {"parts": [{"text": t}]}
                }
                for t in texts
            ]
        }
        response = requests.post(url, json=payload)
        response.raise_for_status() # Will trigger retry if 429 or 5xx
        return response.json()
    
    for i in range(0, len(all_chunks), BATCH_SIZE):
        batch = all_chunks[i:i + BATCH_SIZE]
        
        texts_to_embed = []
        for chunk in batch:
            if chunk["chunk_type"] == "table":
                embed_text = format_table_for_embedding(chunk["content"])
            else:
                embed_text = chunk["content"]
                
            # Add context to the string to improve embedding quality
            context = f"Spec {chunk['spec_id']} {chunk['version']} Clause {chunk['clause_id']} {chunk['clause_title']}\n"
            texts_to_embed.append(context + embed_text)
            
        print(f"Embedding batch {i//BATCH_SIZE + 1} ({len(batch)} chunks)...")
        
        try:
            # We call the raw REST API to correctly batch embed
            data = embed_batch_with_retry(texts_to_embed)
            embeddings = data.get("embeddings", [])
            
            if len(embeddings) != len(batch):
                raise ValueError(f"Expected {len(batch)} embeddings, got {len(embeddings)}")
                
            vectors = [emb["values"] for emb in embeddings]
        except Exception as e:
            print(f"Embedding failed on batch {i//BATCH_SIZE + 1} after retries: {e}")
            sys.exit(1)
            
        # Build Qdrant points
        points = []
        for j, chunk in enumerate(batch):
            point_id = str(uuid4())
            points.append(
                PointStruct(
                    id=point_id,
                    vector=vectors[j],
                    payload=chunk # The entire chunk dict becomes the payload!
                )
            )
            
        # Upsert to Qdrant
        q_client.upsert(
            collection_name=COLLECTION_NAME,
            points=points
        )
        points_indexed += len(points)
        print(f"  -> Indexed {points_indexed}/{len(all_chunks)} chunks.")
        
        # Tiny sleep to respect rate limits gracefully
        time.sleep(1)
        
    print("Indexing complete!")

if __name__ == "__main__":
    run_indexer()
