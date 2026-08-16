import os
import json
import glob
import sys
import time
from uuid import uuid4

# Check dependencies
try:
    from fastembed import TextEmbedding
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct
except ImportError:
    print("Missing dependencies. Please run: pip install qdrant-client fastembed")
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
    
    # Re-create collection for 384 dimensions (BAAI/bge-small-en-v1.5)
    collections = [c.name for c in q_client.get_collections().collections]
    if COLLECTION_NAME in collections:
        print(f"Deleting existing collection '{COLLECTION_NAME}' to update dimensions...")
        q_client.delete_collection(collection_name=COLLECTION_NAME)
        
    print(f"Creating collection '{COLLECTION_NAME}' (size: 384)...")
    q_client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    )

    # 2. Setup FastEmbed Local Model
    print("Initializing FastEmbed Model (BAAI/bge-small-en-v1.5)...")
    try:
        # This will download the weights on first run, then cache them locally
        model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
    except Exception as e:
        print(f"FAILED to initialize FastEmbed: {e}")
        sys.exit(1)

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
            # Generate local embeddings natively using ONNX! (Yields numpy arrays)
            vectors = list(model.embed(texts_to_embed))
        except Exception as e:
            print(f"Embedding failed on batch {i//BATCH_SIZE + 1}: {e}")
            sys.exit(1)
            
        # Build Qdrant points
        points = []
        for j, chunk in enumerate(batch):
            point_id = str(uuid4())
            points.append(
                PointStruct(
                    id=point_id,
                    vector=vectors[j].tolist(), # Convert numpy array to python list for Qdrant
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
        
    print("Indexing complete!")

if __name__ == "__main__":
    run_indexer()
