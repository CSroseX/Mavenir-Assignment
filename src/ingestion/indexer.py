import json
import glob
import sys
from uuid import uuid4

# Check dependencies
try:
    from fastembed import TextEmbedding
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct
except ImportError:
    print("Missing dependencies. Please run: pip install qdrant-client fastembed")
    sys.exit(1)

from src.config import get_settings

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
    cfg = get_settings()

    # 1. Connect to Qdrant
    print(f"Connecting to Qdrant at {cfg.qdrant_url}...")
    try:
        q_client = QdrantClient(url=cfg.qdrant_url, check_compatibility=False)
        # Test connection
        q_client.get_collections()
    except Exception as e:
        print(f"FAILED to connect to Qdrant: {e}")
        print("Please ensure your Qdrant docker container is running (e.g., docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant)")
        sys.exit(1)

    collection_name = cfg.qdrant_collection

    # Re-create collection for configured dimensions
    collections = [c.name for c in q_client.get_collections().collections]
    if collection_name in collections:
        print(f"Deleting existing collection '{collection_name}' to update dimensions...")
        q_client.delete_collection(collection_name=collection_name)

    print(f"Creating collection '{collection_name}' (size: {cfg.vector_size})...")
    q_client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=cfg.vector_size, distance=Distance.COSINE),
    )

    # 2. Setup FastEmbed Local Model
    print(f"Initializing FastEmbed Model ({cfg.embedding_model})...")
    try:
        # This will download the weights on first run, then cache them locally
        model = TextEmbedding(model_name=cfg.embedding_model)
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
    batch_size = cfg.indexer_batch_size
    points_indexed = 0

    for i in range(0, len(all_chunks), batch_size):
        batch = all_chunks[i:i + batch_size]

        texts_to_embed = []
        for chunk in batch:
            if chunk["chunk_type"] == "table":
                embed_text = format_table_for_embedding(chunk["content"])
            else:
                embed_text = chunk["content"]

            # Add context to the string to improve embedding quality
            context = f"Spec {chunk['spec_id']} {chunk['version']} Clause {chunk['clause_id']} {chunk['clause_title']}\n"
            texts_to_embed.append(context + embed_text)

        print(f"Embedding batch {i//batch_size + 1} ({len(batch)} chunks)...")

        try:
            # Generate local embeddings natively using ONNX! (Yields numpy arrays)
            vectors = list(model.embed(texts_to_embed))
        except Exception as e:
            print(f"Embedding failed on batch {i//batch_size + 1}: {e}")
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
            collection_name=collection_name,
            points=points
        )
        points_indexed += len(points)
        print(f"  -> Indexed {points_indexed}/{len(all_chunks)} chunks.")

    print("Indexing complete!")

if __name__ == "__main__":
    run_indexer()
