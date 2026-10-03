import glob
import hashlib
import json
import sys

# Check dependencies
try:
    from fastembed import TextEmbedding
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, PointStruct, VectorParams
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
    chunk_dir = cfg.project_root / "data" / "chunks"
    chunk_files = sorted(glob.glob(str(chunk_dir / "*_chunks.json")))
    if not chunk_files:
        print(f"No chunk files found in {chunk_dir}")
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

    # Deterministic chunk_id: sha1(spec_id|clause_id|ordinal), where ordinal
    # is the chunk's position among chunks sharing the same (spec_id,
    # clause_id) — clause_id alone is not unique (e.g. frontmatter chunks
    # all share clause_id "0"). Re-running the indexer over the same chunk
    # files yields identical ids, which snapshot-based CI depends on.
    clause_ordinals = {}
    for chunk in all_chunks:
        key = (chunk["spec_id"], chunk["clause_id"])
        ordinal = clause_ordinals.get(key, 0)
        clause_ordinals[key] = ordinal + 1
        digest = hashlib.sha1(
            f"{chunk['spec_id']}|{chunk['clause_id']}|{ordinal}".encode("utf-8")
        ).hexdigest()
        chunk["chunk_id"] = digest
        # Qdrant point ids must be an unsigned int or a UUID string; derive a
        # stable UUID from the same digest so the point id is reproducible.
        chunk["_point_id"] = f"{digest[:8]}-{digest[8:12]}-{digest[12:16]}-{digest[16:20]}-{digest[20:32]}"

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
            point_id = chunk.pop("_point_id")
            points.append(
                PointStruct(
                    id=point_id,
                    vector=vectors[j].tolist(), # Convert numpy array to python list for Qdrant
                    payload=chunk # The entire chunk dict (incl. deterministic chunk_id) becomes the payload!
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
