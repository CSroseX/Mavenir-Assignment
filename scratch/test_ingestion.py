import subprocess
import sys
import glob
import json
import os
import shutil

def run_step(command):
    print(f"\n[{' '.join(command)}]")
    subprocess.run(command, check=True)

def main():
    print("=== 3GPP RAG Ingestion Tester ===")
    
    # 1. Clean old data
    print("\n--- Cleaning old chunks & trees ---")
    old_files = glob.glob("data/chunks/*")
    for f in old_files:
        os.remove(f)
        print(f"Removed {f}")
        
    # 2. Re-run pipeline for all parsed JSONs
    parsed_files = glob.glob("data/parsed/*.json")
    if not parsed_files:
        print("No parsed JSON files found in data/parsed/!")
        sys.exit(1)
        
    # Spec mapping for naming output files
    spec_mapping = {
        "ts_123501v171100p.json": "23.501",
        "ts_124301v170900p.json": "24.301",
        "ts_124501v171200p.json": "24.501"
    }

    print("\n--- Running Tree Builder & Chunker ---")
    for parsed_path in parsed_files:
        basename = os.path.basename(parsed_path)
        spec_id = spec_mapping.get(basename, basename.split("v")[0].replace("ts_", ""))
        
        tree_path = f"data/chunks/{spec_id}_tree.json"
        chunk_path = f"data/chunks/{spec_id}_chunks.json"
        
        # Build tree
        run_step([sys.executable, "src/ingestion/tree_builder.py", parsed_path, tree_path])
        
        # Chunk (Crucially, pass the spec_id to avoid the default fallback bug!)
        run_step([sys.executable, "src/ingestion/chunker.py", tree_path, chunk_path, spec_id])

    # 3. Index into Qdrant
    print("\n--- Running Indexer (Qdrant & FastEmbed) ---")
    # Note: indexer.py automatically wipes the collection and rebuilds it.
    run_step([sys.executable, "src/ingestion/indexer.py"])
    
    # 4. Final Audit
    print("\n--- Ingestion Audit ---")
    total_local = 0
    spec_counts = {}
    chunk_files = glob.glob("data/chunks/*_chunks.json")
    for f in chunk_files:
        spec = os.path.basename(f).replace("_chunks.json", "")
        with open(f, "r", encoding="utf-8") as file:
            c = json.load(file)
            total_local += len(c)
            spec_counts[spec] = len(c)
            
    print("Local JSON Counts:")
    for spec, count in spec_counts.items():
        print(f"  - {spec}: {count} chunks")
    print(f"  Total Local: {total_local}")
    
    try:
        from qdrant_client import QdrantClient
        q_client = QdrantClient(url="http://localhost:6333", check_compatibility=False)
        count_info = q_client.count(collection_name="3gpp_specs")
        print(f"\nQdrant Total Points: {count_info.count}")
        
        if count_info.count == total_local:
            print("SUCCESS: Local chunk count matches Qdrant points.")
        else:
            print("WARNING: Qdrant count does not match local count!")
            
    except Exception as e:
        print(f"Could not connect to Qdrant to verify counts: {e}")

if __name__ == "__main__":
    main()
