import os
import glob
import subprocess
import json
import random
import re
import sys

print("STEP 1: Cleaning old outputs...")
for f in glob.glob("data/chunks/*.json"):
    os.remove(f)
    print(f"Deleted {f}")

log_file = "scratch/filtered_ambiguous.log"
if os.path.exists(log_file):
    os.remove(log_file)
    print(f"Deleted {log_file}")

print("\nSTEP 2: Re-running pipeline...")
specs = [
    ("ts_123501v171100p.json", "23.501"),
    ("ts_124301v170900p.json", "24.301"),
    ("ts_124501v171200p.json", "24.501")
]

for parsed_name, spec_id in specs:
    parsed_path = os.path.join("data", "parsed", parsed_name)
    tree_path = os.path.join("data", "chunks", f"{spec_id}_tree.json")
    chunk_path = os.path.join("data", "chunks", f"{spec_id}_chunks.json")
    
    print(f"Building tree for {spec_id}...")
    subprocess.run([sys.executable, "src/ingestion/tree_builder.py", parsed_path, tree_path], check=True)
    
    print(f"Chunking {spec_id}...")
    subprocess.run([sys.executable, "src/ingestion/chunker.py", tree_path, chunk_path, spec_id], check=True)

print("Indexing all chunks...")
subprocess.run([sys.executable, "src/ingestion/indexer.py"], check=True)

print("\nSTEP 3: VERIFICATION")
print("1. Boilerplate count check")
exact_junk = ["ETSI", "Contents", "Foreword", "Trademarks", "Copyright Notification", "Intellectual Property Rights", "Essential patents"]

boilerplate_count = 0
total_chunks_per_spec = {}
all_chunks = []
duplicates_set = set()
duplicate_count = 0

chunk_files = glob.glob("data/chunks/*_chunks.json")
for f in chunk_files:
    spec_id = os.path.basename(f).replace("_chunks.json", "")
    with open(f, "r", encoding="utf-8") as file:
        chunks = json.load(file)
        total_chunks_per_spec[spec_id] = len(chunks)
        all_chunks.extend(chunks)
        
        for chunk in chunks:
            text = chunk.get("text", "")
            
            # Count duplicates based on text content exactly
            if text in duplicates_set:
                duplicate_count += 1
            else:
                duplicates_set.add(text)
            
            # Check for boilerplate patterns in the text
            is_bp = False
            for line in text.split("\n"):
                line = line.strip()
                if line in exact_junk:
                    is_bp = True
                if re.match(r'^3GPP TS \d+\.\d+ version \d+\.\d+\.\d+ Release \d+$', line, re.IGNORECASE):
                    is_bp = True
                if re.match(r'^ETSI TS \d+\s+\d+\s+V\d+\.\d+\.\d+\s+\(\d{4}-\d{2}\)$', line, re.IGNORECASE):
                    is_bp = True
                if re.match(r'^\d{1,4}$', line):
                    is_bp = True
                if re.match(r'^The present document.*ETSI.*$', line, re.IGNORECASE):
                    is_bp = True
                if re.match(r'^3GPP TS \d+\.\d+ V\d+\.\d+\.\d+ \(\d{4}-\d{2}\)$', line, re.IGNORECASE):
                    is_bp = True
            
            if is_bp:
                boilerplate_count += 1

print(f"-> Chunks containing boilerplate patterns: {boilerplate_count}")

print(f"\n2. Duplicate chunks across corpus: {duplicate_count}")

print("\n3. Total local chunk count vs Qdrant point count:")
total_local = sum(total_chunks_per_spec.values())
print(f"-> Local total: {total_local}")
try:
    from qdrant_client import QdrantClient
    q_client = QdrantClient(url="http://localhost:6333", check_compatibility=False)
    count_info = q_client.count(collection_name="3gpp_specs")
    print(f"-> Qdrant total: {count_info.count}")
except Exception as e:
    print(f"-> Qdrant count failed: {e}")

print("\n4. 5 Random text chunks:")
random.seed(42)
random_chunks = random.sample([c for c in all_chunks if c.get('text')], min(5, len(all_chunks)))
for i, c in enumerate(random_chunks):
    print(f"\n--- CHUNK {i+1} [{c.get('metadata', {}).get('spec', 'unknown')}] ---")
    print(c.get("text"))

print("\n5. filtered_ambiguous.log status:")
if os.path.exists(log_file):
    with open(log_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
        print(f"-> Line count: {len(lines)}")
        if len(lines) > 0:
            print("-> Samples:")
            for l in lines[:5]:
                print(l.strip())
else:
    print("-> Line count: 0 (File does not exist, as expected)")

print("\n6. Chunk count per spec:")
for spec_id, count in sorted(total_chunks_per_spec.items()):
    print(f"-> {spec_id}: {count}")
print(f"-> Total (new): {total_local} (Old total was 4,429)")
