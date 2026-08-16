import os
import glob
import json
import random
import re

print("VERIFICATION")
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
            text = ""
            if chunk["chunk_type"] == "text":
                text = chunk.get("content", "")
            elif chunk["chunk_type"] == "table":
                # Convert list to string for duplicate check
                text = str(chunk.get("content", ""))
                
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
random.seed(43)
text_chunks = [c for c in all_chunks if c.get("chunk_type") == "text"]
random_chunks = random.sample(text_chunks, min(5, len(text_chunks)))
for i, c in enumerate(random_chunks):
    print(f"\n--- CHUNK {i+1} [{c.get('spec_id', 'unknown')}] ---")
    print(c.get("content"))

print("\n5. filtered_ambiguous.log status:")
log_file = "scratch/filtered_ambiguous.log"
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
