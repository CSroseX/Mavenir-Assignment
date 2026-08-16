import json
import re
import glob
import random
from qdrant_client import QdrantClient

chunk_files = glob.glob("data/chunks/*_chunks.json")
all_chunks = []
for f in chunk_files:
    with open(f, 'r', encoding='utf-8') as file:
        all_chunks.extend(json.load(file))

# 1. Count boilerplate
patterns = [
    r'ETSI',
    r'3GPP TS \d+\.\d+ version',
    r'Release 17',
    r'^\s*\d+\s*$'
]
combined_pattern = re.compile('|'.join(patterns), re.MULTILINE)
boilerplate_count = sum(1 for c in all_chunks if isinstance(c.get('content'), str) and combined_pattern.search(c.get('content')))
print(f"1. Chunks containing boilerplate patterns: {boilerplate_count}")

# 2. Count duplicates
seen = set()
duplicates = 0
for c in all_chunks:
    if c.get("chunk_type") == "text":
        content = c.get("content", "").strip()
        if content in seen:
            duplicates += 1
        else:
            seen.add(content)
print(f"2. Duplicate text chunks across all specs: {duplicates}")

# 3. Qdrant point count vs total chunk count
try:
    q_client = QdrantClient(url="http://localhost:6333", check_compatibility=False)
    q_info = q_client.get_collection("3gpp_specs")
    print(f"3. Consistency Check: Total local chunks = {len(all_chunks)} | Qdrant Points = {q_info.points_count}")
    if len(all_chunks) == q_info.points_count:
        print("   -> MATCH CONFIRMED.")
    else:
        print("   -> MISMATCH!")
except Exception as e:
    print(f"3. Qdrant Error: {e}")

# 4. 5 random text chunks
print("\n4. Five Random Text Chunks for Visual Confirmation:")
text_chunks = [c for c in all_chunks if c.get("chunk_type") == "text"]
random_5 = random.sample(text_chunks, 5)
for i, c in enumerate(random_5):
    print(f"\n--- Chunk {i+1} [ID: {c.get('clause_id')}] ---")
    print(c.get('content', '')[:1000])
