import os
import glob
import json
import random

chunk_files = glob.glob("data/chunks/*_chunks.json")

all_texts = {} # text -> list of (spec_id, clause_id, chunk_type)
duplicates = []

for f in chunk_files:
    spec_id = os.path.basename(f).replace("_chunks.json", "")
    with open(f, "r", encoding="utf-8") as file:
        chunks = json.load(file)
        
        for chunk in chunks:
            if chunk["chunk_type"] == "text":
                text = chunk.get("content", "").strip()
            else:
                text = str(chunk.get("content", ""))
                
            if not text:
                continue
                
            if text in all_texts:
                all_texts[text].append((spec_id, chunk["clause_id"], chunk["chunk_type"]))
            else:
                all_texts[text] = [(spec_id, chunk["clause_id"], chunk["chunk_type"])]

# Filter to texts that appear more than once
for text, occurrences in all_texts.items():
    if len(occurrences) > 1:
        duplicates.append((text, occurrences))

random.seed(42)
sample_dups = random.sample(duplicates, min(5, len(duplicates)))

print(f"Total unique duplicated blocks: {len(duplicates)}")
print("\n--- 5 RANDOM DUPLICATE PAIRS ---")

for i, (text, occurrences) in enumerate(sample_dups):
    print(f"\n[DUPLICATE {i+1}]")
    
    # Check if they are actually cross-spec
    specs = list(set([occ[0] for occ in occurrences]))
    if len(specs) > 1:
        print(f"Type: CROSS-SPEC REPEAT ({', '.join(specs)})")
    else:
        print(f"Type: INTRA-SPEC REPEAT (within {specs[0]})")
        
    print("Found in:")
    for occ in occurrences:
        print(f"  - Spec: {occ[0]}, Clause: {occ[1]} ({occ[2]})")
        
    print(f"Text:\n{text}")
    print("-" * 50)
