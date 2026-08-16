import json

tree_path = "data/chunks/ts_123501v171100p_tree.json"
with open(tree_path, 'r', encoding='utf-8') as f:
    clauses = json.load(f)

target_ids = ["5", "5.3", "5.3.4", "4.3.3", "4.4.2", "3"]

print("--- RAW CLAUSE CONTENTS ---")
for cid in target_ids:
    c = next((x for x in clauses if x.get('clause_id') == cid), None)
    if c:
        print(f"\n[{cid}]")
        print(json.dumps(c.get('content', []), indent=2))
    else:
        print(f"\n[{cid}] NOT FOUND IN TREE")

# Check clustering of the 179 zero-chunk clause IDs
chunk_path = "data/chunks/ts_123501v171100p_chunks.json"
with open(chunk_path, 'r', encoding='utf-8') as f:
    chunks = json.load(f)

tree_ids = [c.get('clause_id') for c in clauses]
chunk_ids = set([c.get('clause_id') for c in chunks])
zero_chunk_ids = set([cid for cid in tree_ids if cid not in chunk_ids])

print(f"\n--- CLUSTERING ANALYSIS ---")
print(f"Total zero-chunk clauses: {len(zero_chunk_ids)}")

parent_is_also_zero = 0
for cid in zero_chunk_ids:
    if '.' in cid:
        parent_id = cid.rsplit('.', 1)[0]
        if parent_id in zero_chunk_ids:
            parent_is_also_zero += 1

print(f"Number of zero-chunk clauses whose direct parent is ALSO a zero-chunk clause: {parent_is_also_zero}")
