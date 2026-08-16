import json
import glob
import random
import statistics
import re
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

# Files
chunk_files = {
    "23.501": "data/chunks/ts_123501v171100p_chunks.json",
    "24.301": "data/chunks/ts_124301v170900p_chunks.json",
    "24.501": "data/chunks/ts_124501v171200p_chunks.json"
}

tree_files = {
    "23.501": "data/chunks/ts_123501v171100p_tree.json",
    "24.301": "data/chunks/ts_124301v170900p_tree.json",
    "24.501": "data/chunks/ts_124501v171200p_tree.json"
}

out_md = "C:/Users/chitr/.gemini/antigravity-ide/brain/7ef86a7e-8e56-433a-8939-411b5a0d0579/chunk_audit.md"
lines = ["# Chunk Quality Audit Report\n"]

total_corpus_chunks = 0
spec_chunk_counts = {}

def get_word_count(text):
    return len(text.split())

def is_cut_off(text):
    if not text: return False
    text = text.strip()
    return not text[-1] in ['.', ';', ':', '?', '!']

for spec_id, c_file in chunk_files.items():
    t_file = tree_files[spec_id]
    
    with open(c_file, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
        
    with open(t_file, 'r', encoding='utf-8') as f:
        tree_clauses = json.load(f)
        
    lines.append(f"## SPEC {spec_id}")
    
    # 1. Total chunk count
    text_chunks = [c for c in chunks if c.get('chunk_type') == 'text']
    table_chunks = [c for c in chunks if c.get('chunk_type') == 'table']
    total_chunks = len(chunks)
    total_corpus_chunks += total_chunks
    spec_chunk_counts[spec_id] = total_chunks
    
    lines.append("### 1. Total Chunk Count")
    lines.append(f"- **Total:** {total_chunks}")
    lines.append(f"- **Text:** {len(text_chunks)}")
    lines.append(f"- **Table:** {len(table_chunks)}\n")
    
    # 2. Word count distribution (text chunks)
    word_counts = [get_word_count(c.get('content', '')) for c in text_chunks]
    if word_counts:
        min_words = min(word_counts)
        max_words = max(word_counts)
        avg_words = sum(word_counts) / len(word_counts)
        med_words = statistics.median(word_counts)
        below_300 = sum(1 for w in word_counts if w < 300)
        above_500 = sum(1 for w in word_counts if w > 500)
    else:
        min_words = max_words = avg_words = med_words = below_300 = above_500 = 0
        
    lines.append("### 2. Token/Word Count Distribution (Text)")
    lines.append(f"- **Min:** {min_words}")
    lines.append(f"- **Max:** {max_words}")
    lines.append(f"- **Average:** {avg_words:.2f}")
    lines.append(f"- **Median:** {med_words}")
    lines.append(f"- **< 300 words:** {below_300}")
    lines.append(f"- **> 500 words:** {above_500}\n")
    
    # 3. Print 5 random text chunks
    lines.append("### 3. Random Text Chunks (5)")
    random_text = random.sample(text_chunks, min(5, len(text_chunks)))
    for c in random_text:
        lines.append(f"**ID:** {c.get('clause_id')} | **Lineage:** {c.get('lineage')}")
        lines.append(f"```text\n{c.get('content', '')[:1000]}...\n```\n")
        
    # 4. Print 3 random table chunks
    lines.append("### 4. Random Table Chunks (3)")
    random_table = random.sample(table_chunks, min(3, len(table_chunks)))
    for c in random_table:
        lines.append(f"**ID:** {c.get('clause_id')} | **Lineage:** {c.get('lineage')}")
        lines.append(f"```json\n{json.dumps(c.get('content', [])[:3], indent=2)}\n...\n```\n")
        
    # 5. Empty or near-empty (<20 char)
    empty_chunks = sum(1 for c in text_chunks if len(c.get('content', '').strip()) < 20)
    lines.append(f"### 5. Empty / Near-Empty Chunks (< 20 chars)")
    lines.append(f"- **Count:** {empty_chunks}\n")
    
    # 6. Missing lineage or length == 1
    bad_lineage = sum(1 for c in chunks if not c.get('lineage') or len(c.get('lineage')) <= 1)
    lines.append(f"### 6. Missing / Shallow Lineage (length <= 1)")
    lines.append(f"- **Count:** {bad_lineage}\n")
    
    # 7. Cut off mid-sentence
    cutoff_chunks = [c for c in text_chunks if is_cut_off(c.get('content'))]
    lines.append(f"### 7. Cut-off Mid-Sentence")
    lines.append(f"- **Count:** {len(cutoff_chunks)}")
    for c in cutoff_chunks[:5]:
        content = c.get('content', '')
        snippet = content[-100:] if len(content) > 100 else content
        lines.append(f"- ...`{snippet}`")
    lines.append("\n")
    
    # 8. Merged unrelated clauses (15 random chunks for AI review)
    lines.append("### 8. Merged Unrelated Clauses Evaluation (15 Random)")
    random_15 = random.sample(text_chunks, min(15, len(text_chunks)))
    merged_count = 0
    for i, c in enumerate(random_15):
        # AI Review logic placeholder - I will output the raw text for review.
        content = c.get('content', '')
        clause_id = str(c.get('clause_id', ''))
        # Simple heuristic check for rogue headers inside text
        rogue_matches = re.findall(r'(?:\n|^)([1-9][0-9A-Za-z]*(?:\.[0-9A-Za-z]+)*\s+[A-Z][a-zA-Z\s]+)', content)
        # Filter out self
        rogue_headers = [m for m in rogue_matches if not m.startswith(clause_id)]
        if rogue_headers:
            merged_count += 1
            lines.append(f"**Found potential merge:** Chunk {c.get('clause_id')} contains possible rogue headers: {rogue_headers}")
    lines.append(f"- **Potential Merges found in sample of 15:** {merged_count}\n")
            
    # 9. Duplicates
    distinct_content = set([c.get('content') if isinstance(c.get('content'), str) else json.dumps(c.get('content')) for c in chunks])
    duplicate_count = total_chunks - len(distinct_content)
    lines.append("### 9. Duplicates")
    lines.append(f"- **Total Chunks:** {total_chunks}")
    lines.append(f"- **Distinct Content:** {len(distinct_content)}")
    lines.append(f"- **Duplicates:** {duplicate_count}\n")
    
    # 10. Per-clause coverage
    tree_clause_ids = [c.get('clause_id') for c in tree_clauses]
    chunk_clause_ids = set([c.get('clause_id') for c in chunks])
    zero_chunks = [cid for cid in tree_clause_ids if cid not in chunk_clause_ids]
    lines.append("### 10. Per-Clause Coverage (Tree vs Chunks)")
    lines.append(f"- **Tree Clauses with 0 chunks:** {len(zero_chunks)}")
    if zero_chunks:
        lines.append(f"- **IDs:** {', '.join(zero_chunks[:50])} ...\n")
    else:
        lines.append("\n")

lines.append("## ACROSS THE WHOLE CORPUS (Qdrant)\n")

# Connect to Qdrant
q_client = QdrantClient(url="http://localhost:6333", check_compatibility=False)
COLLECTION_NAME = "3gpp_specs"

# 11. points_count
try:
    collection_info = q_client.get_collection(COLLECTION_NAME)
    points_count = collection_info.points_count
    lines.append("### 11. Qdrant Points Count")
    lines.append(f"- **Total Points:** {points_count}\n")
    
    # 12. Confirm sums
    lines.append("### 12. Points Count Check")
    lines.append(f"- **Sum of local JSON chunks:** {total_corpus_chunks}")
    if points_count == total_corpus_chunks:
        lines.append("- **Match:** YES. Collection count exactly matches chunk sum.\n")
    else:
        lines.append(f"- **Match:** NO. Expected {total_corpus_chunks} but Qdrant has {points_count}.\n")
        
    # 13. Break down by spec_id
    lines.append("### 13. Qdrant Break Down by Spec_ID")
    qdrant_spec_counts = {}
    for spec_id in chunk_files.keys():
        count_res = q_client.count(
            collection_name=COLLECTION_NAME,
            count_filter=Filter(
                must=[FieldCondition(key="spec_id", match=MatchValue(value=spec_id))]
            )
        )
        qdrant_spec_counts[spec_id] = count_res.count
        lines.append(f"- **{spec_id}:** {count_res.count} (Expected: {spec_chunk_counts[spec_id]})")
        
except Exception as e:
    lines.append(f"### QDRANT ERROR\nFailed to query Qdrant: {e}\n")

with open(out_md, 'w', encoding='utf-8') as f:
    f.write("\n".join(lines))
    
print(f"Audit report generated at: {out_md}")
