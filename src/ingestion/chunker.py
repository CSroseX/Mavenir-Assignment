import json
import sys
import os

def estimate_tokens(text):
    # Lightweight heuristic: ~4 characters per token
    return len(text) // 4

def chunk_tree(input_json, output_json, spec_id="23.501", version="17.11.0"):
    print(f"Loading tree from {input_json}...")
    with open(input_json, 'r', encoding='utf-8') as f:
        clauses = json.load(f)
        
    chunks = []
    
    # Target and hard cap limits
    SOFT_LIMIT = 400
    HARD_CAP = 800
    
    for clause in clauses:
        clause_id = clause.get('clause_id')
        clause_title = clause.get('clause_title')
        lineage = clause.get('lineage', [])
        
        current_text_buffer = []
        current_token_count = 0
        
        def flush_text_chunk():
            nonlocal current_text_buffer, current_token_count
            if not current_text_buffer:
                return
                
            merged_text = "\n\n".join(current_text_buffer)
            chunks.append({
                "spec_id": spec_id,
                "version": version,
                "clause_id": clause_id,
                "clause_title": clause_title,
                "lineage": lineage,
                "chunk_type": "text",
                "content": merged_text
            })
            current_text_buffer = []
            current_token_count = 0

        for item in clause.get('content', []):
            if item.get('type') == 'table':
                # If we hit a table, flush any accumulated text first to maintain reading order
                flush_text_chunk()
                
                # Tables are NEVER split or flattened. They are saved entirely intact.
                chunks.append({
                    "spec_id": spec_id,
                    "version": version,
                    "clause_id": clause_id,
                    "clause_title": clause_title,
                    "lineage": lineage,
                    "chunk_type": "table",
                    "content": item.get('grid', [])
                })
                
            elif item.get('type') == 'text':
                text = item.get('text', '').strip()
                if not text:
                    continue
                    
                item_tokens = estimate_tokens(text)
                
                # If adding this item exceeds the soft limit (and we already have some text), flush first
                if current_text_buffer and (current_token_count + item_tokens) > SOFT_LIMIT:
                    flush_text_chunk()
                    
                current_text_buffer.append(text)
                current_token_count += item_tokens
                
                # If a single massive paragraph somehow exceeds the hard cap on its own, we flush immediately
                if current_token_count > HARD_CAP:
                    flush_text_chunk()
                    
        # Flush any remaining text at the end of the clause
        flush_text_chunk()
        
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, indent=2)
        
    print(f"Successfully generated {len(chunks)} chunks.")
    print(f"Saved to {output_json}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python chunker.py <input_tree_json> <output_chunks_json> [spec_id] [version]")
        sys.exit(1)
        
    spec = sys.argv[3] if len(sys.argv) > 3 else "23.501"
    ver = sys.argv[4] if len(sys.argv) > 4 else "17.11.0"
    chunk_tree(sys.argv[1], sys.argv[2], spec, ver)
