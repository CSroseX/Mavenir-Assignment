import json
import sys

def inspect(json_path):
    print(f"Loading {json_path}...\n")
    with open(json_path, 'r', encoding='utf-8') as f:
        d = json.load(f)
        
    texts = d.get('texts', [])
    tables = d.get('tables', [])
    
    # 1. Heading Tree
    print("=== 1. Heading Tree (First 40) ===")
    heading_count = 0
    total_headings = sum(1 for t in texts if t.get('label') == 'section_header')
    
    for t in texts:
        if t.get('label') == 'section_header':
            level = t.get('level', 1)
            indent = "  " * (level - 1)
            print(f"{indent}{t.get('text', '')}")
            heading_count += 1
            if heading_count >= 40:
                print(f"  ... (truncating, {total_headings} total headings)")
                break
                
    # 2. Table Count & Structure
    print(f"\n=== 2. Tables ===")
    print(f"Total tables detected: {len(tables)}")
    if len(tables) > 10:
        print("\nStructure of table #11 (skipping first few which might be TOC/Revision history):")
        table = tables[10]
        grid = table.get('data', {}).get('grid', [])
        for row in grid[:8]: # print first 8 rows
            row_texts = [cell.get('text', '').strip().replace('\n', ' ') for cell in row]
            print(" | ".join(row_texts))
        if len(grid) > 8:
            print("... (more rows)")
            
    # 3. Stats per Top-Level Clause
    print(f"\n=== 3. Character Count per Top-Level Clause ===")
    current_clause = "Frontmatter"
    clause_stats = {current_clause: 0}
    
    for t in texts:
        # A top-level clause in 3GPP usually starts with a single digit, e.g. "4 Architecture model"
        # But we'll rely on Docling's level==1 for now to see if it worked.
        if t.get('label') == 'section_header' and t.get('level', 1) == 1:
            current_clause = t.get('text', 'Unknown').strip()
            clause_stats[current_clause] = 0
            
        text_content = t.get('text', '')
        clause_stats[current_clause] += len(text_content)
        
    for clause, char_count in clause_stats.items():
        if char_count > 0:
            # truncate clause name for clean alignment
            short_clause = (clause[:50] + '..') if len(clause) > 50 else clause
            print(f"{short_clause:<52} : {char_count:>8} chars")

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else 'data/parsed/ts_123501v171100p.json'
    inspect(path)
