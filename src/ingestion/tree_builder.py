import json
import re
import sys
import os

# Regex patterns for 3GPP clause detection
# 1. Numeric clauses (e.g., "5.3.4 UE Mobility", "4.2.5a Radio Cap", "5.30.2.10.2.1")
NUMERIC_PATTERN = re.compile(r'^([1-9][0-9A-Za-z]*(?:\.[0-9A-Za-z]+)*)(?:\s+(.*))?$')
# 2. Annex sub-clauses (e.g., "D.2 Support", "6.2.27a MB-UPF")
ANNEX_SUB_PATTERN = re.compile(r'^([A-Z](?:\.[0-9A-Za-z]+)+)(?:\s+(.*))?$')
# 3. Annex top-level (e.g., "Annex D (informative): 5GS support...")
ANNEX_TOP_PATTERN = re.compile(r'^Annex\s+([A-Z])\s*(?:\([^)]+\))?:\s*(.*)$', re.IGNORECASE)

def parse_ref(ref_str):
    # e.g., "#/texts/45" -> ("texts", 45)
    parts = ref_str.strip('#/').split('/')
    if len(parts) == 2:
        return parts[0], int(parts[1])
    return None, None

def get_depth(clause_id):
    # '5' -> 1
    # '5.3' -> 2
    # 'D' -> 1
    # 'D.2' -> 2
    return clause_id.count('.') + 1

def build_tree(input_json_path, output_json_path):
    print(f"Loading parsed JSON: {input_json_path}")
    with open(input_json_path, 'r', encoding='utf-8') as f:
        doc = json.load(f)
        
    reading_order = doc.get('body', {}).get('children', [])
    
    # We will output a flat list of enriched Clause objects
    clauses = []
    
    # Initialize with a Frontmatter dummy clause for anything before Chapter 1
    current_clause = {
        "clause_id": "0",
        "clause_title": "Frontmatter",
        "depth": 0,
        "lineage": ["Frontmatter"],
        "content": []
    }
    clauses.append(current_clause)
    
    # Stack to maintain hierarchy
    stack = [current_clause]
    
    for ref_obj in reading_order:
        ref_str = ref_obj.get('$ref')
        if not ref_str:
            continue
            
        list_name, idx = parse_ref(ref_str)
        if not list_name or list_name not in doc:
            continue
            
        node = doc[list_name][idx]
        
        # Handle Tables
        if list_name == 'tables':
            grid = node.get('data', {}).get('grid', [])
            
            # Docling grid is a list of rows, where each row is a list of cells
            simple_grid = []
            for row in grid:
                row_texts = []
                for cell in row:
                    clean_text = cell.get('text', '').strip().replace('\n', ' ')
                    row_texts.append(clean_text)
                simple_grid.append(row_texts)
            
            current_clause['content'].append({
                "type": "table",
                "grid": simple_grid
            })
            continue
            
        # Handle Texts
        if list_name == 'texts':
            label = node.get('label')
            text = node.get('text', '').strip()
            
            if not text:
                continue
                
            # --- BOILERPLATE FILTER ---
            exact_junk = ["ETSI", "Contents", "Foreword", "Trademarks", "Copyright Notification", "Intellectual Property Rights", "Essential patents"]
            if text in exact_junk:
                continue
                
            if re.match(r'^3GPP TS \d+\.\d+ version \d+\.\d+\.\d+ Release \d+$', text, re.IGNORECASE):
                continue
            if re.match(r'^ETSI TS \d+\s+\d+\s+V\d+\.\d+\.\d+\s+\(\d{4}-\d{2}\)$', text, re.IGNORECASE):
                continue
            if re.match(r'^\d{1,4}$', text):
                continue
            if re.match(r'^The present document.*ETSI.*$', text, re.IGNORECASE):
                continue
                
            # Strict boilerplate filter for page furniture only.
            # We only drop the block if the entire text string exactly matches a known page furniture format.
            if re.match(r'^3GPP TS \d+\.\d+ V\d+\.\d+\.\d+ \(\d{4}-\d{2}\)$', text, re.IGNORECASE):
                continue
            # --------------------------
                
            is_heading = False
            match = None
            clause_id = None
            title = None
            
            if label == 'section_header':
                # Try to parse it
                match = NUMERIC_PATTERN.match(text)
                if match:
                    clause_id, title = match.groups()
                else:
                    match = ANNEX_SUB_PATTERN.match(text)
                    if match:
                        clause_id, title = match.groups()
                    else:
                        match = ANNEX_TOP_PATTERN.match(text)
                        if match:
                            clause_id, title = match.groups()
                            
                if match:
                    is_heading = True
            
            if is_heading:
                depth = get_depth(clause_id)
                
                # Pop stack until the top is parent of current
                while len(stack) > 0 and stack[-1]['depth'] >= depth:
                    stack.pop()
                    
                # Build lineage
                parent_lineage = stack[-1]['lineage'] if stack else []
                full_lineage = parent_lineage + [f"{clause_id} {title}"]
                
                new_clause = {
                    "clause_id": clause_id,
                    "clause_title": title,
                    "depth": depth,
                    "lineage": full_lineage,
                    "content": []
                }
                
                stack.append(new_clause)
                clauses.append(new_clause)
                current_clause = new_clause
                
            else:
                # Normal text (or a rogue section header that failed RegEx)
                current_clause['content'].append({
                    "type": "text",
                    "text": text
                })
                
    # Save output
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(clauses, f, indent=2)
        
    print(f"Successfully built tree with {len(clauses)} logical clauses.")
    print(f"Saved to: {output_json_path}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python tree_builder.py <input_docling_json> <output_tree_json>")
        sys.exit(1)
        
    build_tree(sys.argv[1], sys.argv[2])
