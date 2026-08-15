import json
import re

NUMERIC_PATTERN = re.compile(r'^([1-9][0-9A-Za-z]*(?:\.[0-9A-Za-z]+)*)(?:\s+(.*))?$')
ANNEX_SUB_PATTERN = re.compile(r'^([A-Z](?:\.[0-9A-Za-z]+)+)(?:\s+(.*))?$')
ANNEX_TOP_PATTERN = re.compile(r'^Annex\s+([A-Z])\s*(?:\([^)]+\))?:\s*(.*)$', re.IGNORECASE)

def run():
    with open('data/parsed/ts_123501v171100p.json', 'r', encoding='utf-8') as f:
        doc = json.load(f)
        
    texts = doc.get('texts', [])
    
    rogues = []
    rescued = []
    
    for t in texts:
        if t.get('label') == 'section_header':
            text = t.get('text', '').strip()
            
            # Old strict pattern for comparison
            old_numeric = re.compile(r'^([1-9]\d*(?:\.\d+)*)\s+(.*)$')
            old_annex_sub = re.compile(r'^([A-Z](?:\.\d+)+)\s+(.*)$')
            
            is_valid_new = (NUMERIC_PATTERN.match(text) or ANNEX_SUB_PATTERN.match(text) or ANNEX_TOP_PATTERN.match(text))
            is_valid_old = (old_numeric.match(text) or old_annex_sub.match(text) or ANNEX_TOP_PATTERN.match(text))
            
            if is_valid_new and not is_valid_old:
                rescued.append(text)
                
            if not is_valid_new:
                rogues.append(text)
                
    print(f"Total downgraded headers (rogues): {len(rogues)}")
    print("\nExamples of remaining rogues:")
    for r in sorted(set(rogues))[:10]:
        print(f"- {r}")
        
    print(f"\nTotal newly rescued headers: {len(rescued)}")
    print("\nExamples of rescued headers:")
    for r in sorted(set(rescued))[:5]:
        print(f"- {r}")

if __name__ == '__main__':
    run()
