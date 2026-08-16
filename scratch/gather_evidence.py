import json
import re
import fitz  # PyMuPDF

out = []

def log(msg):
    print(msg)
    out.append(msg)

log("========== PART A ==========")
parsed_path = "data/parsed/ts_123501v171100p.json"
with open(parsed_path, 'r', encoding='utf-8') as f:
    parsed = json.load(f)

texts = parsed.get("texts", [])
reading_order = [t for t in texts] # Docling outputs texts in reading order usually

target_ids = ["5", "5.3", "5.3.4", "4.3.3", "4.4.2", "3"]

for tid in target_ids:
    log(f"\n--- Investigating Clause {tid} in Raw Docling ---")
    # find the heading
    found_idx = -1
    for i, t in enumerate(texts):
        text_val = t.get("text", "").strip()
        if text_val.startswith(tid + " ") or text_val == tid:
            if t.get("label") == "section_header":
                found_idx = i
                break
    
    if found_idx != -1:
        log(f"Found heading at index {found_idx}: {texts[found_idx].get('text')}")
        log("Subsequent 3 items:")
        for j in range(1, 4):
            if found_idx + j < len(texts):
                log(f"  [{found_idx+j}] Label: {texts[found_idx+j].get('label')} | Text: {repr(texts[found_idx+j].get('text'))}")
    else:
        log(f"Heading {tid} not found in raw texts.")

log("\n--- Checking PDF visually (via PyMuPDF) ---")
pdf_path = "data/raw_pdfs/ts_123501v171100p.pdf"
doc = fitz.open(pdf_path)

# Just doing a fast text search in PyMuPDF for these headings
for tid in target_ids:
    log(f"\nSearching PDF for heading {tid}...")
    found = False
    for page_num in range(10, min(100, len(doc))): # Search first 100 pages to save time
        text = doc[page_num].get_text()
        if re.search(r'^' + re.escape(tid) + r'\s+[A-Z]', text, re.MULTILINE):
            log(f"Found on page {page_num + 1}")
            # print surrounding snippet
            match = re.search(r'(^' + re.escape(tid) + r'\s+.*?(?:\n.*){0,3})', text, re.MULTILINE)
            if match:
                log(f"Snippet:\n{match.group(1)}")
            found = True
            break
    if not found:
        log("Not found in first 100 pages.")

log("\n========== PART B ==========")
chunk_files = [
    "data/chunks/ts_123501v171100p_chunks.json",
    "data/chunks/ts_124301v170900p_chunks.json",
    "data/chunks/ts_124501v171200p_chunks.json"
]

patterns = [
    r'ETSI',
    r'3GPP TS \d+\.\d+ version',
    r'Release 17',
    r'^\s*\d+\s*$'
]
combined_pattern = re.compile('|'.join(patterns), re.MULTILINE)

total_affected = {}
for cf in chunk_files:
    with open(cf, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
        
    affected = 0
    for c in chunks:
        content = c.get('content', '')
        if isinstance(content, str) and combined_pattern.search(content):
            affected += 1
    total_affected[cf] = affected

log("Affected Chunks by File:")
for k, v in total_affected.items():
    log(f"  {k}: {v}")

log("\nChecking raw docling for page headers (ETSI, etc):")
# Just look at first 30 items of parsed JSON to see if ETSI is tagged as page_header
for i in range(30):
    t = texts[i]
    if "ETSI" in t.get("text", "") or "3GPP TS" in t.get("text", ""):
        log(f"Docling item: {t.get('text').strip()} => LABEL: {t.get('label')}")

log("\n========== PART C ==========")
# Let's find 5 duplicates in 23.501
with open(chunk_files[0], 'r', encoding='utf-8') as f:
    chunks = json.load(f)

seen = {}
duplicates = []
for c in chunks:
    if c.get("chunk_type") == "text":
        content = c.get("content", "").strip()
        if content in seen:
            duplicates.append((seen[content], c))
        else:
            seen[content] = c

log(f"Total pure text duplicates found in 23.501: {len(duplicates)}")
for i, (c1, c2) in enumerate(duplicates[:5]):
    log(f"\n--- DUPLICATE PAIR {i+1} ---")
    log(f"Chunk 1 ID: {c1.get('clause_id')} | Chunk 2 ID: {c2.get('clause_id')}")
    log(f"TEXT:\n{c1.get('content')}")

with open("scratch/evidence.txt", 'w', encoding='utf-8') as f:
    f.write("\n".join(out))
