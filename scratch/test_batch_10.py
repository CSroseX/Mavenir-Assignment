import os
import json
import glob
import requests

api_key = os.environ.get("GEMINI_API_KEY").strip()

# Load chunks
chunk_files = glob.glob("data/chunks/*_chunks.json")
all_chunks = []
for f in chunk_files:
    with open(f, 'r', encoding='utf-8') as file:
        all_chunks.extend(json.load(file))

batch = all_chunks[900:1000]

def format_table_for_embedding(grid):
    if not grid: return ""
    lines = []
    for i, row in enumerate(grid):
        lines.append(" | ".join([str(cell) for cell in row]))
        if i == 0:
            lines.append(" | ".join(["---"] * len(row)))
    return "\n".join(lines)

texts_to_embed = []
for chunk in batch:
    if chunk["chunk_type"] == "table":
        embed_text = format_table_for_embedding(chunk["content"])
    else:
        embed_text = chunk["content"]
    context = f"Spec {chunk['spec_id']} {chunk['version']} Clause {chunk['clause_id']} {chunk['clause_title']}\n"
    texts_to_embed.append(context + embed_text)

url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-2:batchEmbedContents?key={api_key}"
payload = {
    "requests": [
        {"model": "models/gemini-embedding-2", "content": {"parts": [{"text": t}]}}
        for t in texts_to_embed
    ]
}

print(f"Sending batch 10 (chunks 900-1000)...")
res = requests.post(url, json=payload)
print(f"Status: {res.status_code}")
if res.status_code != 200:
    print(f"Error payload:\n{res.text}")
else:
    print("Success on batch 10!")
