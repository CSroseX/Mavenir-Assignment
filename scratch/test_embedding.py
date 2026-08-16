import os
import sys
from google import genai

print("Initializing standard client...")
client = genai.Client()

print("\nInitializing v1alpha client...")
client_alpha = genai.Client(http_options={'api_version': 'v1alpha'})

models = [
    "text-embedding-004",
    "models/text-embedding-004",
    "embedding-001",
    "text-embedding-004-latest"
]

for c_name, c in [("v1beta (default)", client), ("v1alpha", client_alpha)]:
    print(f"\n=== Testing with {c_name} ===")
    for m in models:
        try:
            res = c.models.embed_content(
                model=m,
                contents="hello world"
            )
            print(f"[SUCCESS] {m} worked! Vector size: {len(res.embeddings[0].values)}")
        except Exception as e:
            # truncate error for readability
            err = str(e).split('Call ModelService.ListModels')[0].strip()
            print(f"[FAILED] {m}: {err}")
