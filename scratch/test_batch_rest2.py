import os
import requests

# Check both GOOGLE_API_KEY and GEMINI_API_KEY
api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
if not api_key:
    print("API key is not set.")
    exit(1)
    
url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-2:batchEmbedContents"
headers = {
    "x-goog-api-key": api_key,
    "Content-Type": "application/json"
}

payload = {
    "requests": [
        {"model": "models/gemini-embedding-2", "content": {"parts": [{"text": "Hello"}]}},
        {"model": "models/gemini-embedding-2", "content": {"parts": [{"text": "World"}]}}
    ]
}

print("Testing batchEmbedContents with headers...")
res = requests.post(url, headers=headers, json=payload)
print(f"Status: {res.status_code}")
if res.status_code == 200:
    data = res.json()
    embeddings = data.get("embeddings", [])
    print(f"Success! Received {len(embeddings)} embeddings.")
else:
    print(f"Error: {res.text}")
