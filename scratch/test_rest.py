import os
import requests
import json

api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    print("GEMINI_API_KEY is not set.")
    exit(1)
    
url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-2:batchEmbedContents?key={api_key}"

texts = ["Hello", "World", "This is a test"]

payload = {
    "requests": [
        {
            "model": "models/gemini-embedding-2",
            "content": {"parts": [{"text": t}]}
        }
        for t in texts
    ]
}

print(f"Sending batchEmbedContents for {len(texts)} texts...")
try:
    response = requests.post(url, json=payload)
    response.raise_for_status()
    data = response.json()
    embeddings = data.get("embeddings", [])
    print(f"Success! Received {len(embeddings)} embeddings.")
    if embeddings:
        print(f"First vector size: {len(embeddings[0]['values'])}")
except Exception as e:
    print(f"Failed: {e}")
    if 'response' in locals():
        print(response.text)
