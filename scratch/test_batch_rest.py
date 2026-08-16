import os
import requests

api_key = os.environ.get("GEMINI_API_KEY")
url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-2:batchEmbedContents?key={api_key}"

payload = {
    "requests": [
        {"model": "models/gemini-embedding-2", "content": {"parts": [{"text": "test"}]}}
    ]
}

print("Testing batchEmbedContents REST API...")
res = requests.post(url, json=payload)
print(f"Status Code: {res.status_code}")
print(f"Response: {res.text}")
