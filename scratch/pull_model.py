import urllib.request
import json
import sys

print("Pulling qwen3.5:4b from Ollama...")
req = urllib.request.Request(
    'http://localhost:11434/api/pull',
    data=json.dumps({"name": "qwen3.5:4b"}).encode('utf-8'),
    headers={'Content-Type': 'application/json'}
)

try:
    with urllib.request.urlopen(req) as response:
        for line in response:
            if line:
                data = json.loads(line.decode('utf-8'))
                status = data.get("status", "")
                if "total" in data and "completed" in data:
                    percent = (data["completed"] / data["total"]) * 100
                    print(f"\r{status}: {percent:.1f}%", end="")
                else:
                    print(f"\r{status}", end="")
                    
    print("\nModel pulled successfully!")
except Exception as e:
    print(f"\nError pulling model: {e}")
    sys.exit(1)
