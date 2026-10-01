import os
import sys
import time
import json
import glob
import io
from dotenv import load_dotenv

# Ensure stdout handles UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Ensure project root is in sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

load_dotenv()

print("=== STEP 1: PREFLIGHT ===\n")

# 1. Check Qdrant
print("--- 1. Qdrant Reachability & Point Counts ---")
try:
    from qdrant_client import QdrantClient
    q_client = QdrantClient(url="http://localhost:6333", check_compatibility=False)
    collections = [c.name for c in q_client.get_collections().collections]
    print(f"Collections found: {collections}")
    if "3gpp_specs" in collections:
        point_count = q_client.count(collection_name="3gpp_specs").count
        print(f"Collection '3gpp_specs' point count: {point_count}")
    else:
        print("Collection '3gpp_specs' does NOT exist!")
        point_count = None
except Exception as e:
    print(f"Qdrant error: {e}")
    point_count = None

# Sum chunk files
chunk_files = sorted(glob.glob("data/chunks/*_chunks.json"))
total_file_chunks = 0
for cf in chunk_files:
    with open(cf, "r", encoding="utf-8") as f:
        data = json.load(f)
        count = len(data)
        total_file_chunks += count
        print(f"  {cf}: {count} chunks")
print(f"Total chunks across data/chunks/*_chunks.json: {total_file_chunks}")
print(f"Point count matches total file chunks: {point_count == total_file_chunks} (Qdrant: {point_count} vs Files: {total_file_chunks})\n")

# 2. Check Env Vars & 1-token LLM ping
print("--- 2. Env Vars & 1-Token LLM Ping ---")
use_remote = os.environ.get("USE_REMOTE_LLM", "")
openai_key = os.environ.get("OPENAI_API_KEY", "")
openai_url = os.environ.get("OPENAI_BASE_URL", "")
model_name = os.environ.get("MODEL_NAME", "")

key_status = f"SET (length {len(openai_key)})" if openai_key else "NOT SET"
print(f"USE_REMOTE_LLM: {use_remote}")
print(f"OPENAI_API_KEY: {key_status}")
print(f"OPENAI_BASE_URL: {openai_url}")
print(f"MODEL_NAME: {model_name}")

try:
    from openai import OpenAI
    if use_remote and use_remote.lower() == "true":
        client = OpenAI(api_key=openai_key, base_url=openai_url)
    else:
        client = OpenAI(api_key="ollama", base_url="http://localhost:11434/v1")
        
    t0 = time.time()
    response = client.chat.completions.create(
        model=model_name if model_name else "gpt-3.5-turbo",
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=1
    )
    t1 = time.time()
    print(f"LLM Ping: SUCCESS (Latency: {(t1 - t0)*1000:.2f} ms)")
    print(f"Response text: {repr(response.choices[0].message.content)}")
    if hasattr(response, "usage") and response.usage:
        print(f"Token Usage: prompt_tokens={response.usage.prompt_tokens}, completion_tokens={response.usage.completion_tokens}, total_tokens={response.usage.total_tokens}")
except Exception as e:
    print(f"LLM Ping FAILED: {e}")

# 3. Retriever Loading Time
print("\n--- 3. Retriever Loading Time (Embedder + Cross-Encoder) ---")
try:
    t0 = time.time()
    from src.retrieval.retriever import Retriever
    retriever = Retriever()
    t1 = time.time()
    load_time = t1 - t0
    print(f"Retriever loaded successfully in {load_time:.3f} s")
except Exception as e:
    print(f"Retriever load FAILED: {e}")
    retriever = None

# 4. Single Retrieval Call for Each Eval Question
print("\n--- 4. Single Retrieval Call per Eval Question ---")
if os.path.exists("eval/eval_set.json") and retriever:
    with open("eval/eval_set.json", "r", encoding="utf-8") as f:
        eval_set = json.load(f)
    print(f"Loaded {len(eval_set)} questions from eval/eval_set.json:\n")
    for i, item in enumerate(eval_set, 1):
        q = item["question"]
        q_type = item["type"]
        t0 = time.time()
        results = retriever.search(q, top_k=5)
        dt = (time.time() - t0) * 1000
        print(f"Q{i:02d} [{q_type}]: \"{q}\"")
        print(f"  -> Returned {len(results)} chunks in {dt:.1f} ms")
        if results:
            top = results[0]
            print(f"  -> Top-1: Spec {top['spec_id']} Clause {top['clause_id']} | Score: {top.get('score')} | Preview: {top['content_preview'][:80]}...")
        print()
else:
    print("eval/eval_set.json missing or retriever failed")
