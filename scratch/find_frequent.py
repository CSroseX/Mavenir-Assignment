import json
from collections import Counter

parsed_path = "data/parsed/ts_123501v171100p.json"
with open(parsed_path, 'r', encoding='utf-8') as f:
    parsed = json.load(f)
    
texts = parsed.get("texts", [])
lines = [t.get("text", "").strip() for t in texts if t.get("text")]
counts = Counter(lines)

print("Top 20 most frequent exact lines:")
for k, v in counts.most_common(20):
    if v > 10:
        print(f"{v} times: {repr(k)}")
