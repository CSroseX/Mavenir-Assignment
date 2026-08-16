import random
import os
import sys

log_file = r"c:\Users\chitr\Desktop\coding\Mavenir Assignment\scratch\filtered_ambiguous.log"

with open(log_file, "r", encoding="utf-8") as f:
    lines = [line.strip() for line in f if line.strip()]

print(f"Total lines in log: {len(lines)}")

random.seed(42) # For reproducibility
sample_15 = random.sample(lines, 15)

print("\n--- 15 RANDOM LINES ---")
for i, line in enumerate(sample_15):
    print(f"[{i+1}] {line}")

# Also output a 100-line sample to a file so we can classify it
sample_100 = random.sample(lines, min(100, len(lines)))
with open("scratch/sample_100.txt", "w", encoding="utf-8") as f:
    for line in sample_100:
        f.write(line + "\n")
