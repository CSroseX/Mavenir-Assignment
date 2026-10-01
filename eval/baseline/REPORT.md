# Baseline diagnostic report

## Retrieval quality
- Mean recall@5: 0.583
- Mean recall@20: 0.750
- Mean MRR: 0.468

## End-to-end answer run
- Total questions: 18
- Verified: 1/18
- Retries triggered: 18/18
- Out-of-scope refusals: 0/6

## Stress test
- Total retrieval runs: 15
- p50 latency: 1.7604s
- p95 latency: 2.2760s

## Per-question retrieval highlights
- Q1 (normal): recall@5=0.000, recall@20=1.000, MRR=0.056
- Q2 (normal): recall@5=1.000, recall@20=1.000, MRR=1.000
- Q3 (normal): recall@5=1.000, recall@20=1.000, MRR=0.250
- Q4 (normal): recall@5=1.000, recall@20=1.000, MRR=0.500
- Q5 (normal): recall@5=0.500, recall@20=0.500, MRR=1.000

## Notes
- Baseline evaluation ran under eval/baseline only and did not modify source files outside that directory.
- Qdrant was confirmed live on localhost:6333 before full execution.
