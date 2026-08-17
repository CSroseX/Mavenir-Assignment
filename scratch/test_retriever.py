import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.retrieval.retriever import Retriever

def test_queries():
    retriever = Retriever()
    
    queries = [
        "What timer does the AMF use for a Tracking Area Update in EPS?",
        "What is the role of the AMF in the 5G system architecture?"
    ]
    
    for q in queries:
        print(f"\nQUERY: {q}")
        results = retriever.search(q, top_k=5)
        for i, r in enumerate(results):
            print(f"[{i+1}] Spec {r['spec_id']} Clause {r['clause_id']} | Score: {r['score']}")
            
if __name__ == "__main__":
    test_queries()
