import sys
import os
import io

if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Ensure the root of the project is in the python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.generation.graph import build_graph
from src.retrieval.retriever import Retriever

def main():
    if len(sys.argv) < 2:
        print("Usage: python verify.py <query>")
        sys.exit(1)
        
    query = sys.argv[1]
    
    print("\n--- RETRIEVING CONTEXT ---")
    retriever = Retriever()
    chunks = retriever.search(query, top_k=3)
    
    if not chunks:
        print("No chunks found for the query.")
        sys.exit(0)
        
    print(f"\nRetrieved {len(chunks)} chunks.")
    
    print("\n--- GENERATING & VERIFYING ANSWER ---")
    app = build_graph()
    
    inputs = {
        "query": query,
        "chunks": chunks,
        "retries": 0
    }
    
    final_state = inputs.copy()
    for output in app.stream(inputs):
        for key, value in output.items():
            print(f"Node Executed: {key}")
            final_state.update(value)
            
    if not final_state:
        print("Error: Graph execution failed.")
        sys.exit(1)
        
    print("\n" + "="*60)
    print("FINAL ANSWER:")
    print("="*60)
    print(final_state["answer"])
    
    print("\n" + "="*60)
    print("SOURCES USED (Retrieved Chunks):")
    print("="*60)
    for i, c in enumerate(chunks):
        print(f"  [{i+1}] Spec: {c['spec_id']} | Clause: {c['clause_id']}")
        
if __name__ == "__main__":
    main()
