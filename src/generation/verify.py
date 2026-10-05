import io
import os
import sys

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

    print("\n--- BUILDING GRAPH WITH RETRIEVAL ---")
    retriever = Retriever()
    app = build_graph(retriever=retriever)

    inputs = {"query": query}

    print("\n--- RETRIEVING, GENERATING & VERIFYING ANSWER ---")
    final_state = inputs.copy()
    for output in app.stream(inputs):
        for key, value in output.items():
            print(f"Node Executed: {key}")
            final_state.update(value)

    if not final_state:
        print("Error: Graph execution failed.")
        sys.exit(1)

    chunks = final_state.get("chunks", [])
    print(f"\nRetrieved {len(chunks)} chunks.")

    print("\n" + "="*60)
    print("FINAL ANSWER:")
    print("="*60)
    print(final_state["answer"])

    print("\n" + "="*60)
    print("VERIFICATION RESULT:")
    print("="*60)
    print(f"Pass/Fail Verdict: {'PASS' if final_state.get('verification_passed') else 'FAIL'}")
    print(f"Retry Count: {final_state.get('retries', 0)}")
    print(f"Reasoning/Feedback:\n{final_state.get('feedback', 'None')}")

    print("\n" + "="*60)
    print("SOURCES USED (Retrieved Chunks):")
    print("="*60)
    for i, c in enumerate(chunks):
        print(f"  [{i+1}] Spec: {c['spec_id']} | Clause: {c['clause_id']}")

if __name__ == "__main__":
    main()
