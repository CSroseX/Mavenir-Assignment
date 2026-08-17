import os
import json
import sys
import time

# Ensure the root of the project is in the python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.generation.graph import build_graph
from src.retrieval.retriever import Retriever

def is_refusal(answer: str) -> bool:
    """Basic heuristic to check if the model refused to answer."""
    lower_ans = answer.lower()
    refusal_keywords = [
        "cannot answer", "out of scope", "only answer 3gpp", 
        "sorry", "cannot provide", "not provided in the",
        "do not have information", "not supported by"
    ]
    return any(kw in lower_ans for kw in refusal_keywords)

def main():
    eval_set_path = os.path.join(os.path.dirname(__file__), "eval_set.json")
    results_path = os.path.join(os.path.dirname(__file__), "results.json")
    
    if not os.path.exists(eval_set_path):
        print(f"Eval set not found at {eval_set_path}")
        sys.exit(1)
        
    with open(eval_set_path, "r", encoding="utf-8") as f:
        eval_set = json.load(f)
        
    print(f"Loaded {len(eval_set)} questions from eval set.")
    
    app = build_graph()
    retriever = Retriever()
    
    results = []
    if os.path.exists(results_path):
        with open(results_path, "r", encoding="utf-8") as f:
            try:
                results = json.load(f)
                print(f"Resuming from {len(results)} existing results...")
            except:
                pass
                
    start_index = len(results)
    
    for i in range(start_index, len(eval_set)):
        item = eval_set[i]
        question = item["question"]
        q_type = item["type"]
        
        print(f"\n[{i+1}/{len(eval_set)}] Evaluating ({q_type}): {question}")
        
        # Retrieve chunks
        chunks = retriever.search(question, top_k=5)
        
        inputs = {
            "query": question,
            "chunks": chunks,
            "retries": 0
        }
        
        final_state = inputs.copy()
        
        # Run graph
        for output in app.stream(inputs):
            for key, value in output.items():
                final_state.update(value)
                
        # Determine status
        if final_state.get("verification_passed"):
            if final_state.get("retries", 0) > 0:
                status = "verified_after_retry"
            else:
                status = "verified"
        else:
            status = "flagged_unverified"
            
        result = {
            "question": question,
            "type": q_type,
            "expected_behavior": item.get("expected_behavior", ""),
            "generated_answer": final_state.get("answer", ""),
            "sources": [{"spec_id": c.get("spec_id"), "clause_id": c.get("clause_id")} for c in chunks],
            "verification_passed": final_state.get("verification_passed", False),
            "retries": final_state.get("retries", 0),
            "status": status,
            "feedback": final_state.get("feedback", "")
        }
        
        results.append(result)
        
        # Save intermediate results in case of crash
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
            
        # Rate limit protection: wait 5 seconds between questions
        if i < len(eval_set) - 1:
            print("Waiting 5 seconds to avoid rate limits...")
            time.sleep(5)
            
    # Compute aggregate metrics
    total = len(results)
    if total == 0:
        return
        
    passed = sum(1 for r in results if r["verification_passed"])
    retried = sum(1 for r in results if r["retries"] > 0)
    flagged = sum(1 for r in results if r["status"] == "flagged_unverified")
    
    out_of_scope = [r for r in results if r["type"] == "out_of_scope"]
    oos_refusals = sum(1 for r in out_of_scope if is_refusal(r["generated_answer"]) or len(r["sources"]) == 0 or "flagged" in r["status"])
    
    adversarial = [r for r in results if r["type"] == "adversarial"]
    adv_caught = sum(1 for r in adversarial if r["retries"] > 0) # Triggered verification failure at least once
    
    print("\n" + "="*60)
    print("EVALUATION AGGREGATE METRICS")
    print("="*60)
    print(f"Total Questions: {total}")
    print(f"Verification Pass Rate: {passed / total * 100:.1f}% ({passed}/{total})")
    print(f"Retry Rate: {retried / total * 100:.1f}% ({retried}/{total})")
    print(f"Flagged Rate: {flagged / total * 100:.1f}% ({flagged}/{total})")
    
    if out_of_scope:
        print(f"Refusal Accuracy (Out-of-Scope): {oos_refusals / len(out_of_scope) * 100:.1f}% ({oos_refusals}/{len(out_of_scope)})")
    
    if adversarial:
        print(f"Conflation Catch Rate (Adversarial): {adv_caught / len(adversarial) * 100:.1f}% ({adv_caught}/{len(adversarial)})")
        
    print("\nResults saved to eval/results.json")

if __name__ == "__main__":
    main()
