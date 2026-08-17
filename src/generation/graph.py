from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, START, END
from src.generation.nodes import generate_node, verify_claims_node

class GraphState(TypedDict):
    query: str
    chunks: List[Dict[str, Any]]
    answer: str
    verification_passed: bool
    feedback: str
    retries: int

def should_retry(state: GraphState):
    """
    Conditional edge router.
    """
    if state["verification_passed"]:
        return END
        
    if state["retries"] >= 3:
        # Exhausted retries. We will flag the unverified parts in the final output.
        return "flag_unverified"
        
    return "generate"

def flag_unverified_node(state: GraphState) -> Dict[str, Any]:
    """
    Appends a warning to the answer if verification failed after max retries.
    """
    answer = state["answer"]
    feedback = state.get("feedback", "")
    
    warning = f"\n\n[WARNING: The following claims could not be verified against the source text:\n{feedback.replace('The following claims were NOT supported by the source text and must be removed or corrected:', '').strip()}]"
    
    return {"answer": answer + warning}

def build_graph():
    workflow = StateGraph(GraphState)
    
    # Add nodes
    workflow.add_node("generate", generate_node)
    workflow.add_node("verify", verify_claims_node)
    workflow.add_node("flag_unverified", flag_unverified_node)
    
    # Add edges
    workflow.add_edge(START, "generate")
    workflow.add_edge("generate", "verify")
    
    workflow.add_conditional_edges(
        "verify",
        should_retry,
        {
            END: END,
            "generate": "generate",
            "flag_unverified": "flag_unverified"
        }
    )
    
    workflow.add_edge("flag_unverified", END)
    
    return workflow.compile()
