from typing import Dict, Any
from langgraph.graph import StateGraph, START, END
from src.generation.nodes import generate_node, verify_claims_node
from src.generation.routers import should_retry
from src.generation.state import GraphState

def flag_unverified_node(state: GraphState) -> Dict[str, Any]:
    """
    Passes through the answer if verification failed after max retries, without appending a visible warning.
    Sets termination_reason unless the verifier itself was the reason we got here.
    """
    result = {"answer": state["answer"]}
    if state.get("termination_reason") != "verifier_unavailable":
        result["termination_reason"] = "verification_failed"
    return result

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
