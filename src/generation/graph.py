from langgraph.graph import StateGraph, START, END
from src.generation.nodes import generate_node, verify_claims_node, abstain_node
from src.generation.routers import should_retry
from src.generation.state import GraphState

def build_graph():
    workflow = StateGraph(GraphState)

    # Add nodes
    workflow.add_node("generate", generate_node)
    workflow.add_node("verify", verify_claims_node)
    workflow.add_node("abstain", abstain_node)

    # Add edges
    workflow.add_edge(START, "generate")
    workflow.add_edge("generate", "verify")

    workflow.add_conditional_edges(
        "verify",
        should_retry,
        {
            END: END,
            "generate": "generate",
            "flag_unverified": "abstain"
        }
    )

    workflow.add_edge("abstain", END)

    return workflow.compile()
