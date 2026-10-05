from langgraph.graph import END, START, StateGraph

from src.generation.nodes import abstain_node, generate_node, make_retrieve_node, verify_claims_node
from src.generation.routers import after_retrieve, should_retry
from src.generation.state import GraphState


def build_graph(retriever=None):
    workflow = StateGraph(GraphState)

    # Add nodes
    workflow.add_node("retrieve", make_retrieve_node(retriever))
    workflow.add_node("generate", generate_node)
    workflow.add_node("verify", verify_claims_node)
    workflow.add_node("abstain", abstain_node)

    # Add edges
    workflow.add_edge(START, "retrieve")

    workflow.add_conditional_edges(
        "retrieve",
        after_retrieve,
        {
            "generate": "generate",
            "abstain": "abstain",
        }
    )

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
