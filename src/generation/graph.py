"""The Phase 4 decision-node graph.

    START -> triage --out_of_domain--> refuse -> END
               | in_domain
               v
            retrieve -> grade_sufficiency --insufficient, rewrites left--> rewrite_query -> retrieve
                              |          --insufficient, exhausted------> abstain -> END
                              | sufficient                (rewrite_query no-op -> abstain)
                              v
                          generate -> verify --passed--> END
                                         | failed, retries left -> generate
                                         | failed, exhausted / verifier down -> abstain -> END

Every conditional edge is a pure function in src/generation/routers.py.
"""

from langgraph.graph import END, START, StateGraph

from src.generation.nodes import (
    abstain_node,
    generate_node,
    make_grade_sufficiency_node,
    make_retrieve_node,
    make_triage_node,
    refuse_node,
    rewrite_query_node,
    verify_claims_node,
)
from src.generation.routers import after_grade, after_rewrite, after_triage, should_retry
from src.generation.state import GraphState
from src.obs.tracing import traced_node


def build_graph(retriever=None, *, embed=None, centroids_loader=None, triage_llm=None, grade_llm=None):
    """Compile the graph.

    retriever: an existing Retriever (app.py / the eval runner cache one);
        constructed lazily on first use otherwise.
    embed: texts -> vectors for triage tier 2. Defaults to the retriever's
        already-loaded BGE model, so triage adds no model load of its own.
    centroids_loader, triage_llm, grade_llm: test seams; default to the
        cached centroids and the shared LLM client.
    """
    retriever_holder = [retriever]

    def get_retriever():
        if retriever_holder[0] is None:
            from src.retrieval.retriever import Retriever
            retriever_holder[0] = Retriever()
        return retriever_holder[0]

    if embed is None:
        def embed(texts):
            from src.generation.centroids import fastembed_embedder
            return fastembed_embedder(get_retriever().model)(texts)

    workflow = StateGraph(GraphState)

    # Add nodes. Each runs inside its own tracing span (a no-op without
    # LANGFUSE_HOST); wrap the stream in trace_query() so they share a trace.
    nodes = {
        "triage": make_triage_node(embed=embed, centroids_loader=centroids_loader, llm=triage_llm),
        "refuse": refuse_node,
        "retrieve": make_retrieve_node(retriever_factory=get_retriever),
        "grade_sufficiency": make_grade_sufficiency_node(grade_llm),
        "rewrite_query": rewrite_query_node,
        "generate": generate_node,
        "verify": verify_claims_node,
        "abstain": abstain_node,
    }
    for name, node in nodes.items():
        workflow.add_node(name, traced_node(name, node))

    # Add edges
    workflow.add_edge(START, "triage")

    workflow.add_conditional_edges(
        "triage",
        after_triage,
        {
            "retrieve": "retrieve",
            "refuse": "refuse",
        }
    )
    workflow.add_edge("refuse", END)

    workflow.add_edge("retrieve", "grade_sufficiency")

    workflow.add_conditional_edges(
        "grade_sufficiency",
        after_grade,
        {
            "generate": "generate",
            "rewrite_query": "rewrite_query",
            "abstain": "abstain",
        }
    )

    workflow.add_conditional_edges(
        "rewrite_query",
        after_rewrite,
        {
            "retrieve": "retrieve",
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
