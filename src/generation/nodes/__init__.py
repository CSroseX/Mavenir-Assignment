"""Re-exports so `from src.generation.nodes import generate_node, ...` and
`from src.generation import nodes as gen_nodes; gen_nodes.client` keep
working unchanged after nodes.py became the nodes/ package (to host
abstain.py alongside it, per issue #16)."""

from src.generation.nodes.abstain import abstain_node  # noqa: F401
from src.generation.nodes.core import (  # noqa: F401
    BANNED_PHRASE_PATTERNS,
    MODEL_NAME,
    _detect_banned_phrase,
    client,
    generate_node,
    verify_claims_node,
)
