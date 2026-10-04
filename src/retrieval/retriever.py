import glob
import json
import re
import sys

try:
    from fastembed import TextEmbedding
    from qdrant_client import QdrantClient
    from qdrant_client.models import FieldCondition, Filter, MatchAny, MatchValue
    from sentence_transformers import CrossEncoder
except ImportError:
    print("Missing dependencies. Please run: pip install qdrant-client fastembed sentence-transformers")
    sys.exit(1)

from src.config import get_settings
from src.obs.logging import get_logger

log = get_logger(__name__)

class Retriever:
    def __init__(self, collection_name=None, qdrant_url=None):
        cfg = get_settings()
        self.collection_name = collection_name or cfg.qdrant_collection
        self.q_client = QdrantClient(url=qdrant_url or cfg.qdrant_url, check_compatibility=False)
        self.model = TextEmbedding(model_name=cfg.embedding_model)
        log.info("retriever.loading_cross_encoder")
        self.cross_encoder = CrossEncoder(cfg.reranker_model)
        log.info("retriever.cross_encoder_loaded")

        # Regex patterns for query routing
        self.spec_pattern = re.compile(r"(?i)(?:TS\s*)?\b(\d{2}\.\d{3})\b")
        self.clause_pattern = re.compile(r"(?i)(?:clause|sec|section|subclause)\s+([1-9A-Z][\d\.a-zA-Z]*)\b")

        # Load all valid clause IDs for prefix matching
        self.valid_clause_ids = set()
        chunk_dir = cfg.project_root / "data" / "chunks"
        chunk_files = glob.glob(str(chunk_dir / "*_chunks.json"))
        if not chunk_files:
            raise FileNotFoundError(
                f"No chunk files found in {chunk_dir}. "
                "Run the ingestion pipeline first (see CLAUDE.md)."
            )
        for f in chunk_files:
            with open(f, "r", encoding="utf-8") as file:
                chunks = json.load(file)
                for c in chunks:
                    self.valid_clause_ids.add(c["clause_id"])
        if not self.valid_clause_ids:
            raise ValueError(
                f"Chunk files in {chunk_dir} contained no clause IDs. "
                "The files may be empty or malformed."
            )

    def _parse_query(self, query: str):
        """
        Parses the query to extract explicit spec_id or clause_id targeting.
        Returns a tuple of (spec_id, clause_id).
        """
        spec_match = self.spec_pattern.search(query)
        clause_match = self.clause_pattern.search(query)

        spec_id = spec_match.group(1) if spec_match else None
        clause_id = clause_match.group(1).rstrip('.') if clause_match else None

        return spec_id, clause_id

    def search(self, query: str, top_k: int = None):
        """
        Executes a dual-path retrieval based on query parsing.
        """
        cfg = get_settings()
        if top_k is None:
            top_k = cfg.default_top_k

        spec_id, clause_id = self._parse_query(query)

        # Determine Path A (Default) vs Path B (Targeted)
        is_targeted = bool(spec_id or clause_id)
        path_name = "TARGETED PATH" if is_targeted else "DEFAULT PATH"

        log.debug("retriever.search.path", path=path_name, query=query)

        # Build Filter if Targeted
        query_filter = None
        if is_targeted:
            conditions = []
            if spec_id:
                log.debug("retriever.search.spec_targeting", spec_id=spec_id)
                conditions.append(
                    FieldCondition(key="spec_id", match=MatchValue(value=spec_id))
                )
            if clause_id:
                # Find all clauses that are this clause or children of this clause
                prefix = clause_id + "."
                matched_clauses = [c for c in self.valid_clause_ids if c == clause_id or c.startswith(prefix)]

                if not matched_clauses:
                    # Fallback to just the raw string if not found
                    matched_clauses = [clause_id]

                log.debug(
                    "retriever.search.clause_targeting",
                    clause_id=clause_id,
                    expanded_count=len(matched_clauses),
                )
                conditions.append(
                    FieldCondition(key="clause_id", match=MatchAny(any=matched_clauses))
                )
            query_filter = Filter(must=conditions)
        else:
            log.debug("retriever.search.no_targeting")

        # Generate embedding
        vector = list(self.model.embed([query]))[0].tolist()

        # Search Qdrant
        results = self.q_client.query_points(
            collection_name=self.collection_name,
            query=vector,
            query_filter=query_filter,
            limit=max(cfg.fetch_k, top_k)
        ).points

        # Format results and prepare for reranking
        candidates = []
        for res in results:
            payload = res.payload
            candidates.append({
                "spec_id": payload.get("spec_id"),
                "clause_id": payload.get("clause_id"),
                "qdrant_score": round(res.score, 4),
                "content": payload.get("content", ""),
                "content_preview": str(payload.get("content", ""))[:150].replace("\n", " ") + "..."
            })

        if not candidates:
            return []

        log.debug(
            "retriever.search.candidates_before_rerank",
            count=len(candidates),
            candidates=[
                {"spec_id": c["spec_id"], "clause_id": c["clause_id"], "qdrant_score": c["qdrant_score"]}
                for c in candidates
            ],
        )

        # Rerank with Cross-Encoder
        log.debug("retriever.search.reranking", count=len(candidates))
        pairs = [(str(query), str(c["content"])) for c in candidates]
        cross_scores = self.cross_encoder.predict(pairs)

        for i, c in enumerate(candidates):
            c["score"] = round(float(cross_scores[i]), 4)

        # Sort by cross-encoder score
        candidates.sort(key=lambda x: x["score"], reverse=True)

        # Take top_k
        formatted_results = candidates[:top_k]

        log.debug(
            "retriever.search.final_results",
            top_k=top_k,
            results=[
                {"spec_id": c["spec_id"], "clause_id": c["clause_id"], "score": c["score"]}
                for c in formatted_results
            ],
        )

        return formatted_results


if __name__ == "__main__":
    retriever = Retriever()

    queries = [
        "What is the AMF?",
        "What is the AMF according to TS 23.501?",
        "Details on procedure in clause 5.5.1",
        "clause 5.5"
    ]

    for q in queries:
        results = retriever.search(q, top_k=3)
        print("Results:")
        for i, r in enumerate(results):
            print(f"  {i+1}. [Score: {r['score']}] Spec {r['spec_id']} Clause {r['clause_id']}")
            print(f"     Preview: {r['content_preview']}")
        print("-" * 60)
