import re
import sys
import json
import glob
import os

try:
    from fastembed import TextEmbedding
    from qdrant_client import QdrantClient
    from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchAny
    from sentence_transformers import CrossEncoder
except ImportError:
    print("Missing dependencies. Please run: pip install qdrant-client fastembed sentence-transformers")
    sys.exit(1)

class Retriever:
    def __init__(self, collection_name="3gpp_specs", qdrant_url="http://localhost:6333"):
        self.collection_name = collection_name
        self.q_client = QdrantClient(url=qdrant_url, check_compatibility=False)
        self.model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
        print("Loading CrossEncoder...")
        self.cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        print("CrossEncoder loaded.")
        
        # Regex patterns for query routing
        self.spec_pattern = re.compile(r"(?i)(?:TS\s*)?\b(\d{2}\.\d{3})\b")
        self.clause_pattern = re.compile(r"(?i)(?:clause|sec|section|subclause)\s+([1-9A-Z][\d\.a-zA-Z]*)\b")
        
        # Load all valid clause IDs for prefix matching
        self.valid_clause_ids = set()
        chunk_files = glob.glob("data/chunks/*_chunks.json")
        for f in chunk_files:
            with open(f, "r", encoding="utf-8") as file:
                chunks = json.load(file)
                for c in chunks:
                    self.valid_clause_ids.add(c["clause_id"])

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

    def search(self, query: str, top_k: int = 5):
        """
        Executes a dual-path retrieval based on query parsing.
        """
        spec_id, clause_id = self._parse_query(query)
        
        # Determine Path A (Default) vs Path B (Targeted)
        is_targeted = bool(spec_id or clause_id)
        path_name = "TARGETED PATH" if is_targeted else "DEFAULT PATH"
        
        print(f"\n--- {path_name} Execution ---")
        print(f"Query: '{query}'")
        
        # Build Filter if Targeted
        query_filter = None
        if is_targeted:
            conditions = []
            if spec_id:
                print(f"Detected targeting -> Spec: {spec_id}")
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
                    
                print(f"Detected targeting -> Clause: {clause_id} (Expanded to {len(matched_clauses)} child clauses)")
                conditions.append(
                    FieldCondition(key="clause_id", match=MatchAny(any=matched_clauses))
                )
            query_filter = Filter(must=conditions)
        else:
            print("No targeting detected. Running full cross-spec semantic search.")
            
        # Generate embedding
        vector = list(self.model.embed([query]))[0].tolist()
        
        # Search Qdrant
        results = self.q_client.query_points(
            collection_name=self.collection_name,
            query=vector,
            query_filter=query_filter,
            limit=max(20, top_k)
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
            
        print(f"\n[DEBUG] --- Candidates retrieved before reranking (Top {len(candidates)}) ---")
        for i, c in enumerate(candidates):
            print(f"[{i+1}] Spec {c['spec_id']} Clause {c['clause_id']} | Qdrant Score: {c['qdrant_score']}")
        print("------------------------------------------------------------\n")
            
        # Rerank with Cross-Encoder
        print(f"Reranking {len(candidates)} candidates...")
        pairs = [(str(query), str(c["content"])) for c in candidates]
        cross_scores = self.cross_encoder.predict(pairs)
        
        for i, c in enumerate(candidates):
            c["score"] = round(float(cross_scores[i]), 4)
            
        # Sort by cross-encoder score
        candidates.sort(key=lambda x: x["score"], reverse=True)
        
        # Take top_k
        formatted_results = candidates[:top_k]
        
        print(f"\n[DEBUG] --- Final Top {top_k} Candidates after reranking ---")
        for i, c in enumerate(formatted_results):
            print(f"[{i+1}] Spec {c['spec_id']} Clause {c['clause_id']} | Cross-Encoder Score: {c['score']}")
        print("----------------------------------------------------------\n")
            
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
