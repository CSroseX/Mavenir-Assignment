"""Per-spec clause centroids for triage tier 2 (embedding similarity).

One centroid per (spec_id, clause prefix at `triage_centroid_clause_depth`)
— e.g. ("23.501", "5.3") — the mean of the BGE vectors of every chunk
under that clause, re-normalised. A query's domain score is its max cosine
against all centroids. Chunks are embedded exactly as indexer.py embeds them
(lineage header + content, tables rendered by format_table_for_embedding),
with the same model, so centroids live in the space retrieval searches.

Frontmatter ("0") and the References clause ("2", a bibliography of RFC and
TS titles) are excluded by default: neither describes the 3GPP system, and
References would pull generic IETF topics (TCP, HTTP, XML) toward the domain.

Built once and cached to data/domain_centroids.npz, stamped with the
embedding model and a fingerprint of the chunk files; a stale cache is
rebuilt rather than silently used.

    python src/generation/centroids.py     # (re)build the cache
"""

from __future__ import annotations

import hashlib
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.config import get_settings  # noqa: E402
from src.obs.logging import get_logger  # noqa: E402

log = get_logger(__name__)

EmbedFn = Callable[[Sequence[str]], np.ndarray]


@dataclass
class Centroids:
    vectors: np.ndarray  # (n, d), L2-normalised
    spec_ids: np.ndarray  # (n,) str
    clause_ids: np.ndarray  # (n,) str — clause prefix
    embedding_model: str = ""
    corpus_fingerprint: str = ""

    def max_similarity(self, query_vector: np.ndarray) -> Tuple[float, str, str]:
        """(max cosine, spec_id, clause prefix) of the nearest centroid."""
        q = _normalise(np.asarray(query_vector, dtype=np.float32).reshape(1, -1))[0]
        sims = self.vectors @ q
        i = int(np.argmax(sims))
        return float(sims[i]), str(self.spec_ids[i]), str(self.clause_ids[i])


def _normalise(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return m / norms


def clause_prefix(clause_id: str, depth: int) -> str:
    return ".".join(str(clause_id).split(".")[:depth])


def embed_text(chunk: Dict[str, Any]) -> str:
    """The exact string indexer.py embeds for a chunk."""
    from src.ingestion.indexer import format_table_for_embedding

    if chunk.get("chunk_type") == "table":
        body = format_table_for_embedding(chunk["content"])
    else:
        body = chunk["content"]
    header = f"Spec {chunk['spec_id']} {chunk.get('version', '')} Clause {chunk['clause_id']} {chunk.get('clause_title', '')}\n"
    return header + str(body)


def build_centroids(chunks: List[Dict[str, Any]], embed: EmbedFn, batch_size: int = 256) -> Centroids:
    cfg = get_settings()
    depth = cfg.triage_centroid_clause_depth
    excluded = set(cfg.triage_centroid_exclude_clauses)

    keys: List[Tuple[str, str]] = []
    texts: List[str] = []
    for c in chunks:
        top = str(c["clause_id"]).split(".")[0]
        if top in excluded:
            continue
        keys.append((c["spec_id"], clause_prefix(c["clause_id"], depth)))
        texts.append(embed_text(c))
    if not texts:
        raise ValueError("No chunks left to build centroids from after exclusions.")

    vectors = []
    for i in range(0, len(texts), batch_size):
        vectors.append(np.asarray(embed(texts[i:i + batch_size]), dtype=np.float32))
        log.info("centroids.embedded", done=min(i + batch_size, len(texts)), total=len(texts))
    matrix = _normalise(np.vstack(vectors))

    groups: Dict[Tuple[str, str], List[int]] = {}
    for idx, key in enumerate(keys):
        groups.setdefault(key, []).append(idx)
    ordered = sorted(groups)
    centroid_matrix = _normalise(np.vstack([matrix[groups[k]].mean(axis=0) for k in ordered]))

    return Centroids(
        vectors=centroid_matrix.astype(np.float32),
        spec_ids=np.array([k[0] for k in ordered]),
        clause_ids=np.array([k[1] for k in ordered]),
        embedding_model=cfg.embedding_model,
    )


def corpus_fingerprint(chunk_dir: Optional[Path] = None) -> str:
    """sha256 over the chunk files plus the centroid-shaping settings, so
    changing the corpus, grouping depth or exclusions invalidates the cache."""
    cfg = get_settings()
    chunk_dir = chunk_dir or cfg.project_root / "data" / "chunks"
    h = hashlib.sha256()
    for path in sorted(chunk_dir.glob("*_chunks.json")):
        h.update(path.name.encode())
        h.update(path.read_bytes())
    h.update(f"depth={cfg.triage_centroid_clause_depth}".encode())
    h.update(f"exclude={sorted(cfg.triage_centroid_exclude_clauses)}".encode())
    return h.hexdigest()


def centroids_path() -> Path:
    cfg = get_settings()
    path = Path(cfg.triage_centroids_path)
    return path if path.is_absolute() else cfg.project_root / path


def save_centroids(c: Centroids, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(
        tmp,
        vectors=c.vectors,
        spec_ids=c.spec_ids,
        clause_ids=c.clause_ids,
        embedding_model=np.array(c.embedding_model),
        corpus_fingerprint=np.array(c.corpus_fingerprint),
    )
    os.replace(tmp, path)


def load_centroids(path: Path) -> Centroids:
    with np.load(path, allow_pickle=False) as data:
        return Centroids(
            vectors=data["vectors"],
            spec_ids=data["spec_ids"],
            clause_ids=data["clause_ids"],
            embedding_model=str(data["embedding_model"]),
            corpus_fingerprint=str(data["corpus_fingerprint"]),
        )


def get_centroids(embed: Optional[EmbedFn] = None, path: Optional[Path] = None) -> Optional[Centroids]:
    """Load the cached centroids if they match the current model and corpus;
    otherwise rebuild them with `embed` and re-cache. Returns None (tier 2
    unavailable) when there is no usable cache and no embedder."""
    cfg = get_settings()
    path = path or centroids_path()
    fingerprint = corpus_fingerprint()

    if path.exists():
        cached = load_centroids(path)
        if cached.embedding_model == cfg.embedding_model and cached.corpus_fingerprint == fingerprint:
            return cached
        log.warning("centroids.stale_cache", path=str(path), cached_model=cached.embedding_model)

    if embed is None:
        log.warning("centroids.unavailable_no_embedder", path=str(path))
        return None

    from src.generation.lexicon import load_corpus_chunks

    log.warning("centroids.building", path=str(path))
    built = build_centroids(load_corpus_chunks(), embed)
    built.corpus_fingerprint = fingerprint
    save_centroids(built, path)
    return built


def fastembed_embedder(model) -> EmbedFn:
    """Adapt a fastembed TextEmbedding (e.g. Retriever.model) to EmbedFn."""
    def embed(texts: Sequence[str]) -> np.ndarray:
        return np.vstack([np.asarray(v, dtype=np.float32) for v in model.embed(list(texts))])
    return embed


def load_default_embedder() -> EmbedFn:
    from fastembed import TextEmbedding

    return fastembed_embedder(TextEmbedding(model_name=get_settings().embedding_model))


def main() -> None:
    path = centroids_path()
    if path.exists():
        path.unlink()
    centroids = get_centroids(load_default_embedder(), path)
    print(f"Wrote {len(centroids.vectors)} centroids ({centroids.vectors.shape[1]}-dim) to {path}")


if __name__ == "__main__":
    main()
