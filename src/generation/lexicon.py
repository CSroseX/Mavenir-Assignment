"""Corpus-derived domain lexicon shared by the Phase 4 decision-nodes.

Everything here is harvested from data/chunks/*_chunks.json — there is no
hand-written list of domain keywords. Three sources feed it:

1. Acronym definitions. Each spec's "3.2 Abbreviations" table, plus every
   in-text "Long Form Words (LF)" definition, validated by a word-initials
   long-form matcher (find_long_form). The abbreviation tables alone are not
   trustworthy: Docling merges adjacent rows in places (e.g. one 23.501 row
   reads NSSAI -> "Network Slice Selection Assistance Network Slice
   Selection Function"), so the in-text definitions outvote them.
2. Acronym-shaped tokens (UE, NAS, gNB, QoS...) that occur often enough in
   spec text, even when the specs never define them (they inherit
   TR 21.905's). A token whose lowercase form is itself common in the spec
   prose (NOTE -> "note") is plain English set in caps, and is dropped.
3. Clause-title bigrams ("session establishment", "tracking area").

Per-generation affinity (4G vs 5G) is also measured, not asserted: an
acronym is 4G-affine when its rate in the EPS spec(s) dwarfs its rate in
the 5G specs (MME, EMM, ESM), and vice versa (AMF, UDM, SMF).
"""

from __future__ import annotations

import glob
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from src.config import get_settings

TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9-]*")
_PAREN_DEF_RE = re.compile(r"\(([A-Za-z0-9][A-Za-z0-9-]{1,14})\)")
_TITLE_WORD_RE = re.compile(r"[a-z]{3,}")

# English function words. These are grammar, not domain vocabulary — they
# only stop a clause-title bigram like "of the" from becoming a domain term.
FUNCTION_WORDS = frozenset(
    "the and for with from into onto over under about this that these those then than "
    "not are was were has have had been being its their there which what when where who "
    "how why does did can could should would will shall may might must per via all any "
    "each other some such only also both".split()
)


def is_acronym_shaped(token: str) -> bool:
    """AMF, 5GS, gNB, QoS, 5G-GUTI, NG-RAN — but not What, France, 2022."""
    if len(token) < 2 or not any(ch.isupper() for ch in token):
        return False
    if re.fullmatch(r"[A-Z][a-z]+", token):
        return False
    return sum(1 for ch in token if ch.isupper() or ch.isdigit()) >= 2


def _spell(words: List[str], target: str) -> bool:
    """Can `words`, in order, spell `target`? Each content word spells its
    initial (an acronym-shaped word may spell a run of characters, so
    "5GS Mobility Management" spells 5GMM); a function word may spell its
    initial or nothing (QoS <- "Quality of Service"); at most one interior
    content word may be skipped (AMF <- "Access and Mobility Management
    Function"). The first and last words must both spell something."""
    parts: List[str] = [p for w in words for p in w.split("-") if p]
    if not parts:
        return False

    @lru_cache(maxsize=None)
    def go(i: int, pos: int, skipped: bool) -> bool:
        if i == len(parts):
            return pos == len(target)
        part = parts[i]
        last = i == len(parts) - 1
        options: List[Tuple[int, int, bool]] = []
        if pos < len(target):
            if is_acronym_shaped(part):
                p = part.lower()
                take = 0
                while take < len(p) and pos + take < len(target) and p[take] == target[pos + take]:
                    take += 1
                for t in range(take, 0, -1):
                    options.append((i + 1, pos + t, skipped))
            elif part[0].lower() == target[pos]:
                options.append((i + 1, pos + 1, skipped))
        if 0 < i and not last:
            if part.lower() in FUNCTION_WORDS:
                options.append((i + 1, pos, skipped))
            elif not skipped:
                options.append((i + 1, pos, True))
        return any(go(*o) for o in options)

    return go(0, 0, False)


def find_long_form(short: str, text: str) -> Optional[str]:
    """Shortest run of words at the end of `text` that spells `short`
    (see _spell), or None. Word-initial matching is deliberately stricter
    than Schwartz-Hearst's character matching, which resolves MME to
    "Management Entity" and "(NOTE)" to whatever words precede it."""
    target = "".join(ch for ch in short.lower() if ch.isalnum())
    if len(target) < 2:
        return None
    words = text.split()
    max_words = len(target) + 3
    for start in range(len(words) - 1, max(-1, len(words) - 1 - max_words), -1):
        candidate = words[start:]
        if candidate[0].lower() in FUNCTION_WORDS or len(candidate) < 2:
            continue
        if _spell(candidate, target):
            long_form = " ".join(candidate).strip(" ,;:")
            if short.lower() not in long_form.lower().split():
                return long_form
    return None



def _content_text(chunk: Dict[str, Any]) -> str:
    content = chunk.get("content", "")
    if isinstance(content, list):
        return "\n".join(" ".join(str(cell) for cell in row) for row in content)
    return str(content)


@dataclass
class DomainLexicon:
    long_forms: Dict[str, str] = field(default_factory=dict)  # acronym -> best long form
    acronyms: Set[str] = field(default_factory=set)  # case-sensitive domain acronyms
    acronym_freq: Dict[str, int] = field(default_factory=dict)
    title_bigrams: Set[Tuple[str, str]] = field(default_factory=set)
    generation: Dict[str, str] = field(default_factory=dict)  # acronym -> "4g" | "5g"
    spec_ids: Set[str] = field(default_factory=set)
    clause_ids: Set[str] = field(default_factory=set)

    # ── derived lookups ─────────────────────────────────────────────
    def __post_init__(self) -> None:
        self._long_form_index: Dict[str, str] = {}
        self._long_form_re: Optional[re.Pattern] = None

    def _ensure_long_form_index(self) -> None:
        if self._long_form_re is not None or not self.long_forms:
            return
        self._long_form_index = {lf.lower(): ac for ac, lf in self.long_forms.items()}
        # Longest first so "PDU Session Establishment" style phrases win
        # over any shorter long form they contain.
        phrases = sorted(self._long_form_index, key=len, reverse=True)
        self._long_form_re = re.compile(
            r"(?<![A-Za-z0-9])(" + "|".join(re.escape(p) for p in phrases) + r")(?![A-Za-z0-9])",
            re.IGNORECASE,
        )

    # ── query-side matching ─────────────────────────────────────────
    def find_acronyms(self, text: str) -> List[str]:
        """Domain acronyms present in text, case-sensitive, first-seen order."""
        seen: List[str] = []
        for tok in TOKEN_RE.findall(text):
            tok = tok.rstrip("-")
            if tok in self.acronyms and tok not in seen:
                seen.append(tok)
        return seen

    def find_long_forms(self, text: str) -> List[Tuple[str, str]]:
        """(matched phrase as written, acronym) for every long form in text."""
        self._ensure_long_form_index()
        if self._long_form_re is None:
            return []
        out: List[Tuple[str, str]] = []
        for m in self._long_form_re.finditer(text):
            out.append((m.group(1), self._long_form_index[m.group(1).lower()]))
        return out

    def find_title_bigrams(self, text: str) -> List[str]:
        words = re.findall(r"[a-z0-9-]+", text.lower())
        hits: List[str] = []
        for a, b in zip(words, words[1:]):
            if (a, b) in self.title_bigrams:
                hits.append(f"{a} {b}")
        return hits

    def generation_of(self, acronym: str) -> Optional[str]:
        # Naming convention, not a keyword list: 3GPP prefixes generation-
        # specific identifiers with their generation (5G-GUTI, 5GMM, 4G-GUTI).
        if acronym.upper().startswith("5G"):
            return "5g"
        if acronym.upper().startswith("4G"):
            return "4g"
        return self.generation.get(acronym)


def _iter_chunk_files(chunk_dir: Path) -> List[str]:
    return sorted(glob.glob(str(chunk_dir / "*_chunks.json")))


def build_lexicon(chunks: Iterable[Dict[str, Any]]) -> DomainLexicon:
    cfg = get_settings()
    min_freq = cfg.lexicon_min_term_freq

    long_form_votes: Dict[str, Counter] = defaultdict(Counter)
    long_form_display: Dict[Tuple[str, str], str] = {}
    token_counts: Counter = Counter()
    lower_word_counts: Counter = Counter()
    per_spec_counts: Dict[str, Counter] = defaultdict(Counter)
    per_spec_tokens: Counter = Counter()
    title_bigrams: Set[Tuple[str, str]] = set()
    spec_ids: Set[str] = set()
    clause_ids: Set[str] = set()

    def vote(short: str, long_form: str, weight: int = 1) -> None:
        key = long_form.lower()
        long_form_votes[short][key] += weight
        long_form_display.setdefault((short, key), long_form)

    for chunk in chunks:
        spec_id = chunk.get("spec_id", "")
        spec_ids.add(spec_id)
        clause_ids.add(str(chunk.get("clause_id", "")))
        title = chunk.get("clause_title") or ""
        if "change history" in title.lower():
            # Release-history annex: CR numbers (CP-090674) and version
            # tags, not domain vocabulary.
            continue
        text = _content_text(chunk)

        tokens = TOKEN_RE.findall(text)
        token_counts.update(tokens)
        per_spec_counts[spec_id].update(tokens)
        per_spec_tokens[spec_id] += len(tokens)
        lower_word_counts.update(t.lower() for t in tokens if t.islower() or re.fullmatch(r"[A-Z][a-z]+", t))

        title_words = _TITLE_WORD_RE.findall(title.lower())
        for a, b in zip(title_words, title_words[1:]):
            if a not in FUNCTION_WORDS and b not in FUNCTION_WORDS:
                title_bigrams.add((a, b))

        # Abbreviation tables: [short, expansion] rows, validated the same
        # way as in-text definitions so merged rows can't poison the vote.
        if "abbreviation" in title.lower() and isinstance(chunk.get("content"), list):
            for row in chunk["content"]:
                if len(row) < 2 or not row[0] or not row[1]:
                    continue
                short, expansion = str(row[0]).strip(), str(row[1]).strip()
                if " " in short or not is_acronym_shaped(short):
                    continue
                # A merged row ("Network Slice Selection Assistance Network
                # Slice Selection Function" for NSSAI) fails to spell its
                # acronym and is skipped instead of voted for.
                lf = find_long_form(short, expansion)
                if lf:
                    vote(short, lf)

        # In-text "Long Form Words (LF)" definitions.
        for m in _PAREN_DEF_RE.finditer(text):
            short = m.group(1)
            if not is_acronym_shaped(short):
                continue
            window_words = text[: m.start()].split()[-(len(short) + 4):]
            lf = find_long_form(short, " ".join(window_words))
            if lf:
                vote(short, lf)

    long_forms: Dict[str, str] = {}
    for short, votes in long_form_votes.items():
        best_key, _ = votes.most_common(1)[0]
        long_forms[short] = long_form_display[(short, best_key)]

    acronyms: Set[str] = set()
    acronym_freq: Dict[str, int] = {}
    for tok, count in token_counts.items():
        if count < min_freq or not is_acronym_shaped(tok):
            continue
        # NOTE / EXAMPLE are English set in caps, not acronyms: their
        # lowercase/Titlecase forms are ordinary words in the same prose.
        if lower_word_counts.get(tok.lower(), 0) >= min_freq:
            continue
        acronyms.add(tok)
        acronym_freq[tok] = count
    # Only keep long forms for acronyms that pass the frequency gate, so a
    # one-off definition can't make a generic term (XML, HTTP) look domain.
    long_forms = {k: v for k, v in long_forms.items() if k in acronyms}

    generation = _generation_affinity(acronyms, per_spec_counts, per_spec_tokens)

    return DomainLexicon(
        long_forms=long_forms,
        acronyms=acronyms,
        acronym_freq=acronym_freq,
        title_bigrams=title_bigrams,
        generation=generation,
        spec_ids=spec_ids,
        clause_ids=clause_ids,
    )


def _generation_affinity(
    acronyms: Set[str], per_spec_counts: Dict[str, Counter], per_spec_tokens: Counter
) -> Dict[str, str]:
    cfg = get_settings()
    eps_specs = set(cfg.rewrite_4g_spec_ids)
    fivegs_specs = set(cfg.rewrite_5g_spec_ids)
    n4 = sum(per_spec_tokens[s] for s in eps_specs) or 1
    n5 = sum(per_spec_tokens[s] for s in fivegs_specs) or 1
    ratio = cfg.rewrite_generation_affinity_ratio

    out: Dict[str, str] = {}
    for ac in acronyms:
        c4 = sum(per_spec_counts[s][ac] for s in eps_specs)
        c5 = sum(per_spec_counts[s][ac] for s in fivegs_specs)
        if c4 + c5 < cfg.lexicon_min_term_freq:
            continue
        rate4 = (c4 + 1) / n4
        rate5 = (c5 + 1) / n5
        if rate4 / rate5 >= ratio:
            out[ac] = "4g"
        elif rate5 / rate4 >= ratio:
            out[ac] = "5g"
    return out


def load_corpus_chunks(chunk_dir: Optional[Path] = None) -> List[Dict[str, Any]]:
    chunk_dir = chunk_dir or get_settings().project_root / "data" / "chunks"
    chunks: List[Dict[str, Any]] = []
    for path in _iter_chunk_files(chunk_dir):
        with open(path, "r", encoding="utf-8") as f:
            chunks.extend(json.load(f))
    if not chunks:
        raise FileNotFoundError(
            f"No chunk files found in {chunk_dir}. Run the ingestion pipeline first (see CLAUDE.md)."
        )
    return chunks


@lru_cache(maxsize=1)
def get_lexicon() -> DomainLexicon:
    """Process-wide lexicon, built once from the chunk files on disk."""
    return build_lexicon(load_corpus_chunks())
