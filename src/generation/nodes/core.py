import json
import os
import re
from typing import Any, Dict

from dotenv import load_dotenv
from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI

from src.config import get_settings
from src.obs.logging import get_logger

cfg = get_settings()
log = get_logger(__name__)

load_dotenv(dotenv_path=str(cfg.project_root / ".env"))

# Re-read settings after dotenv loads (env vars may have changed)
get_settings.cache_clear()
cfg = get_settings()

# Unified client for Ollama or Remote API
if cfg.use_remote_llm:
    openrouter_key = cfg.open_router_api_key
    openai_key = cfg.openai_api_key
    # OPENAI_BASE_URL is a routing signal: empty means "use the provider default"
    preferred_base_url = os.environ.get("OPENAI_BASE_URL", "")

    if openrouter_key and (not openai_key or "openrouter.ai" in preferred_base_url.lower()):
        api_key = openrouter_key
        base_url = preferred_base_url or cfg.openrouter_base_url
        default_model = cfg.remote_llm_model_openrouter
    else:
        api_key = openai_key or "dummy"
        base_url = preferred_base_url or cfg.openai_base_url
        default_model = cfg.remote_llm_model_openai

    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
    )
    MODEL_NAME = cfg.model_name or default_model
else:
    # Local Ollama endpoint
    client = OpenAI(
        api_key="ollama",  # required but ignored
        base_url=cfg.ollama_base_url,
    )
    MODEL_NAME = cfg.local_llm_model

# Word-boundary regexes: bare substring matching on "retrieved"/"chunks"
# false-positives on legitimate 3GPP prose (e.g. "the UE retrieved the
# subscription data"). These patterns only catch the system talking about
# its own retrieval mechanism, not the domain's own use of these words.
BANNED_PHRASE_PATTERNS = [
    re.compile(r"\bprovided source\b", re.IGNORECASE),
    re.compile(r"\bprovided chunks?\b", re.IGNORECASE),
    re.compile(r"\bprovided text\b", re.IGNORECASE),
    re.compile(r"\bprovided specifications?\b", re.IGNORECASE),
    re.compile(r"\bsource chunks?\b", re.IGNORECASE),
    re.compile(r"\bretrieved (?:information|context|text|chunks?|documents?)\b", re.IGNORECASE),
    re.compile(r"\bthe chunks?\b", re.IGNORECASE),
]


def _detect_banned_phrase(answer: str) -> str | None:
    for pattern in BANNED_PHRASE_PATTERNS:
        match = pattern.search(answer)
        if match:
            return match.group(0)
    return None


def _build_generate_messages(query: str, chunks, feedback: str, leak_rephrase: bool = False):
    chunk_text = "\n\n".join([f"--- CHUNK {c['clause_id']} (Spec {c['spec_id']}) ---\n{c['content']}" for c in chunks])

    system_prompt = (
        "You are a senior 3GPP standards expert with deep knowledge of 4G and 5G core network architecture, "
        "NAS procedures, and mobility management. Answer the user's question directly and confidently, "
        "the way a telecom engineer would explain it to a colleague.\n\n"
        "Rules:\n"
        "- Never mention \"chunks,\" \"provided source,\" \"provided text,\" \"provided specifications,\" "
        "\"retrieved information,\" or any variation referring to how you got this information. "
        "The user should never know this is a retrieval system.\n"
        "- If the available information is incomplete or doesn't cover part of the question, say so in natural "
        "expert language — e.g. \"The exact mechanism for X isn't specified for this scenario\" or \"This detail "
        "isn't defined in the 5G specifications for this procedure\" — never \"the provided chunks don't mention X.\"\n"
        "- Do not fabricate details not present in your context. Speak with the confidence of an expert citing "
        "what's known, and plainly note what isn't, without exposing that this is a limitation of retrieved text.\n"
        "- Do not use citation brackets like [1]."
    )

    user_prompt = f"<SOURCE CHUNKS>\n{chunk_text}\n</SOURCE CHUNKS>\n\n<QUERY>\n{query}\n</QUERY>"

    if feedback:
        user_prompt += f"\n\n<FEEDBACK FROM PREVIOUS ATTEMPT>\n{feedback}\nPlease revise your answer to ensure all claims are strictly supported by the sources.\n</FEEDBACK FROM PREVIOUS ATTEMPT>"

    if leak_rephrase:
        user_prompt += (
            "\n\n<REPHRASE REQUIRED>\nYour previous answer referred to how this information was obtained "
            "(e.g. mentioning \"chunks\" or \"retrieved\" text). Rephrase the same factual content as confident "
            "expert knowledge, without any reference to sources, retrieval, or documents.\n</REPHRASE REQUIRED>"
        )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]


def generate_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generates a natural prose answer based on the query and retrieved chunks.
    Does NOT format claims or citations.

    Owns the `attempt` counter: this node increments it, not the verifier,
    so a first-pass success records attempt == 1 (previously the verifier
    incremented `retries` on every call including the first, so a first-pass
    success recorded retries == 1).
    """
    query = state["query"]
    chunks = state["chunks"]
    feedback = state.get("feedback", "")
    attempt = state.get("attempt", 0) + 1

    messages_payload = _build_generate_messages(query, chunks, feedback)
    log.debug("generate_node.payload", messages=messages_payload, attempt=attempt)

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages_payload,
        temperature=cfg.generate_temperature
    )
    answer = response.choices[0].message.content

    # Safety net: strip out any <think> blocks that might leak through
    answer = re.sub(r'<think>.*?</think>', '', answer, flags=re.DOTALL).strip()

    # Banned-phrase gate: a real gate, not just a log line. One targeted
    # rephrase, then let the leak stand (the caller/verifier path handles
    # it from there — this node does not loop on itself).
    leaked_phrase = _detect_banned_phrase(answer)
    if leaked_phrase:
        log.warning("generate_node.banned_phrase_leak", phrase=leaked_phrase, attempt=attempt)
        rephrase_messages = _build_generate_messages(query, chunks, feedback, leak_rephrase=True)
        rephrase_response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=rephrase_messages,
            temperature=cfg.generate_temperature
        )
        rephrased = re.sub(
            r'<think>.*?</think>', '', rephrase_response.choices[0].message.content, flags=re.DOTALL
        ).strip()
        if not _detect_banned_phrase(rephrased):
            answer = rephrased
        else:
            log.warning("generate_node.banned_phrase_leak_persisted", attempt=attempt)

    return {"answer": answer, "attempt": attempt, "retries": max(attempt - 1, 0)}

def _build_verify_prompt(answer: str, chunks) -> str:
    chunk_text = "\n\n".join([f"--- CHUNK {c['clause_id']} (Spec {c['spec_id']}) ---\n{c['content']}" for c in chunks])
    return f"""You are a strict, impartial fact-checker. You will be provided with an ANSWER generated by an AI assistant, and a set of SOURCE CHUNKS retrieved from 3GPP telecom specifications.

Verify if every factual claim in the ANSWER is strictly supported by the SOURCE CHUNKS.
Do NOT use outside knowledge. Strict entailment is required. If a claim is not supported, you must fail the verification.

<SOURCE CHUNKS>
{chunk_text}
</SOURCE CHUNKS>

<ANSWER>
{answer}
</ANSWER>

Evaluate the answer step-by-step:
1. Extract the core claims made in the ANSWER.
2. For each claim, check if it is explicitly stated or strictly entailed by the SOURCE CHUNKS.
3. If ANY claim is unsupported or contradicts the chunks, fail the verification.

Provide your output in the following JSON format ONLY:
{{
  "is_supported": boolean,
  "reasoning": "brief explanation of your judgement",
  "unsupported_claims": [
    "list any claims that were not supported by the sources, or leave empty if all are supported"
  ]
}}"""


def _call_verifier(prompt: str):
    """One raw call to the verifier LLM. Raises on transport/API failure;
    raises json.JSONDecodeError on malformed JSON. Caller interprets both."""
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=cfg.verify_temperature,
        max_tokens=cfg.verify_max_tokens
    )
    raw_json = response.choices[0].message.content
    return json.loads(raw_json)


def verify_claims_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Acts as a strict judge to ensure every claim in the answer is supported by the chunks.

    Typed exception boundary, pure with respect to the grounding-retry
    counter: only a valid JSON response with is_supported: false represents
    a genuine grounding failure and increments grounding_retries. A
    transport failure (rate limit, connection, timeout) or malformed JSON
    is an operational hiccup, not evidence the answer is wrong — it must
    not be indistinguishable from one in the returned state.
    """
    answer = state["answer"]
    chunks = state["chunks"]
    grounding_retries = state.get("grounding_retries", 0)
    transport_attempts = state.get("transport_attempts", 0)

    prompt = _build_verify_prompt(answer, chunks)

    verification = None
    transport_failure = False

    try:
        verification = _call_verifier(prompt)
    except json.JSONDecodeError as e:
        # Malformed JSON: one reask, then treat as transport.
        log.warning("verify_claims_node.malformed_json_reask", error=str(e))
        transport_attempts += 1
        try:
            verification = _call_verifier(prompt)
        except json.JSONDecodeError as e2:
            log.error("verify_claims_node.malformed_json_after_reask", error=str(e2))
            transport_failure = True
            transport_attempts += 1
        except (APIStatusError, APIConnectionError, APITimeoutError) as e2:
            log.error("verify_claims_node.transport_error_after_reask", error=str(e2))
            transport_failure = True
            transport_attempts += 1
    except (APIStatusError, APIConnectionError, APITimeoutError) as e:
        log.error("verify_claims_node.transport_error", error=str(e))
        transport_failure = True
        transport_attempts += 1

    if transport_failure:
        return {
            "verification_passed": False,
            "feedback": "",
            "retries": grounding_retries,  # legacy field: app.py / eval scripts
            "grounding_retries": grounding_retries,
            "transport_attempts": transport_attempts,
            "termination_reason": "verifier_unavailable",
        }

    is_supported = verification.get("is_supported", False)
    unsupported = verification.get("unsupported_claims", [])

    feedback = ""
    if not is_supported and unsupported:
        feedback = "The following claims were NOT supported by the source text and must be removed or corrected:\n- " + "\n- ".join(unsupported)

    new_grounding_retries = grounding_retries + (0 if is_supported else 1)

    return {
        "verification_passed": is_supported,
        "feedback": feedback,
        "retries": new_grounding_retries,  # legacy field: app.py / eval scripts
        "grounding_retries": new_grounding_retries,
        "transport_attempts": transport_attempts,
        "termination_reason": "verified" if is_supported else None,
    }
