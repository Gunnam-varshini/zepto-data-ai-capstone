"""
graph.py - LangGraph StateGraph for the Zepto Support Assistant.

Nodes:
  1. classify_intent      -> routes to "policy_question" or "general_question"
  2. retrieve_and_answer  -> for policy_question: retrieves top-3 chunks from
                              ChromaDB (always runs for real), then answers
                              (mock canned template, or real LLM if MOCK_LLM=0)
  3. direct_answer        -> for general_question: answers with no retrieval
                              (mock fixed string, or real LLM if MOCK_LLM=0)

MOCK_LLM env var:
  - unset or "1" -> fully offline, deterministic mock mode (REQUIRED, GRADED BASELINE).
                     No LLM call, no API key, no network call of any kind.
  - "0"          -> OPTIONAL, UNGRADED extension. Calls a real LLM (Groq free
                     tier by default, or any compatible free-tier endpoint).
"""

import json
import os
from typing import List, Optional, TypedDict

from langgraph.graph import END, StateGraph

from ingest import embed_texts, get_collection
from schemas import AskResponse

MOCK_LLM = os.environ.get("MOCK_LLM", "1")

POLICY_KEYWORDS = [
    "delivery", "return", "refund", "membership",
    "tracking", "cancel", "gift card", "support hours",
]

# ---------------------------------------------------------------------------
# Task 2: structured prompt template (role-context-task-format-length skeleton),
# including an explicit negative constraint and a few-shot example.
# Used only by the optional MOCK_LLM=0 real-LLM extension.
# ---------------------------------------------------------------------------
PROMPT_TEMPLATE = """ROLE:
You are Zepto's customer support assistant. You answer customer questions strictly
based on Zepto's official policy documents.

CONTEXT:
The following are the top retrieved policy excerpts relevant to the customer's question.
Treat this as the ONLY source of truth:
---
{retrieved_context}
---

TASK:
Answer the customer's question below using only the information contained in the CONTEXT above.

CUSTOMER QUESTION: {query}

FORMAT:
Respond with a single valid JSON object and nothing else, with exactly these fields:
{{"answer": "<your answer as a string>", "confidence": <float between 0 and 1>}}

LENGTH:
Keep the "answer" field to 1-3 sentences.

NEGATIVE CONSTRAINT:
Do not answer using information not present in the provided CONTEXT. If the CONTEXT does
not contain enough information to answer, say so explicitly in the "answer" field and set
"confidence" to 0.2 or lower. Do not invent policy details.

FEW-SHOT EXAMPLE:
CONTEXT: "Standard delivery is free on orders over INR 149; orders below this threshold
incur a flat INR 25 delivery fee."
CUSTOMER QUESTION: "Is delivery free?"
RESPONSE: {{"answer": "Delivery is free on orders over INR 149; orders below that incur a flat INR 25 fee.", "confidence": 0.95}}
"""

DIRECT_PROMPT_TEMPLATE = """ROLE:
You are Zepto's customer support assistant.

CONTEXT:
This question is unrelated to Zepto's delivery, returns, membership, tracking, cancellation,
damaged items, gift card, or support-hours policies, so no policy documents were retrieved.

TASK:
Politely tell the customer you can only answer questions about Zepto policies right now,
in response to: {query}

FORMAT:
Respond with a single valid JSON object and nothing else: {{"answer": "<string>"}}

LENGTH:
1 sentence.

NEGATIVE CONSTRAINT:
Do not attempt to answer the question itself. Do not invent Zepto policy information.

FEW-SHOT EXAMPLE:
CUSTOMER QUESTION: "What's the weather today?"
RESPONSE: {{"answer": "I can only answer questions about Zepto policies right now."}}
"""


class GraphState(TypedDict):
    query: str
    intent: Optional[str]
    retrieved_chunks: Optional[List[dict]]
    answer: Optional[str]
    sources: Optional[List[str]]
    confidence: Optional[float]


# ---------------------------------------------------------------------------
# Node 1: classify_intent
# ---------------------------------------------------------------------------
def classify_intent(state: GraphState) -> GraphState:
    query = state["query"]

    if MOCK_LLM != "0":
        lowered = query.lower()
        intent = "policy_question" if any(k in lowered for k in POLICY_KEYWORDS) else "general_question"
    else:
        intent = _classify_intent_llm(query)

    return {**state, "intent": intent}


def _classify_intent_llm(query: str) -> str:
    prompt = (
        "Classify the following customer question as exactly one word: "
        "'policy_question' if it relates to Zepto's delivery, returns, membership, "
        "tracking, cancellation, damaged items, gift cards, or support hours; "
        f"otherwise 'general_question'.\n\nQuestion: {query}\n\nAnswer with one word only."
    )
    try:
        raw = call_llm(prompt).strip().lower()
        return "policy_question" if "policy" in raw else "general_question"
    except Exception:
        lowered = query.lower()
        return "policy_question" if any(k in lowered for k in POLICY_KEYWORDS) else "general_question"


# ---------------------------------------------------------------------------
# Conditional edge routing
# ---------------------------------------------------------------------------
def route_from_intent(state: GraphState) -> str:
    return "retrieve_and_answer" if state["intent"] == "policy_question" else "direct_answer"


# ---------------------------------------------------------------------------
# Node 2: retrieve_and_answer
# ---------------------------------------------------------------------------
def retrieve_and_answer(state: GraphState) -> GraphState:
    query = state["query"]
    collection = get_collection()

    # Retrieval always runs for real in both modes: embedding + ChromaDB need
    # no API key and no network call.
    query_embedding = embed_texts([query])[0]
    results = collection.query(query_embeddings=[query_embedding], n_results=3)

    ids = results["ids"][0]
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]

    retrieved_chunks = [
        {"id": ids[i], "text": documents[i], "source": metadatas[i].get("source", ids[i])}
        for i in range(len(ids))
    ]
    sources = [c["id"] for c in retrieved_chunks]

    if MOCK_LLM != "0":
        top_chunk_snippet = retrieved_chunks[0]["text"][:200] if retrieved_chunks else ""
        answer = f"Based on the retrieved context: {top_chunk_snippet}"
        confidence = 1.0
    else:
        context_text = "\n---\n".join(c["text"] for c in retrieved_chunks)
        prompt = PROMPT_TEMPLATE.format(retrieved_context=context_text, query=query)
        answer, confidence = _call_llm_with_retry(prompt, default_confidence=0.8)

    return {
        **state,
        "retrieved_chunks": retrieved_chunks,
        "answer": answer,
        "sources": sources,
        "confidence": confidence,
    }


# ---------------------------------------------------------------------------
# Node 3: direct_answer
# ---------------------------------------------------------------------------
def direct_answer(state: GraphState) -> GraphState:
    query = state["query"]

    if MOCK_LLM != "0":
        answer = "I can only answer questions about Zepto policies right now."
        confidence = 1.0
    else:
        prompt = DIRECT_PROMPT_TEMPLATE.format(query=query)
        answer, confidence = _call_llm_with_retry(prompt, default_confidence=0.9, expect_confidence=False)

    return {**state, "retrieved_chunks": [], "answer": answer, "sources": [], "confidence": confidence}


# ---------------------------------------------------------------------------
# Real-LLM helper (only exercised when MOCK_LLM=0). Defaults to Groq's free
# tier (OpenAI-compatible chat completions endpoint); any other free-tier
# LLM API can be substituted via LLM_BASE_URL / LLM_MODEL / GROQ_API_KEY.
# ---------------------------------------------------------------------------
def call_llm(prompt: str) -> str:
    import requests

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("MOCK_LLM=0 requires GROQ_API_KEY (or a compatible LLM API key) to be set.")

    base_url = os.environ.get("LLM_BASE_URL", "https://api.groq.com/openai/v1/chat/completions")
    model = os.environ.get("LLM_MODEL", "llama-3.1-8b-instant")

    response = requests.post(
        base_url,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": 0},
        timeout=30,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


def _call_llm_with_retry(prompt: str, default_confidence: float, expect_confidence: bool = True, max_retries: int = 2):
    """Calls the real LLM and validates its JSON output. Retries up to
    `max_retries` additional times with a corrective instruction appended
    before giving up and returning a clearly marked error response."""
    current_prompt = prompt
    last_error = None
    for _ in range(max_retries + 1):
        try:
            raw = call_llm(current_prompt)
            cleaned = raw.strip().strip("`")
            if cleaned.lower().startswith("json"):
                cleaned = cleaned[4:].strip()
            parsed = json.loads(cleaned)

            answer = parsed["answer"]
            confidence = float(parsed.get("confidence", default_confidence)) if expect_confidence else default_confidence
            return answer, confidence
        except Exception as exc:
            last_error = exc
            current_prompt = (
                prompt
                + f"\n\nYour previous response was invalid ({last_error}). "
                "Respond again with ONLY a single valid JSON object matching the FORMAT above, no extra text."
            )
    return f"[LLM error: could not produce a valid response after retries: {last_error}]", 0.0


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------
def build_graph():
    graph = StateGraph(GraphState)
    graph.add_node("classify_intent", classify_intent)
    graph.add_node("retrieve_and_answer", retrieve_and_answer)
    graph.add_node("direct_answer", direct_answer)

    graph.set_entry_point("classify_intent")
    graph.add_conditional_edges(
        "classify_intent",
        route_from_intent,
        {"retrieve_and_answer": "retrieve_and_answer", "direct_answer": "direct_answer"},
    )
    graph.add_edge("retrieve_and_answer", END)
    graph.add_edge("direct_answer", END)

    return graph.compile()


_compiled_graph = None


def get_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph


def run_pipeline(query: str) -> AskResponse:
    graph = get_graph()
    final_state = graph.invoke({"query": query})
    return AskResponse(
        answer=final_state["answer"],
        sources=final_state.get("sources") or [],
        confidence=final_state.get("confidence", 0.0),
    )
