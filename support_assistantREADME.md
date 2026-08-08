# Module 3 — Zepto Support Assistant (`/support_assistant`)

A small GenAI service for Zepto: an 8-document policy corpus embedded into
ChromaDB, a 3-node LangGraph pipeline that routes and answers each query, a
Pydantic-validated JSON response, and a FastAPI wrapper — all fully correct
and gradeable **offline**, with `MOCK_LLM` left at its default.

## Running locally

```bash
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 7860
```

On startup the app calls `ensure_ingested()`, which embeds all 8 corpus
documents with `all-MiniLM-L6-v2` and stores them in a persistent ChromaDB
collection (`chroma_db/`) if that collection is empty. No API key and no
network call are needed for this — the model runs locally and only downloads
its weights from Hugging Face the first time (cached afterward).

Or with Docker:

```bash
docker build -t support-assistant .
docker run -p 7860:7860 support-assistant
```

Both serve `POST /ask` at `http://localhost:7860/ask`.

## Architecture

The pipeline has four stages:

1. **Ingestion** — `ingest.py` reads all 8 `docs/doc_NN.txt` files. Given how
   short each document is, each one becomes a single chunk (a
   `CHUNK_SIZE`-based fallback splits any document that exceeds 500
   characters, though none currently do).
2. **Embedding** — also in `ingest.py`: `embed_texts()` uses
   `sentence-transformers`' `all-MiniLM-L6-v2` model, run entirely locally
   (no API key, no network call). Embeddings are stored, along with the
   chunk text and a `source` metadata field (the doc id), in the
   `zepto_policies` ChromaDB collection under `chroma_db/`.
3. **Retrieval** — the `retrieve_and_answer` node in `graph.py`. It embeds
   the incoming query with the same local model and runs a top-3 cosine
   similarity query against the ChromaDB collection. This step runs for
   real in **both** `MOCK_LLM` states, since it needs no LLM.
4. **Generation** — branches on `MOCK_LLM`:
   - **`retrieve_and_answer`** (policy questions): in mock mode (default),
     it returns a deterministic templated answer built from the first ~200
     characters of the top retrieved chunk, and sets `sources` to the ids of
     the retrieved chunks with `confidence = 1.0`. In the optional
     `MOCK_LLM=0` extension, it instead fills the Task 2 structured prompt
     template (role/context/task/format/length + negative constraint +
     few-shot example, see `PROMPT_TEMPLATE` in `graph.py`) with the
     retrieved context and calls a real LLM (Groq's free tier by default),
     validating and retrying the JSON output up to 2 extra times before
     giving up with a marked error.
   - **`direct_answer`** (general questions): in mock mode (default), it
     returns the fixed string `"I can only answer questions about Zepto
     policies right now."` with `sources = []`, `confidence = 1.0`, no LLM
     call. In the `MOCK_LLM=0` extension, it prompts the LLM directly with
     `DIRECT_PROMPT_TEMPLATE`, with no retrieval.

Routing itself — the `classify_intent` node and its conditional edge to
either `retrieve_and_answer` or `direct_answer` — does **not** depend on
`MOCK_LLM` for the graph wiring; only `classify_intent`'s own classification
step branches on it (keyword heuristic by default, an LLM call under the
`MOCK_LLM=0` extension).

```
query
  │
  ▼
[classify_intent] ── keyword heuristic (mock) / LLM (MOCK_LLM=0)
  │
  ├── policy_question ──▶ [retrieve_and_answer] ──▶ answer (templated / real LLM)
  │                         (ChromaDB top-3, always real)
  │
  └── general_question ─▶ [direct_answer] ──▶ answer (fixed string / real LLM)
```

## Example calls (recorded with `MOCK_LLM` left at its default)

**Call 1 — triggers retrieval (`policy_question`)**

Request:
```json
POST /ask
{"query": "Can I still cancel my order if it's already been packed?"}
```

Response:
```json
{
  "answer": "Based on the retrieved context: Orders can be cancelled free of cost any time before the order status changes to 'Packed', typically within the first 2 minutes of placing the order. Once an order ha",
  "sources": ["doc_05_chunk0", "doc_04_chunk0", "doc_02_chunk0"],
  "confidence": 1.0
}
```

**Call 2 — does not trigger retrieval (`general_question`)**

Request:
```json
POST /ask
{"query": "What's the capital of France?"}
```

Response:
```json
{
  "answer": "I can only answer questions about Zepto policies right now.",
  "sources": [],
  "confidence": 1.0
}
```

## Optional extensions (not required for full marks)

- **Real LLM (`MOCK_LLM=0`)**: set `GROQ_API_KEY` (free tier, no card
  required) and run with `MOCK_LLM=0`. Optionally override `LLM_BASE_URL`
  and `LLM_MODEL` to point at any other genuinely-free-tier OpenAI-compatible
  chat completions endpoint.
- **Hugging Face Spaces deployment**: the same `Dockerfile` can be pushed to
  a free CPU Space, with `GROQ_API_KEY` stored as a Space secret (never
  hardcoded). Not attempted in this submission — the required baseline
  (local `docker build` + `docker run`) is what's graded.
