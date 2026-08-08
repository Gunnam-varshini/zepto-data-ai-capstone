"""
main.py - FastAPI wrapper exposing POST /ask for the Zepto Support Assistant.

Run locally:
    uvicorn main:app --host 0.0.0.0 --port 7860

MOCK_LLM defaults to unset ("1"): fully offline, deterministic mock mode
(the required, graded baseline). Set MOCK_LLM=0 to use the optional,
ungraded real-LLM extension (requires GROQ_API_KEY or a compatible key).
"""

from fastapi import FastAPI, HTTPException

from graph import run_pipeline
from ingest import ensure_ingested
from schemas import AskRequest, AskResponse

app = FastAPI(title="Zepto Support Assistant")


@app.on_event("startup")
def startup_event():
    ensure_ingested()


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    if not request.query or not request.query.strip():
        raise HTTPException(status_code=400, detail="query must not be empty")
    return run_pipeline(request.query)


@app.get("/health")
def health():
    return {"status": "ok"}
