# Zepto Data & AI Platform — Capstone Project

One connected repository, three modules:

- `/data_pipeline` (25 marks) — scrape → clean → convert → store → query, catalog-style
  pricing data from books.toscrape.com. See `data_pipeline/README.md`.
- `/analytics` (50 marks) — end-to-end EDA, cleaning, and predictive modeling pipeline
  on the Titanic dataset (Logistic Regression, Decision Tree, Random Forest, with
  hyperparameter tuning). See `analytics/README.md`.
- `/support_assistant` (25 marks) — GenAI assistant answering Zepto policy questions,
  grounded via RAG (ChromaDB + LangGraph + FastAPI). See `support_assistant/README.md`.

## Setup

Each module has its own `requirements.txt` — see that module's folder.

## Run

See each module's own README for exact run steps:
- `data_pipeline/README.md`
- `analytics/README.md`
- `support_assistant/README.md`

## Git workflow note

This repo's history includes a feature branch (`feature/analytics`) created,
committed to multiple times, and merged back into `main` via pull request #1
(visible under the repo's Pull requests tab and in `git log --graph --all`).
