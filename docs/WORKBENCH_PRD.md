# QuantQuery-A Realtime Multi-Agent Workbench PRD

## Goal

Build a local, single-user portfolio project for AI Agent engineering internships. The product turns a natural-language A-share research request into a traceable workflow: route, collect bounded market evidence, run allowlisted quantitative tools, apply deterministic risk checks, and produce an evidence-linked report.

## Core loop

`request -> three-way intent route -> five-agent graph -> market evidence -> strategy/factor tools -> risk gate -> one allowlisted repair -> report -> evaluation feedback`

## Users and definition of done

- Primary user: the repository owner demonstrating agent engineering and quantitative platform skills.
- React and FastAPI run locally; FastAPI can serve the production React build.
- Five Qwen-backed roles communicate through typed graph state.
- Daily and intraday data are fetched only for a bounded watchlist and retain source, freshness, adjustment status, and fingerprints.
- MACD, Bollinger, and a price-volume factor portfolio share no-lookahead and cost-aware execution rules.
- The UI shows agent progress, evidence, charts, risk results, token usage, latency, and evaluation metrics.
- Legacy APIs remain compatible.

## Constraints

- No RAG, Chroma, real-money orders, arbitrary code execution, complete local datasets, or public deployment.
- Existing DashScope credentials are read from environment configuration and are never changed, committed, or logged.
- Runtime persistence uses SQLite; no Redis, PostgreSQL, or Docker in this version.
- Live provider failures must be visible. Cached data is labelled stale and missing evidence blocks research conclusions.

## Validation

- Existing tests remain green.
- Twenty frozen routing cases achieve at least 90% accuracy.
- Eight deterministic end-to-end cases pass in fake-LLM mode; at least seven pass in an optional bounded live-Qwen smoke run.
- Timing, cost, evidence, adjustment, token-budget, and risk gates pass 100% of applicable automated cases.

