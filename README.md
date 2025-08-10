# FinDocGPT

Evidence-grounded financial document analysis → short-term return forecast → rule-based BUY/SELL/HOLD with risk controls.  
Built for hackathons: fast, explainable, and demo-ready.

---

## Who this helps

- **Equity analysts / PMs** – rapid triage of filings/letters with citations, quick baseline forecast, and an auditable action.
- **Fintech builders** – a clean, modular RAG + forecasting scaffold with strict JSON outputs and a simple UI.
- **Founders & product managers** – a working demo that converts raw docs into decision-ready signals, with a chat explainer.
- **Hackathon judges** – transparent pipeline, reproducible outputs, evidence snippets, and downloadable JSON.

---

## Features

- **Stage 1 – Docs & Q&A**
  - Upload PDFs/TXTs; Q&A via **Embeddings + FAISS** (cached to disk) or **GPT-4.1 synthesis** returning **strict JSON**.
  - **Top-k evidence** with similarity scores for credibility.
  - **Sentiment** (overall summary) and **Anomalies** (simple z-score flags).
  - **PDF table extraction** (experimental): `pdfplumber` (default), optional Camelot.

- **Stage 2 – Forecast**
  - ARIMA on daily returns (from `yfinance`) with **80/95% intervals**.
  - **Backtest metrics** (RMSE / sMAPE / MAPE). _Lower RMSE = better._

- **Stage 3 – Decision**
  - Rule-based **BUY/SELL/HOLD**, **position size**, **stop-loss / take-profit**, **review date**.
  - Uses forecast signal + **sentiment** as a secondary (exogenous) signal.
  - **Rationale bullets** (optional GPT-4.1 polish).

- **Run All – Demo Pipeline**
  - One click: docs → Q&A → sentiment → anomalies → forecast → decision.
  - **Forecast chart** (Plotly) + **Download all results (JSON)**.

- **Chat – Decision Q&A (GPT-4.1)**
  - Ask about the **action, risks, backtest, or framework**; grounded in the latest run.
  - Clear guardrails: explain & educate; **no investment advice**.

- **Performance & DX**
  - **FAISS index persistence** (`data/index_cache/`) for fast repeat queries.
  - Strict JSON contracts across stages → easy to test, audit, and extend.

---

## Architecture (high level)

1. **Retrieval:** sentence-transformers (`all-MiniLM-L6-v2`) → FAISS (cosine) → **top-k passages** (cached to disk).  
2. **Q&A:**
   - Default: TF-IDF fallback (no external deps).
   - Embeddings: extract answer span from best passage.
   - GPT-4.1 (optional): **synthesizes** final answer over top-k, returns **strict JSON** (Responses API or Chat Completions JSON fallback).
3. **Sentiment & Anomalies:** quick summarization + z-score flags on user series.
4. **Forecast:** ARIMA on daily returns (AAPL default) → forecast + **backtest**.
5. **Decision:** deterministic rules (mean/interval + risk flags + sentiment adjust) → action, size, stops/TP, review date.
6. **Chat:** GPT-4.1 explains decision & framework using the **latest run** (session state).

_All stage outputs are JSON and visible in the UI._

---

## Quickstart

```bash
# 1) Create & activate a virtualenv (Windows PowerShell shown)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2) Install
pip install -r requirements.txt
# If anything is missing, also:
pip install sentence-transformers faiss-cpu pdfplumber openai yfinance plotly

# 3) (Optional) Enable GPT-4.1 features
$env:OPENAI_API_KEY="sk-...your-key..."

# 4) Run the app
streamlit run .\app\streamlit_app.py

## How to use (UI pages)
Stage 1: Docs & Q&A
Upload PDFs/TXTs or click Use sample docs.

Tick Use Embeddings (FAISS) for better retrieval; optionally tick Use GPT-4.1 synthesis for concise answers.

Click Run Q&A → see the answer + Top-k evidence with similarity scores.

Click Analyze sentiment (overall) and Detect anomalies (enter a small series).

Extract tables from PDFs (experimental) to preview parsed numeric tables.

Stage 2: Forecast
Set Ticker / Start / End / Horizon → Fetch & Forecast.

View Forecast JSON + Chart (mean + 80/95 bands).

Backtest hint: Lower RMSE is better (sMAPE/MAPE are % errors; lower is better).

Stage 3: Decision
Click Decide → action, position size, stops/TP, review date.

See rationale bullets (and optionally refine with GPT-4.1).

Download all results (JSON).

Run All: Demo Pipeline
Fill Question / Ticker / Dates / Horizon → Run end-to-end.

Shows all JSON blocks, the forecast chart, and a single Download JSON.

Chat: Decision Q&A
Ask about the decision and the framework (“Why HOLD?”, “How do RMSE/sMAPE affect confidence?”, “Which risks were flagged?”).
Grounded in the latest run (no advice).

GPT-4.1 integration (what it adds)
Better summarized answers over your evidence with strict JSON output (schema via Responses API, or JSON mode fallback).

Decision explanations in the Chat page (and optional rationale polishing in Stage 3).

Keeps trading logic deterministic (rules + signals); GPT clarifies and cites context.

License & Disclaimer
License: MIT (or your preferred license).

Disclaimer: Educational demo. 