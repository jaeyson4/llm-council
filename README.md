# LLM Council — Equity Research Edition

An AI-powered stock research tool that puts several frontier LLMs on a "council," feeds them live market data, and has them debate and rank investment ideas together. A Chairman model synthesizes the council's work into a single, structured analysis.

Forked from and inspired by [karpathy/llm-council](https://github.com/karpathy/llm-council). The original is a general-purpose multi-LLM Q&A app; this fork rebuilds it into a disciplined equity-research pipeline with live data, external signals, and note-taking.

> **This is a research and learning tool, not financial advice.** It organizes analysis and forces disciplined thinking; it does not predict the future. See [Limitations](#limitations) before doing anything with its output.

## What it does

Instead of asking one model "what stocks look good," this runs a structured pipeline that gathers real data first, then has four different models analyze it independently, critique each other, and produce a synthesized verdict — all saved to an Obsidian vault as a dated research journal.

## How it works

1. **Stage A — Screening (cheap).** One inexpensive model (`gemini-3.5-flash`) does a top-down macro → sector → candidate pass and returns a shortlist of tickers. Premium models are deliberately not used here to control cost.
2. **Data enrichment (Python, deterministic).** For every shortlisted ticker, the app pulls data from yfinance and computes — in code, not by the LLM — valuation multiples (P/E, P/S, EV/EBITDA), growth CAGRs, margins, free cash flow, valuation percentiles vs. the stock's own history, forward base rates, and max drawdown.
3. **Catalyst & thematic layer (external signals).**
   - **Insider activity:** SEC EDGAR Form 4 filings, parsed to compute net insider buying vs. selling and flag cluster buying.
   - **Policy/regulatory news:** recent, dated items via the Tavily search API, bounded to control cost.
   - **Tech-trend / leadership context:** low-confidence signal, clearly labeled.
   - Every news-based catalyst must be classified by the models as a *durable structural driver* vs. *already priced in*, to resist hype-chasing.
4. **Stage B — Deep dive (the full council).** Four premium models (`gpt-5.5`, `gemini-3.1-pro-preview`, `claude-fable-5`, `grok-4.3`) each independently analyze the shortlist with all of the above data injected, producing a fixed structure per pick: bull thesis, bear thesis, key numbers, bear/base/bull 2-year price targets with per-stock assumptions, bull-case probability, biggest risk, and a conviction score.
5. **Adversarial peer review.** Each model reviews the others' work — attacking the weakest pick and naming what would break the strongest — rather than politely agreeing.
6. **Stage C — Chairman synthesis.** A Chairman model reconciles the council, corrects errors the review surfaced, and produces the final ranked answer.
7. **Obsidian export.** Each analysis is written to an Obsidian vault as a dated, linked markdown note (ticker, date, price, targets, thesis) — building a searchable research journal over time.

## Features

- Two-stage funnel (cheap screen → premium deep dive) to control credits
- Live market data computed deterministically in Python (not hallucinated)
- SEC insider-activity and policy-news signal layers
- Per-stock scenario assumptions (not one-size-fits-all targets)
- Adversarial peer review to surface real disagreement
- Automatic export to an Obsidian research vault
- Cost controls throughout (token caps that account for reasoning tokens, bounded API calls, shared data cache)

## Setup

Requires [uv](https://docs.astral.sh/uv/), Node.js, and Python 3.10+.

**Backend:**
```bash
uv sync
```

**Frontend:**
```bash
cd frontend
npm install
cd ..
```

**API keys** — create a `.env` file in the project root:
```
OPENROUTER_API_KEY=sk-or-v1-...      # required — get at openrouter.ai
TAVILY_API_KEY=tvly-...              # optional — enables the news/policy layer (tavily.com)
OBSIDIAN_VAULT_PATH=/path/to/vault   # optional — where analysis notes are saved
```
SEC EDGAR insider data needs no key. If `TAVILY_API_KEY` or `OBSIDIAN_VAULT_PATH` is missing, those features skip silently and the rest of the pipeline runs unaffected.

## Configuration

All knobs live in `backend/config.py`: council/chairman models, shortlist size, token caps, cache TTL, and the cost controls for the catalyst and insider layers.

## Running

```bash
./start.sh
```
Then open http://localhost:5173.

## Tech stack

- **Backend:** FastAPI (Python 3.10+), async httpx, OpenRouter API, yfinance, pandas
- **Data/signals:** yfinance, SEC EDGAR (Form 4), Tavily search
- **Frontend:** React + Vite
- **Storage:** JSON conversations + Obsidian markdown export

## Limitations

Read this before trusting any output.

- **Markets are largely efficient.** Public information an LLM can access is already reflected in prices. Most professional funds fail to beat the S&P 500 over time; this tool has no structural edge they lack.
- **The models cannot predict the future.** Two-year price targets are formula outputs from stated assumptions, not forecasts. Garbage assumptions produce garbage targets.
- **Valuation-history stats are shallow.** yfinance provides only ~3 years of fundamentals, so valuation-percentile signals are indicative, not definitive.
- **Thematic and policy signals are the most hype-prone inputs.** By the time news reaches you, it is usually priced in.
- **This is not financial advice.** It is a structured research aid. Verify everything independently; do not move real money on its output alone.

## Credit

Original concept and base app by [Andrej Karpathy](https://github.com/karpathy/llm-council). This fork extends it into an equity-research pipeline.
