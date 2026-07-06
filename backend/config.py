"""Configuration for the LLM Council."""

import os
from dotenv import load_dotenv

load_dotenv()

# OpenRouter API key
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# ---------------------------------------------------------------------------
# Stage A — Screening (cheap, single model, no peer review)
# ---------------------------------------------------------------------------
# One inexpensive model does the top-down macro -> sector -> candidate pass and
# returns a shortlist. Premium models are deliberately NOT used here.
SCREENING_MODEL = "google/gemini-3.5-flash"

# How many tickers the screening stage should shortlist for the deep dive.
SHORTLIST_SIZE = 5

# ---------------------------------------------------------------------------
# Stage B — Deep dive (the full council, premium models allowed)
# ---------------------------------------------------------------------------
# Council members - list of OpenRouter model identifiers. These (potentially
# expensive) models only run in Stage B, on the shortlist, never on screening.
COUNCIL_MODELS = [
    "openai/gpt-5.5",
    "google/gemini-3.1-pro-preview",
    "anthropic/claude-fable-5",
    "x-ai/grok-4.3",
]

# Chairman model - synthesizes final response
CHAIRMAN_MODEL = "openai/gpt-5.5"

# ---------------------------------------------------------------------------
# Per-model output token limits (each model at its OWN maximum)
# ---------------------------------------------------------------------------
# NOTE ON REASONING MODELS: every model here (GPT-5.x, Gemini 3.x, Claude Fable
# 5, Grok 4.x) is reasoning-capable, and OpenRouter counts hidden reasoning
# tokens against `max_tokens`. If the cap is too low the model can burn the
# ENTIRE budget on reasoning and return EMPTY content with finish_reason="length"
# — which looks exactly like "the model returned nothing".
#
# For a maximum-quality run we DON'T share one ceiling across models. Each model's
# max_tokens is set to THAT model's own maximum supported output limit, so answers
# never truncate. `max_tokens` is only an UPPER BOUND — the model stops when its
# answer is complete — so raising the cap prevents truncation without adding cost
# on normal-length answers. We never remove max_tokens (some models default to a
# small value, or error, when it's omitted), and never set it above the provider's
# real limit (which would trigger an API error).
#
# Values are each provider's real max completion tokens as advertised by
# OpenRouter's model catalog (top_provider.max_completion_tokens) — which is also
# the ceiling OpenRouter itself enforces. Verified live against
# https://openrouter.ai/api/v1/models on 2026-07-07:
#   openai/gpt-5.5                 -> 128000  (OpenRouter cap 128000)
#   google/gemini-3.1-pro-preview  ->  65536  (OpenRouter cap 65536)
#   anthropic/claude-fable-5       -> 128000  (OpenRouter cap 128000)
#   x-ai/grok-4.3                  -> 128000  (OpenRouter advertises no completion
#                                     cap; xAI documents no output limit below the
#                                     1M context, so 128000 is a safe flagship-tier
#                                     value, well under every known provider cap)
#   google/gemini-3.5-flash        ->  65536  (OpenRouter cap 65536; screening)
MODEL_MAX_TOKENS = {
    "openai/gpt-5.5": 128000,
    "google/gemini-3.1-pro-preview": 65536,
    "anthropic/claude-fable-5": 128000,
    "x-ai/grok-4.3": 128000,
    "google/gemini-3.5-flash": 65536,
}

# Fallback for any model not listed above (e.g. if the council roster changes).
# Generous enough that structured reports don't truncate, yet within essentially
# every flagship provider's output ceiling.
DEFAULT_MAX_TOKENS = 32000


def max_tokens_for(model: str) -> int:
    """Return `model`'s OWN maximum supported output limit (see MODEL_MAX_TOKENS).

    Always returns an int, never None — callers should send an explicit
    max_tokens because some models default to a small value (or error) when it's
    omitted.
    """
    return MODEL_MAX_TOKENS.get(model, DEFAULT_MAX_TOKENS)


# Screening (Stage A) uses the screening model's own maximum too. It's a terse
# shortlist, so the model won't get near this — but the cap must clear its hidden
# reasoning tokens, or it can return an empty shortlist (finish_reason="length")
# -> no tickers -> the whole live-data flow silently degrades to training-data
# answers.
SCREENING_MAX_TOKENS = max_tokens_for(SCREENING_MODEL)

# ---------------------------------------------------------------------------
# Market data
# ---------------------------------------------------------------------------
# Seconds to cache a ticker's computed metrics so all council models (and both
# stages) share a single yfinance fetch instead of refetching per model.
METRICS_CACHE_TTL = 900  # 15 minutes

# ---------------------------------------------------------------------------
# Obsidian export
# ---------------------------------------------------------------------------
# Absolute path to the folder in your Obsidian vault where analysis notes are
# written. Defaults to the user's vault folder; override with the
# OBSIDIAN_VAULT_PATH env var (in .env). When empty, export is skipped silently
# so the rest of the flow is unaffected. The folder is created if missing.
OBSIDIAN_VAULT_PATH = os.getenv("OBSIDIAN_VAULT_PATH", "/Users/jaeson/Documents/Stock Research")

# ---------------------------------------------------------------------------
# Catalyst & thematic layer — Tavily (external news signals)
# ---------------------------------------------------------------------------
# Tavily search API key (from .env). When empty, the whole catalyst layer is
# skipped silently and the existing flow is unaffected (same fail-safe posture
# as the Obsidian export). Get a key at https://tavily.com.
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
TAVILY_API_URL = "https://api.tavily.com/search"

# Cost controls for news pulls (both credits AND prompt tokens). Every knob here
# is deliberately small so the layer can't blow up as the shortlist grows:
#   - basic depth  => 1 Tavily credit per search (advanced would be 2)
#   - 1 search per ticker, so a 5-ticker run = 5 credits for Stage 1
#   - at most CATALYST_MAX_RESULTS dated items injected per ticker
CATALYST_SEARCH_DEPTH = "basic"   # 'basic' (1 credit) or 'advanced' (2 credits)
CATALYST_MAX_RESULTS = 4          # dated items INJECTED per ticker (token cap)
CATALYST_POLICY_DAYS = 60         # recency window for Stage 1 policy/regulatory news
# Candidate pool fetched per search before Python relevance-ranks and keeps the
# top CATALYST_MAX_RESULTS. A bigger pool costs NOTHING extra (Tavily bills per
# search, not per result) but lets us rank real policy items above market noise.
CATALYST_CANDIDATE_POOL = 8

# Stage 3 — tech-trend / leadership context (LOW-CONFIDENCE, lowest priority).
# One extra Tavily search per ticker (so it ~doubles the per-run Tavily credits);
# set CATALYST_THEME_ENABLED = False to switch it off and keep only Stage 1.
CATALYST_THEME_ENABLED = True
CATALYST_THEME_DAYS = 90          # tech trends / exec changes move slower than policy

# ---------------------------------------------------------------------------
# Insider activity (SEC EDGAR Form 4) — Stage 2 of the catalyst layer
# ---------------------------------------------------------------------------
# SEC EDGAR is free and needs no key, but REQUIRES a descriptive User-Agent that
# identifies the caller with contact info, and rate-limits to 10 requests/sec.
SEC_USER_AGENT = "llm-council/1.0 (sonj01071@gmail.com)"
INSIDER_LOOKBACK_MONTHS = 6       # window for the net buy/sell computation (3-6)
INSIDER_MAX_FILINGS = 25          # Form 4s parsed per ticker (bounds latency + requests)
INSIDER_CLUSTER_MIN_BUYERS = 3    # distinct open-market buyers => "cluster buying"

# OpenRouter API endpoint
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

# Data directory for conversation storage
DATA_DIR = "data/conversations"
