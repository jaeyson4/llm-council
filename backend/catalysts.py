"""Catalyst & thematic layer — bounded external signals injected into the council.

STAGE 1 (implemented here): recent government policy / subsidy / regulatory news
per ticker, via the Tavily search API. The formatted block is appended to the
SAME market-data context that already feeds every council model (Stage 1), the
peer-review stage (Stage 2), and the chairman (Stage 3) — so all four models see
the same dated, bounded news. Wiring lives in `council.prepare_deepdive`.

Design (mirrors the Obsidian/metrics fail-safe philosophy):
- DISABLED-BY-DEFAULT & SAFE: with no TAVILY_API_KEY every function returns empty
  and the existing flow is completely unaffected.
- BOUNDED: one `basic` Tavily search per ticker (= 1 Tavily credit each), at most
  CATALYST_MAX_RESULTS dated items injected per ticker, snippets truncated — so
  neither Tavily credits nor prompt tokens grow unbounded with the shortlist.
- DATED: only the `news` topic is used, so each item carries a publish date, and
  every item is printed with that date.
- ANTI-HYPE GUARDRAIL: the injected block requires each model to classify every
  catalyst as a DURABLE STRUCTURAL DRIVER vs ALREADY PRICED-IN / NARRATIVE and to
  estimate the % already reflected in the price. The point is to resist
  hype-chasing, not amplify it.
- FAIL-SAFE: any network/parse error is logged and swallowed; news is a bonus
  signal, never a hard dependency.
"""

import asyncio
import re
import traceback
from datetime import datetime
from typing import Any, Dict, List, Optional

import httpx

from .config import (
    TAVILY_API_KEY,
    TAVILY_API_URL,
    CATALYST_MAX_RESULTS,
    CATALYST_CANDIDATE_POOL,
    CATALYST_POLICY_DAYS,
    CATALYST_SEARCH_DEPTH,
    CATALYST_THEME_ENABLED,
    CATALYST_THEME_DAYS,
)

# Snippet cap per item (chars) — bounds prompt-token cost regardless of source.
_SNIPPET_CHARS = 160
# Network timeout per search (seconds).
_TAVILY_TIMEOUT = 20.0

# --- Relevance ranking (policy signal over market noise) --------------------
# Tavily's `news` topic returns real policy items mixed with market chatter
# (price-target changes, "stock soars", earnings beats). We fetch a larger
# candidate pool (free — Tavily bills per search, not per result), score each by
# how policy-relevant vs market-noisy it reads, and keep only the top items. This
# demotes noise to the bottom so that when there are enough real policy items the
# noise is dropped entirely.
_POLICY_KEYWORDS = (
    "regulat", "policy", "subsid", "tariff", "legislat", "congress", "senate",
    "white house", "administration", "antitrust", "ftc", "doj", "fda", "epa",
    "medicare", "medicaid", "340b", "export control", "export restriction",
    "export", "sanction", "tax credit", "chips act", "executive order",
    "lawsuit", "probe", "investigation", "regulator", "commission", "ferc",
    "nrc", "department of", "federal", "government", "emissions", "mandate",
    "national security", "reform", "rate case", "utility commission",
    "clean energy", "china", "trade war", "geopolit", "quota",
)
_NOISE_KEYWORDS = (
    "price target", "raises price", "raised price", "upgrade", "downgrade",
    "buy rating", "sell rating", "analyst", "earnings beat", "tops estimates",
    "beats estimates", "stock soar", "stock jump", "rally", "wiped out",
    "selloff", "sell-off", "52-week", "gifts for", "best gifts",
    "record high", "new high", "all-time high", "closes at", "stock purchase",
)

# Stage 3 — tech-trend + leadership vocabulary. Leadership changes must be about
# THIS company (the on-topic bonus enforces that); tech-trend terms are broader.
_THEME_KEYWORDS = (
    "ceo", "chief executive", "cfo", "chief financial", "cto",
    "chief technology", "coo", "chairman", "resign", "step down", "steps down",
    "appoint", "succession", "successor", "departure", "named ceo", "new ceo",
    "interim", "leadership", "management change", "executive", "board",
    "breakthrough", "next-generation", "next-gen", "roadmap", "unveil",
    "launch", "innovation", "disrupt", "generative", "artificial intelligence",
    " ai ", "quantum", "architecture", "platform", "patent", "r&d",
)

# Leading scraped-page cruft to skip when cleaning snippets.
_BOILERPLATE_RE = re.compile(
    r"home page|power to investors|search for symbols|get a free trial|"
    r"enter your email|sign up|subscribe|search the site|best gifts|"
    r"featured stories|browse world|browse business|learn more about|"
    r"skip to (?:content|main)|cookie|newsletter|apr cards|balance transfer|"
    r"high-yield savings|money market account|privacy policy|terms of service|"
    r"recaptcha|by signing up",
    re.I,
)


# ---------------------------------------------------------------------------
# Date handling
# ---------------------------------------------------------------------------
def _normalize_date(raw: Optional[str]) -> str:
    """Best-effort normalize Tavily's published_date to YYYY-MM-DD; fall back to
    the raw string, then to 'undated'. Tavily's `news` topic returns dates in a
    few shapes across sources, so try several before giving up."""
    if not raw:
        return "undated"
    raw = str(raw).strip()
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d", "%a, %d %b %Y %H:%M:%S %Z",
                "%a, %d %b %Y %H:%M:%S GMT", "%d %b %Y", "%Y/%m/%d"):
        try:
            return datetime.strptime(raw, fmt).strftime("%Y-%m-%d")
        except (ValueError, TypeError):
            continue
    # ISO-ish with timezone suffix (e.g. 2026-06-01T12:00:00Z / +00:00): take the
    # date prefix if the first 10 chars already look like YYYY-MM-DD.
    if len(raw) >= 10 and raw[4] == "-" and raw[7] == "-":
        return raw[:10]
    return raw


def _is_dated(item: Dict[str, str]) -> bool:
    """True when the item's date parsed to a real YYYY-MM-DD (sortable)."""
    d = item.get("date", "")
    return len(d) >= 4 and d[:4].isdigit()


# ---------------------------------------------------------------------------
# Snippet cleaning + relevance scoring
# ---------------------------------------------------------------------------
def _clean_snippet(text: Optional[str]) -> str:
    """Strip scraped-page cruft from a Tavily content snippet: drop markdown
    heading markers, collapse whitespace, skip leading boilerplate/nav fragments
    (short bits like 'Search the Site' or 'Home page'), and truncate. Keeps the
    first substantive sentence onward so the model sees article text, not chrome."""
    if not text:
        return ""
    t = re.sub(r"#+", " ", str(text))
    t = re.sub(r"\s+", " ", t).strip()
    frags = t.split(". ")
    kept: List[str] = []
    started = False
    for frag in frags:
        frag = frag.strip()
        if not frag:
            continue
        if not started:
            # Skip nav/boilerplate lead-ins until the first real sentence.
            if len(frag.split()) < 4 or _BOILERPLATE_RE.search(frag):
                continue
            started = True
        kept.append(frag)
    cleaned = ". ".join(kept).strip() or t
    if len(cleaned) > _SNIPPET_CHARS:
        cleaned = cleaned[:_SNIPPET_CHARS].rstrip(" ,.;:-") + "…"
    return cleaned


# Sector vocabulary so genuine SECTOR policy (not just company-named items) is
# recognized as on-topic — and, conversely, so policy about a DIFFERENT sector is
# not. Keyed by yfinance's sector string (lowercased, first word match).
_SECTOR_SYNONYMS = {
    "utilities": ("utilit", "nuclear", "electric", "power grid", "grid", "ferc",
                  "energy", "reactor"),
    "energy": ("oil", "gas", "energy", "drilling", "pipeline", "opec", "crude"),
    "healthcare": ("drug", "pharma", "health", "medicare", "medicaid", "fda",
                   "biotech", "340b", "clinical"),
    "technology": ("chip", "semiconductor", "ai ", "software", "tech", "export"),
    "financial services": ("bank", "financ", "lending", "capital", "basel"),
    "industrials": ("manufactur", "industrial", "defense", "aerospace"),
    "consumer defensive": ("retail", "grocery", "consumer"),
    "consumer cyclical": ("retail", "consumer", "auto"),
    "communication services": ("telecom", "media", "broadband", "spectrum"),
    "basic materials": ("mining", "metals", "chemical", "materials"),
    "real estate": ("reit", "housing", "property", "mortgage"),
}

# The on-topic bonus must DOMINATE the raw policy-keyword score, so an off-topic
# policy story (high policy score, wrong company/sector) can't outrank a real
# on-topic one. Tuned above the largest realistic policy-keyword count.
_ONTOPIC_BONUS = 8


def _relevance_score(item: Dict[str, str], subject_tokens=(),
                     positive_keywords=_POLICY_KEYWORDS) -> int:
    """Score an item for THIS-name/THIS-sector relevance vs market noise: +2 per
    positive (topic) keyword, +_ONTOPIC_BONUS if it mentions this company/ticker
    or its sector's vocabulary (so cross-topic bleed — e.g. an AI export-control
    story under a nuclear utility, or a CEO change at another firm — is demoted),
    −1 per market-noise keyword. `positive_keywords` selects the topic (policy for
    Stage 1, tech-trend/leadership for Stage 3)."""
    text = f"{item.get('title', '')} {item.get('snippet', '')}".lower()
    positive = sum(1 for k in positive_keywords if k in text)
    noise = sum(1 for k in _NOISE_KEYWORDS if k in text)
    ontopic = _ONTOPIC_BONUS if any(tok and tok in text for tok in subject_tokens) else 0
    return positive * 2 + ontopic - noise


def _subject_tokens(ticker: str, name: str, sector: str = ""):
    """Lowercased match tokens marking an item as on-topic for this name: the
    ticker, the first meaningful word of the company name (NVDA -> 'nvidia'), and
    the sector's vocabulary (Utilities -> 'nuclear','electric','grid',...)."""
    toks = {(ticker or "").lower().strip()}
    first = re.split(r"\W+", (name or "").strip())[0].lower() if name else ""
    if len(first) >= 3:
        toks.add(first)
    sec_first = (sector or "").strip().lower()
    for key, syns in _SECTOR_SYNONYMS.items():
        if sec_first and (sec_first == key or sec_first.startswith(key.split()[0])):
            toks.update(syns)
            break
    return {t for t in toks if t}


# ---------------------------------------------------------------------------
# Tavily search (one bounded call)
# ---------------------------------------------------------------------------
async def _tavily_search(
    query: str,
    *,
    topic: str = "news",
    days: int = CATALYST_POLICY_DAYS,
    max_results: int = CATALYST_MAX_RESULTS,
) -> List[Dict[str, Any]]:
    """One Tavily search. Returns the raw `results` list (possibly empty). Never
    raises — logs and returns [] on any error or when no key is configured.

    Auth is sent BOTH as an `Authorization: Bearer` header and as an `api_key`
    body field so this works across Tavily API revisions without guessing."""
    if not TAVILY_API_KEY:
        return []
    payload = {
        "api_key": TAVILY_API_KEY,
        "query": query,
        "topic": topic,
        "search_depth": CATALYST_SEARCH_DEPTH,  # 'basic' = 1 credit/search
        "max_results": max_results,
        "include_answer": False,       # we don't need Tavily's synthesized answer
        "include_raw_content": False,  # snippets only — keeps tokens/credits down
        "include_images": False,
    }
    if topic == "news":
        payload["days"] = days
    try:
        async with httpx.AsyncClient(timeout=_TAVILY_TIMEOUT) as client:
            resp = await client.post(
                TAVILY_API_URL,
                json=payload,
                headers={"Authorization": f"Bearer {TAVILY_API_KEY}"},
            )
            resp.raise_for_status()
            data = resp.json()
        results = data.get("results") or []
        return results if isinstance(results, list) else []
    except Exception:
        # Never let a bad key / rate limit / network blip break the research flow.
        print(f"[catalysts] Tavily search FAILED for query {query!r}:")
        traceback.print_exc()
        return []


# ---------------------------------------------------------------------------
# Per-ticker query builders (Stage 1 policy, Stage 3 tech-trend/leadership)
# ---------------------------------------------------------------------------
def _policy_query(ticker: str, name: str, sector: str) -> str:
    """Build a sector-aware, POLICY-front-loaded query for one ticker. Leading
    with regulatory/legislative language (rather than the ticker) steers Tavily
    toward government action and away from pure stock-market chatter. Sector
    targets the right regime (semis, energy, healthcare, ...); it's omitted for
    ETFs/thin names where yfinance has no sector."""
    subject = (name or "").strip() or ticker
    sector = (sector or "").strip()
    lead = ("government policy, regulation, legislation, subsidies, tariffs, "
            "antitrust, or export controls affecting")
    return (f"{lead} the {sector} sector and {subject} ({ticker})"
            if sector else f"{lead} {subject} ({ticker})")


def _theme_query(ticker: str, name: str, sector: str) -> str:
    """Stage 3 query: leadership changes + major technology trends for this name.
    Leads with executive-change and tech-trend language so Tavily returns thematic
    context rather than price chatter; sector broadens the tech-trend reach."""
    subject = (name or "").strip() or ticker
    sector = (sector or "").strip()
    lead = ("CEO and executive leadership changes (appointments, resignations, "
            "succession) and major technology trends or product breakthroughs "
            "affecting")
    return (f"{lead} {subject} ({ticker}) and the {sector} sector"
            if sector else f"{lead} {subject} ({ticker})")


# ---------------------------------------------------------------------------
# Generic fetch + clean + relevance-rank (shared by every Tavily stage)
# ---------------------------------------------------------------------------
async def _fetch_ranked(ticker: str, name: str, sector: str, query: str,
                        positive_keywords, days: int) -> List[Dict[str, str]]:
    """Fetch a candidate pool for one ticker, clean the snippets, relevance-rank
    against `positive_keywords` (this-name/this-sector first, recency second), and
    return the top CATALYST_MAX_RESULTS as [{'title','url','date','snippet'}].
    Never raises."""
    results = await _tavily_search(query, days=days, max_results=CATALYST_CANDIDATE_POOL)
    items: List[Dict[str, str]] = []
    for r in results:
        if not isinstance(r, dict):
            continue
        items.append({
            "title": (r.get("title") or "").strip(),
            "url": (r.get("url") or "").strip(),
            "date": _normalize_date(r.get("published_date")),
            "snippet": _clean_snippet(r.get("content")),
        })
    subject = _subject_tokens(ticker, name, sector)
    items.sort(
        key=lambda x: (_relevance_score(x, subject, positive_keywords),
                       x["date"] if _is_dated(x) else "0000-00-00"),
        reverse=True,
    )
    return items[:CATALYST_MAX_RESULTS]


async def fetch_policy_news(ticker: str, name: str = "", sector: str = "") -> List[Dict[str, str]]:
    """Stage 1: top policy/regulatory news items for one ticker (policy-ranked)."""
    return await _fetch_ranked(ticker, name, sector, _policy_query(ticker, name, sector),
                               _POLICY_KEYWORDS, CATALYST_POLICY_DAYS)


async def fetch_theme_news(ticker: str, name: str = "", sector: str = "") -> List[Dict[str, str]]:
    """Stage 3: top tech-trend / leadership items for one ticker (LOW CONFIDENCE)."""
    return await _fetch_ranked(ticker, name, sector, _theme_query(ticker, name, sector),
                               _THEME_KEYWORDS, CATALYST_THEME_DAYS)


async def _fetch_many(
    shortlist: List[Dict[str, str]],
    metrics_by_ticker: Dict[str, Dict[str, Any]],
    fetch_fn,
    label: str,
) -> Dict[str, List[Dict[str, str]]]:
    """Run `fetch_fn(ticker, name, sector)` concurrently across the shortlist —
    one Tavily search per ticker (= N basic credits for N tickers). Returns
    {ticker: [items]} for tickers with results; empty dict if disabled or nothing
    came back. Never raises."""
    if not TAVILY_API_KEY or not shortlist:
        return {}
    tickers = [it["ticker"] for it in shortlist]
    names = {t: (metrics_by_ticker.get(t) or {}).get("name", "") for t in tickers}
    sectors = {t: (metrics_by_ticker.get(t) or {}).get("sector", "") for t in tickers}
    tasks = [fetch_fn(t, names.get(t, ""), sectors.get(t, "")) for t in tickers]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    out: Dict[str, List[Dict[str, str]]] = {}
    for t, res in zip(tickers, results):
        if isinstance(res, Exception):
            print(f"[catalysts] {label} task errored for {t}: {res!r}")
            continue
        if res:
            out[t] = res
    return out


async def fetch_policy_news_many(shortlist, metrics_by_ticker):
    """Concurrent Stage 1 policy-news fetch across the shortlist (N credits)."""
    return await _fetch_many(shortlist, metrics_by_ticker, fetch_policy_news, "policy news")


async def fetch_theme_news_many(shortlist, metrics_by_ticker):
    """Concurrent Stage 3 theme-news fetch across the shortlist (N credits)."""
    return await _fetch_many(shortlist, metrics_by_ticker, fetch_theme_news, "theme news")


# ---------------------------------------------------------------------------
# Formatting (labeled, dated, + anti-hype guardrail)
# ---------------------------------------------------------------------------
_GUARDRAIL = (
    "ANTI-HYPE GUARDRAIL — for EVERY news item above you MUST, in your analysis:\n"
    "  (1) classify it as either [DURABLE STRUCTURAL DRIVER] or "
    "[ALREADY PRICED-IN / NARRATIVE]; and\n"
    "  (2) estimate what % of this news is ALREADY reflected in the current price "
    "(0-100%).\n"
    "The goal is to resist hype-chasing, not amplify it: recent, widely-reported "
    "news is usually largely priced in. Do NOT raise a price target on the basis "
    "of a catalyst you judge already priced in."
)


def _format_news_block(news_by_ticker: Dict[str, List[Dict[str, str]]],
                       header: str, section_label: str) -> str:
    """Render fetched news into one labeled, dated, injectable block with the
    anti-hype guardrail appended. Shared by the Stage 1 (policy) and Stage 3
    (tech-trend/leadership) formatters. Returns '' when there's nothing to inject."""
    if not news_by_ticker:
        return ""
    blocks: List[str] = []
    for ticker, items in news_by_ticker.items():
        if not items:
            continue
        lines = [f"### {ticker} — {section_label}"]
        for it in items:
            lines.append(f"- [{it.get('date', 'undated')}] {it.get('title') or '(untitled)'}")
            if it.get("snippet"):
                lines.append(f"    {it['snippet']}")
        blocks.append("\n".join(lines))
    if not blocks:
        return ""
    return f"{header}\n\n" + "\n\n".join(blocks) + f"\n\n{_GUARDRAIL}"


def format_policy_news_for_prompt(news_by_ticker: Dict[str, List[Dict[str, str]]]) -> str:
    """Render Stage 1 policy news into its injectable block."""
    header = (
        "CATALYST & THEMATIC LAYER — POLICY / REGULATORY NEWS (via Tavily; recent, "
        f"last ~{CATALYST_POLICY_DAYS} days, bounded to {CATALYST_MAX_RESULTS} "
        "items/ticker). These are external, dated news signals for context — weigh "
        "them critically against the Python-computed fundamentals above."
    )
    return _format_news_block(news_by_ticker, header, "recent policy / regulatory news")


def format_theme_news_for_prompt(news_by_ticker: Dict[str, List[Dict[str, str]]]) -> str:
    """Render Stage 3 tech-trend/leadership news into its injectable block, clearly
    marked LOW CONFIDENCE so the council treats it as soft background only."""
    header = (
        "CATALYST & THEMATIC LAYER — TECH-TREND & LEADERSHIP CONTEXT (via Tavily; "
        f"⚠ LOW CONFIDENCE; last ~{CATALYST_THEME_DAYS} days, bounded to "
        f"{CATALYST_MAX_RESULTS} items/ticker). Soft, thematic background on major "
        "technology trends and CEO/executive changes — noisier and LESS "
        "decision-grade than the policy and fundamental data above. Treat as "
        "low-confidence context only; do NOT let it drive a thesis on its own."
    )
    return _format_news_block(news_by_ticker, header,
                              "tech-trend & leadership context (⚠ LOW CONFIDENCE)")


async def fetch_and_format_policy_news(
    shortlist: List[Dict[str, str]],
    metrics_by_ticker: Dict[str, Dict[str, Any]],
) -> str:
    """Stage 1 top-level: fetch + format policy news in one call. Returns ''
    when disabled (no key) or nothing came back, so callers can unconditionally
    append the result to the context block."""
    if not TAVILY_API_KEY:
        print("[catalysts] policy news skipped: no TAVILY_API_KEY configured")
        return ""
    if not shortlist:
        return ""
    news = await fetch_policy_news_many(shortlist, metrics_by_ticker)
    block = format_policy_news_for_prompt(news)
    n_items = sum(len(v) for v in news.values())
    print(f"[catalysts] policy news: {n_items} item(s) across {len(news)}/"
          f"{len(shortlist)} ticker(s) — {len(shortlist)} Tavily "
          f"{CATALYST_SEARCH_DEPTH} search(es) = {len(shortlist)} credit(s)")
    return block


async def fetch_and_format_theme_news(
    shortlist: List[Dict[str, str]],
    metrics_by_ticker: Dict[str, Dict[str, Any]],
) -> str:
    """Stage 3 top-level: fetch + format LOW-CONFIDENCE tech-trend/leadership news.
    Returns '' when disabled (flag off / no key / empty shortlist) or nothing came
    back, so callers can unconditionally append it."""
    if not CATALYST_THEME_ENABLED:
        print("[catalysts] theme news skipped: CATALYST_THEME_ENABLED is False")
        return ""
    if not TAVILY_API_KEY:
        print("[catalysts] theme news skipped: no TAVILY_API_KEY configured")
        return ""
    if not shortlist:
        return ""
    news = await fetch_theme_news_many(shortlist, metrics_by_ticker)
    block = format_theme_news_for_prompt(news)
    n_items = sum(len(v) for v in news.values())
    print(f"[catalysts] theme news (LOW-CONF): {n_items} item(s) across {len(news)}/"
          f"{len(shortlist)} ticker(s) — {len(shortlist)} Tavily "
          f"{CATALYST_SEARCH_DEPTH} search(es) = {len(shortlist)} credit(s)")
    return block
