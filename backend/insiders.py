"""Catalyst layer, STAGE 2 — insider activity from SEC EDGAR Form 4 filings.

For each ticker we pull recent Form 4 (insider transaction) filings straight from
SEC EDGAR (free, no API key), parse the ownership XML, and compute IN PYTHON the
net open-market insider buying vs selling over the lookback window, plus a
cluster-buying flag. The LLM does NO math — it receives finished, structured
numbers, appended to the same context block that feeds every council model.

What counts as signal:
- Only OPEN-MARKET transactions: code P (purchase) vs code S (sale). Grants/awards
  (A), option exercises (M), tax-withholding (F), gifts (G), etc. are EXCLUDED
  from the buy/sell sentiment (they aren't discretionary market votes) but their
  count is reported so nothing is hidden.
- Insider BUYING is treated as the stronger signal; selling is often liquidity or
  diversification. That caveat is stated in the injected block.

Bounded & polite (SEC allows 10 req/s and requires a descriptive User-Agent):
- One submissions call + at most INSIDER_MAX_FILINGS Form 4 fetches per ticker.
- Every request goes through a shared ~8 req/s throttle with SEC_USER_AGENT.

Fail-safe: any network/parse error yields no signal for that ticker (logged, not
raised) so the research flow is never broken by EDGAR being slow or a filing
being malformed.
"""

import asyncio
import re
import traceback
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import httpx

from .config import (
    SEC_USER_AGENT,
    INSIDER_LOOKBACK_MONTHS,
    INSIDER_MAX_FILINGS,
    INSIDER_CLUSTER_MIN_BUYERS,
)

_SEC_TIMEOUT = 20.0
_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
_ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik_int}/{acc}/{doc}"

# ticker -> zero-padded 10-digit CIK (loaded once from SEC's ticker map)
_CIK_CACHE: Dict[str, str] = {}
_CIK_LOADED = False
_CIK_LOCK = asyncio.Lock()


# ---------------------------------------------------------------------------
# Shared request throttle (~8 req/s, safely under SEC's 10/s ceiling)
# ---------------------------------------------------------------------------
class _Throttle:
    """Serializes request *starts* to >= min_interval apart across all coroutines,
    so concurrent per-ticker fetches never exceed SEC's rate limit as a group."""

    def __init__(self, min_interval: float = 0.12):
        self._min = min_interval
        self._lock = asyncio.Lock()
        self._next = 0.0

    async def wait(self) -> None:
        async with self._lock:
            loop = asyncio.get_event_loop()
            now = loop.time()
            if self._next > now:
                await asyncio.sleep(self._next - now)
                now = loop.time()
            self._next = now + self._min


_throttle = _Throttle()


async def _get(client: httpx.AsyncClient, url: str) -> Optional[httpx.Response]:
    """Throttled GET with the SEC User-Agent. Returns the response or None on
    error (logged). raise_for_status is applied so 4xx/5xx become None."""
    await _throttle.wait()
    try:
        r = await client.get(url)
        r.raise_for_status()
        return r
    except Exception:
        print(f"[insiders] GET failed: {url}")
        traceback.print_exc()
        return None


# ---------------------------------------------------------------------------
# ticker -> CIK
# ---------------------------------------------------------------------------
async def _load_cik_map(client: httpx.AsyncClient) -> None:
    global _CIK_LOADED
    if _CIK_LOADED:
        return
    async with _CIK_LOCK:
        if _CIK_LOADED:
            return
        r = await _get(client, _TICKERS_URL)
        if r is None:
            return
        try:
            data = r.json()
            for entry in data.values():
                t = str(entry.get("ticker", "")).upper().strip()
                cik = str(entry.get("cik_str", "")).strip()
                if t and cik.isdigit():
                    _CIK_CACHE[t] = cik.zfill(10)
            _CIK_LOADED = True
        except Exception:
            print("[insiders] failed to parse SEC ticker->CIK map:")
            traceback.print_exc()


# ---------------------------------------------------------------------------
# Form 4 discovery + parsing
# ---------------------------------------------------------------------------
def _num(x: Any) -> Optional[float]:
    try:
        v = float(str(x).replace(",", "").strip())
        return v if v == v else None  # drop NaN
    except (TypeError, ValueError):
        return None


# Names shaped like a fund/holding vehicle rather than a person — used (together
# with the reported relationship) to separate structural 10%+ owner flows from
# individual officer/director conviction trades.
_ENTITY_NAME_RE = re.compile(
    r"\b(l\.?p\.?|llc|l\.l\.c\.|inc\.?|corp|trust|fund|partners|capital|holdings|"
    r"group|ltd|management|ventures|associates|company|advisors)\b", re.I)


def _is_entity(t: Dict[str, Any]) -> bool:
    """True when a transaction is by a 10%+ owner ENTITY (fund/holding company),
    not an individual officer/director. A director/officer is always treated as an
    individual even if also a 10%+ owner; a pure '10%+' relationship, or a blank
    relationship with a fund-shaped name, is treated as a structural entity."""
    rel = (t.get("rel") or "").strip()
    if "Dir" in rel:
        return False
    if rel and rel != "10%+":
        return False  # has an officer title => individual
    if rel == "10%+":
        return True
    return bool(_ENTITY_NAME_RE.search(t.get("owner") or ""))


async def _recent_form4_filings(
    client: httpx.AsyncClient, cik: str, cutoff: str
) -> List[Dict[str, str]]:
    """From the submissions API, return recent Form 4 filings (form == '4', filed
    on/after `cutoff` YYYY-MM-DD) as [{'accession','doc','filed'}], newest first,
    capped at INSIDER_MAX_FILINGS. Excludes amendments ('4/A') to avoid double
    counting restated transactions."""
    r = await _get(client, _SUBMISSIONS_URL.format(cik=cik))
    if r is None:
        return []
    try:
        recent = (r.json().get("filings") or {}).get("recent") or {}
    except Exception:
        return []
    forms = recent.get("form") or []
    accs = recent.get("accessionNumber") or []
    docs = recent.get("primaryDocument") or []
    dates = recent.get("filingDate") or []
    out: List[Dict[str, str]] = []
    for i, form in enumerate(forms):
        if form != "4":
            continue
        filed = dates[i] if i < len(dates) else ""
        if filed and filed < cutoff:
            continue
        out.append({
            "accession": accs[i] if i < len(accs) else "",
            # primaryDocument may carry an 'xslF345X05/<file>.xml' rendering prefix;
            # the bare filename at the filing root is the raw ownership XML.
            "doc": (docs[i] if i < len(docs) else "").rsplit("/", 1)[-1],
            "filed": filed,
        })
        if len(out) >= INSIDER_MAX_FILINGS:
            break
    return out


def _parse_form4_xml(xml_text: str) -> List[Dict[str, Any]]:
    """Parse one Form 4 ownership XML into a list of non-derivative transactions:
    {owner, title, code, ad ('A'/'D'), shares, price, value, date}. Derivative
    transactions (options/RSAs) are ignored — we want common-stock buy/sell votes.
    Malformed input yields [] (never raises)."""
    try:
        root = ET.fromstring(xml_text)
    except Exception:
        return []

    # Reporting owner (usually one per Form 4); take the first name + relationship.
    owner = ""
    rel = ""
    ro = root.find(".//reportingOwner")
    if ro is not None:
        owner = (ro.findtext(".//rptOwnerName") or "").strip()
        r = ro.find(".//reportingOwnerRelationship")
        if r is not None:
            bits = []
            if (r.findtext("isDirector") or "").strip() in ("1", "true"):
                bits.append("Dir")
            if (r.findtext("isOfficer") or "").strip() in ("1", "true"):
                bits.append((r.findtext("officerTitle") or "Officer").strip() or "Officer")
            if (r.findtext("isTenPercentOwner") or "").strip() in ("1", "true"):
                bits.append("10%+")
            rel = ", ".join(bits)

    txns: List[Dict[str, Any]] = []
    for t in root.findall(".//nonDerivativeTransaction"):
        code = (t.findtext(".//transactionCoding/transactionCode") or "").strip().upper()
        ad = (t.findtext(".//transactionAcquiredDisposedCode/value") or "").strip().upper()
        shares = _num(t.findtext(".//transactionShares/value"))
        price = _num(t.findtext(".//transactionPricePerShare/value"))
        date = (t.findtext(".//transactionDate/value") or "").strip()
        title = (t.findtext(".//securityTitle/value") or "").strip()
        if not code or shares is None:
            continue
        value = shares * price if (price is not None and price > 0) else None
        txns.append({
            "owner": owner, "rel": rel, "title": title, "code": code, "ad": ad,
            "shares": shares, "price": price, "value": value, "date": date,
        })
    return txns


async def _fetch_and_parse_filing(
    client: httpx.AsyncClient, cik: str, filing: Dict[str, str]
) -> List[Dict[str, Any]]:
    acc_nodash = (filing.get("accession") or "").replace("-", "")
    doc = filing.get("doc") or ""
    if not acc_nodash or not doc:
        return []
    url = _ARCHIVE_URL.format(cik_int=int(cik), acc=acc_nodash, doc=doc)
    r = await _get(client, url)
    if r is None:
        return []
    return _parse_form4_xml(r.text)


# ---------------------------------------------------------------------------
# Python-computed summary (net buy/sell + cluster detection)
# ---------------------------------------------------------------------------
def _side(txs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Net open-market buy/sell figures for one group of transactions. Buys =
    code P & acquired (A); sells = code S & disposed (D)."""
    buys = [t for t in txs if t["code"] == "P" and t["ad"] == "A"]
    sells = [t for t in txs if t["code"] == "S" and t["ad"] == "D"]

    def val(x):
        return sum(t["value"] for t in x if t.get("value"))

    def sh(x):
        return sum(t["shares"] for t in x if t.get("shares"))

    def largest(x):
        priced = [t for t in x if t.get("value")]
        if not priced:
            return None
        t = max(priced, key=lambda z: z["value"])
        return {"owner": t["owner"], "rel": t.get("rel", ""), "value": t["value"],
                "shares": t["shares"], "date": t["date"]}

    bv, sv = val(buys), val(sells)
    return {
        "buy_value": bv, "sell_value": sv, "net_value": bv - sv,
        "buy_shares": sh(buys), "sell_shares": sh(sells),
        "buy_txns": len(buys), "sell_txns": len(sells),
        "distinct_buyers": sorted({t["owner"] for t in buys if t.get("owner")}),
        "distinct_sellers": sorted({t["owner"] for t in sells if t.get("owner")}),
        "largest_buy": largest(buys), "largest_sell": largest(sells),
        "has_activity": bool(buys or sells),
    }


def compute_insider_summary(
    transactions: List[Dict[str, Any]],
    filings_parsed: int,
    truncated: bool,
    window_desc: str,
) -> Dict[str, Any]:
    """Aggregate parsed transactions into the net open-market buy/sell signal,
    SEPARATING individual officers/directors (the conviction signal, promoted to
    the top level) from 10%+ owner entities (structural flows — reorg, secondary,
    distribution — kept under 'entity'). Non-open-market codes (grants, option
    exercises, tax) are counted as 'other' but excluded from sentiment. Cluster
    buying counts distinct OFFICER/DIRECTOR buyers only."""
    indiv_txns = [t for t in transactions if not _is_entity(t)]
    entity_txns = [t for t in transactions if _is_entity(t)]
    indiv = _side(indiv_txns)
    entity = _side(entity_txns)

    def is_ps(t):
        return (t["code"] == "P" and t["ad"] == "A") or (t["code"] == "S" and t["ad"] == "D")

    other = [t for t in transactions if not is_ps(t)]

    summary = dict(indiv)  # promote individual-insider figures to the top level
    summary.update({
        "cluster_buying": len(indiv["distinct_buyers"]) >= INSIDER_CLUSTER_MIN_BUYERS,
        "entity": entity,
        "other_txn_count": len(other),
        "filings_parsed": filings_parsed,
        "truncated": truncated,
        "window": window_desc,
        "any_activity": indiv["has_activity"] or entity["has_activity"],
    })
    return summary


# ---------------------------------------------------------------------------
# Public: per-ticker + many
# ---------------------------------------------------------------------------
async def fetch_insider_activity(
    client: httpx.AsyncClient, ticker: str, months: int = INSIDER_LOOKBACK_MONTHS
) -> Optional[Dict[str, Any]]:
    """Fetch + parse + summarize insider activity for one ticker. Returns the
    summary dict, or None if the ticker has no CIK / no Form 4s / all fetches
    failed."""
    ticker = (ticker or "").upper().strip()
    await _load_cik_map(client)
    cik = _CIK_CACHE.get(ticker)
    if not cik:
        return None
    cutoff = (datetime.now() - timedelta(days=int(months * 30.44))).strftime("%Y-%m-%d")
    filings = await _recent_form4_filings(client, cik, cutoff)
    if not filings:
        return None
    # Fetch the filings concurrently; the shared throttle keeps the aggregate
    # request rate under SEC's ceiling regardless of how many run at once.
    results = await asyncio.gather(
        *[_fetch_and_parse_filing(client, cik, f) for f in filings],
        return_exceptions=True,
    )
    transactions: List[Dict[str, Any]] = []
    parsed = 0
    for res in results:
        if isinstance(res, Exception) or res is None:
            continue
        parsed += 1
        # Only keep transactions dated within the window (a filing near the cutoff
        # can restate an older transaction).
        for t in res:
            if not t.get("date") or t["date"] >= cutoff:
                transactions.append(t)
    truncated = len(filings) >= INSIDER_MAX_FILINGS
    window_desc = f"last ~{months} months (since {cutoff})"
    return compute_insider_summary(transactions, parsed, truncated, window_desc)


async def fetch_insider_activity_many(
    tickers: List[str], months: int = INSIDER_LOOKBACK_MONTHS
) -> Dict[str, Dict[str, Any]]:
    """Concurrent insider-activity fetch for several tickers. Returns
    {ticker: summary} for those with any parseable Form 4 data. Never raises."""
    uniq: List[str] = []
    for t in tickers:
        u = (t or "").upper().strip()
        if u and u not in uniq:
            uniq.append(u)
    if not uniq:
        return {}
    headers = {"User-Agent": SEC_USER_AGENT, "Accept-Encoding": "gzip, deflate"}
    out: Dict[str, Dict[str, Any]] = {}
    try:
        async with httpx.AsyncClient(timeout=_SEC_TIMEOUT, headers=headers) as client:
            results = await asyncio.gather(
                *[fetch_insider_activity(client, t, months) for t in uniq],
                return_exceptions=True,
            )
    except Exception:
        print("[insiders] insider activity batch failed:")
        traceback.print_exc()
        return {}
    for t, res in zip(uniq, results):
        if isinstance(res, Exception):
            print(f"[insiders] {t} errored: {res!r}")
            continue
        if res is not None:
            out[t] = res
    return out


# ---------------------------------------------------------------------------
# Formatting for prompt injection
# ---------------------------------------------------------------------------
def _money(x: Optional[float]) -> str:
    x = _num(x)
    if x is None:
        return "$0"
    sign = "-" if x < 0 else ""
    x = abs(x)
    for div, unit in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if x >= div:
            return f"{sign}${x / div:,.1f}{unit}"
    return f"{sign}${x:,.0f}"


def _summary_lines(ticker: str, s: Dict[str, Any]) -> List[str]:
    lines = [f"### {ticker} — insider activity ({s['filings_parsed']} Form 4"
             f"{'s' if s['filings_parsed'] != 1 else ''} parsed, {s['window']})"]
    if not s["any_activity"]:
        lines.append("- No open-market insider or 10%+ owner transactions in the "
                     f"window (grants/exercises/other: {s['other_txn_count']}).")
        return lines

    # Primary signal: individual officers/directors.
    if s["has_activity"]:
        net = s["net_value"]
        direction = "net BUYING" if net > 0 else "net SELLING" if net < 0 else "flat"
        lines.append(
            f"- Officers/directors — BUYS {_money(s['buy_value'])} "
            f"({s['buy_txns']} txn, {len(s['distinct_buyers'])} insider) vs "
            f"SELLS {_money(s['sell_value'])} ({s['sell_txns']} txn, "
            f"{len(s['distinct_sellers'])} insider) => NET {_money(net)} ({direction})")
        flag = "YES" if s["cluster_buying"] else "no"
        lines.append(f"- Cluster buying (>= {INSIDER_CLUSTER_MIN_BUYERS} distinct "
                     f"officer/director buyers): {flag}"
                     + (f" — {', '.join(s['distinct_buyers'])}" if s["cluster_buying"] else ""))
        lb, ls = s.get("largest_buy"), s.get("largest_sell")
        if lb:
            lines.append(f"- Largest insider buy: {_money(lb['value'])} by {lb['owner']}"
                         + (f" ({lb['rel']})" if lb.get("rel") else "") + f" on {lb['date']}")
        if ls:
            lines.append(f"- Largest insider sell: {_money(ls['value'])} by {ls['owner']}"
                         + (f" ({ls['rel']})" if ls.get("rel") else "") + f" on {ls['date']}")
    else:
        lines.append("- Officers/directors: no open-market buys or sells in the window.")

    # Secondary: 10%+ owner / entity flows (structural, weak conviction signal).
    e = s.get("entity") or {}
    if e.get("has_activity"):
        enet = e["net_value"]
        edir = "net buying" if enet > 0 else "net selling" if enet < 0 else "flat"
        lines.append(
            f"- 10%+ owner / entity flows (STRUCTURAL — often reorg / secondary / "
            f"distribution, NOT a conviction signal): BUYS {_money(e['buy_value'])} vs "
            f"SELLS {_money(e['sell_value'])} => NET {_money(enet)} ({edir})")
        el = e.get("largest_sell") or e.get("largest_buy")
        if el:
            lines.append(f"    largest: {_money(el['value'])} by {el['owner']} on {el['date']}")

    if s["other_txn_count"]:
        lines.append(f"- (Excluded from sentiment: {s['other_txn_count']} grant/"
                     "exercise/tax/gift txn(s).)")
    if s["truncated"]:
        lines.append(f"- NOTE: capped at {INSIDER_MAX_FILINGS} most-recent Form 4s "
                     "— older filings in the window were not parsed.")
    return lines


def format_insider_activity_for_prompt(summaries: Dict[str, Dict[str, Any]]) -> str:
    """Render Python-computed insider summaries into one labeled block. Returns ''
    when there's nothing to inject."""
    if not summaries:
        return ""
    blocks = ["\n".join(_summary_lines(t, s)) for t, s in summaries.items()]
    header = (
        "INSIDER ACTIVITY — SEC EDGAR FORM 4 (Python-computed). Open-market "
        "Purchases (code P) vs Sales (code S) only; grants, option exercises, and "
        "tax-withholding are excluded. Individual OFFICER/DIRECTOR trades (the "
        "conviction signal) are reported separately from 10%+ OWNER ENTITY flows "
        "(structural — reorg/secondary/distribution). Interpretation: insider "
        "BUYING — especially CLUSTER buying by several officers/directors — is a "
        "meaningful signal; insider SELLING is weaker (often liquidity or "
        "diversification), and entity flows weaker still — do not over-read them."
    )
    return f"{header}\n\n" + "\n\n".join(blocks)


async def fetch_and_format_insider_activity(tickers: List[str]) -> str:
    """Top-level entry for the deep-dive prep: fetch + format in one call. Returns
    '' when nothing came back, so callers can unconditionally append it."""
    if not tickers:
        return ""
    summaries = await fetch_insider_activity_many(tickers)
    block = format_insider_activity_for_prompt(summaries)
    n_active = sum(1 for s in summaries.values() if s.get("has_activity"))
    print(f"[insiders] Form 4 signal for {len(summaries)}/{len(tickers)} ticker(s) "
          f"({n_active} with open-market activity)")
    return block
