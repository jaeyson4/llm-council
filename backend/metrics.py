"""Deterministic stock metrics computed in Python (never by an LLM).

Given a ticker, this module pulls price + fundamentals + price history from
yfinance ONCE (cached), then CALCULATES valuation ratios, growth, margins, free
cash flow, simple 2-year scenario price targets, and historical context
(valuation percentile vs the stock's own history, max drawdown, and 2-year
forward base rates by valuation bucket).

Design principles:
- Every number here is computed in Python from raw statement/price data. The LLM
  receives these figures and interprets them; it never invents them.
- Fail safe. Any missing field yields None for that metric, never an exception,
  so the research flow continues even with partial data.
- One network fetch per ticker, cached with a TTL, so all council models and
  both stages share a single fetch instead of refetching per model.
- Historical valuation features are bounded by what yfinance exposes (~4 years
  of annual fundamentals). The lookback window and sample size are returned
  alongside every historical stat so neither the model nor the user overreads
  a thin sample.
"""

import asyncio
import logging
import math
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

try:
    import yfinance as yf
    import pandas as pd
    logging.getLogger("yfinance").setLevel(logging.CRITICAL)
except Exception:  # pragma: no cover - import guard so the app still boots
    yf = None
    pd = None

from .config import METRICS_CACHE_TTL

# Minimum number of forward-return observations before a base-rate stat is
# considered meaningful enough to report (below this we say "insufficient").
_MIN_BASE_RATE_OBS = 40

# Assumed lag between a fiscal-period end and when its annual EPS is actually
# filed/public. Used to avoid lookahead bias when building the P/E history.
_EPS_REPORTING_LAG_DAYS = 75

# ---------------------------------------------------------------------------
# TTL cache: {ticker -> (expires_at_epoch, metrics_dict)}
# ---------------------------------------------------------------------------
_cache: Dict[str, tuple] = {}


def _cache_get(ticker: str) -> Optional[Dict[str, Any]]:
    entry = _cache.get(ticker)
    if entry and entry[0] > time.time():
        return entry[1]
    return None


def _cache_put(ticker: str, value: Dict[str, Any]) -> None:
    _cache[ticker] = (time.time() + METRICS_CACHE_TTL, value)


# ---------------------------------------------------------------------------
# Small numeric helpers
# ---------------------------------------------------------------------------
def _num(value: Any) -> Optional[float]:
    """Coerce to float, treating NaN/None/garbage as missing (None)."""
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    return f


def _safe_div(a: Optional[float], b: Optional[float]) -> Optional[float]:
    a, b = _num(a), _num(b)
    if a is None or b is None or b == 0:
        return None
    return a / b


def _row_series(statement, *candidate_rows):
    """Return the first matching row of a yfinance statement as a float Series
    sorted oldest->newest, or None. Tries several row-label spellings."""
    if pd is None or statement is None:
        return None
    try:
        if getattr(statement, "empty", True):
            return None
    except Exception:
        return None
    for label in candidate_rows:
        if label in statement.index:
            try:
                s = statement.loc[label]
                # A duplicated index label can return a DataFrame; take row 0.
                if isinstance(s, pd.DataFrame):
                    s = s.iloc[0]
                s = pd.to_numeric(s, errors="coerce").dropna()
                s = s.sort_index()  # columns are timestamps; ascending by date
                if len(s):
                    return s
            except Exception:
                continue
    return None


def _latest_aligned(numerator, denominator):
    """Return (num_value, denom_value) at the latest fiscal date present in BOTH
    series, or (None, None). This prevents pairing e.g. the latest year's
    Operating Cash Flow with a prior year's CapEx when the newest column is
    populated for one row but NaN for the other."""
    if numerator is None or denominator is None:
        return None, None
    try:
        common = numerator.index.intersection(denominator.index)
        if len(common) == 0:
            return None, None
        d = common.max()
        return _num(numerator.loc[d]), _num(denominator.loc[d])
    except Exception:
        return None, None


def _cagr(series, target_years: int = 3) -> Optional[Dict[str, Any]]:
    """Compute compound annual growth rate over ~target_years from an
    oldest->newest Series. Returns {'value', 'years'} or None. Requires positive
    endpoints (CAGR is undefined across a sign change)."""
    if series is None or len(series) < 2:
        return None
    latest = _num(series.iloc[-1])
    if len(series) > target_years:
        earliest = _num(series.iloc[-(target_years + 1)])
        years = target_years
    else:
        earliest = _num(series.iloc[0])
        years = len(series) - 1
    if latest is None or earliest is None or latest <= 0 or earliest <= 0 or years < 1:
        return None
    return {"value": (latest / earliest) ** (1.0 / years) - 1.0, "years": years}


# ---------------------------------------------------------------------------
# Raw fetch (blocking) — one round of yfinance calls per ticker
# ---------------------------------------------------------------------------
def _raw_fetch(ticker: str) -> Optional[Dict[str, Any]]:
    if yf is None:
        return None
    try:
        t = yf.Ticker(ticker)
    except Exception:
        return None

    raw: Dict[str, Any] = {}
    try:
        raw["info"] = t.info or {}
    except Exception:
        raw["info"] = {}
    for key, attr in (("income", "income_stmt"), ("cashflow", "cashflow")):
        try:
            raw[key] = getattr(t, attr)
        except Exception:
            raw[key] = None
    try:
        raw["history"] = t.history(period="max", auto_adjust=True)
    except Exception:
        raw["history"] = None
    return raw


# ---------------------------------------------------------------------------
# Historical context (valuation percentile, drawdown, forward base rates)
# ---------------------------------------------------------------------------
def _historical_context(info, income, history, current_pe: Optional[float]) -> Dict[str, Any]:
    """All three historical features share one daily P/E series built from
    forward-filled annual EPS × daily close. Everything is bounded by the ~4y of
    annual EPS yfinance provides; window and n are always reported."""
    out: Dict[str, Any] = {
        "max_drawdown": None,
        "valuation_percentile": None,
        "forward_base_rate": None,
        "notes": [],
    }
    if pd is None or history is None:
        out["notes"].append("no price history available")
        return out
    try:
        if history.empty or "Close" not in history.columns:
            out["notes"].append("no price history available")
            return out
        close = pd.to_numeric(history["Close"], errors="coerce").dropna()
        if close.empty:
            return out
        # Drop timezone so we can compare against naive fiscal-period dates.
        close.index = pd.to_datetime(close.index).tz_localize(None)
    except Exception:
        return out

    # --- Max drawdown over full available history (robust) ---
    try:
        running_max = close.cummax()
        dd = (close / running_max - 1.0).min()
        out["max_drawdown"] = _num(dd)
    except Exception:
        pass

    # --- Build a daily trailing-P/E series from annual EPS (forward-filled) ---
    # yfinance keys annual EPS on the fiscal-period-END date, but that EPS is not
    # public until the 10-K is filed ~1-3 months later. We shift each EPS date
    # forward by a reporting lag so a given day only ever uses EPS that had
    # actually been reported by then (avoids lookahead bias).
    eps_series = _row_series(income, "Diluted EPS", "Basic EPS")
    pe_series = None
    if eps_series is not None and len(eps_series) >= 2:
        try:
            eps_df = pd.DataFrame(
                {"eps": eps_series.values},
                index=pd.to_datetime(eps_series.index).tz_localize(None)
                + pd.Timedelta(days=_EPS_REPORTING_LAG_DAYS),
            ).sort_index()
            eps_df = eps_df[eps_df["eps"] > 0]  # P/E undefined for negative EPS
            if len(eps_df) >= 2:
                # Restrict prices to the fundamental window, then as-of merge each
                # trading day to the most recent already-reported fiscal-year EPS.
                px = close[close.index >= eps_df.index.min()].to_frame("close")
                merged = pd.merge_asof(
                    px.sort_index(),
                    eps_df,
                    left_index=True,
                    right_index=True,
                    direction="backward",
                )
                merged = merged.dropna()
                merged = merged[merged["eps"] > 0]
                if len(merged) >= 60:
                    pe_series = merged["close"] / merged["eps"]
        except Exception:
            pe_series = None

    if pe_series is None or pe_series.empty:
        out["notes"].append(
            "valuation-history features unavailable (insufficient annual EPS data)"
        )
        return out

    window = {
        "start": str(pe_series.index.min().date()),
        "end": str(pe_series.index.max().date()),
        "years": round((pe_series.index.max() - pe_series.index.min()).days / 365.25, 1),
        "observations": int(len(pe_series)),
    }

    # Rank the CURRENT P/E on the SAME basis as the historical series (price ÷
    # most-recent annual EPS), not the TTM P/E — otherwise we'd compare a TTM
    # number against an annual-EPS distribution, which mis-states the percentile
    # and can mis-select the base-rate bucket. This is the last point of the
    # series: today's price over the currently-in-effect annual EPS.
    ref_pe = _num(pe_series.iloc[-1])

    # --- Valuation percentile: where does today's P/E sit in its own history? ---
    if ref_pe is not None:
        try:
            pct = float((pe_series <= ref_pe).mean() * 100.0)
            out["valuation_percentile"] = {
                "metric": "P/E (annual-EPS basis)",
                "current": round(ref_pe, 2),
                "percentile": round(pct, 1),
                "median": round(float(pe_series.median()), 2),
                "window": window,
            }
        except Exception:
            pass

    # --- 2y forward base rates conditioned on the current valuation bucket ---
    try:
        # Forward 2y return per start day: pair each day's price with the price
        # ~730 days later. We time-interpolate onto weekend/holiday target dates,
        # but MASK any target beyond the last real close so unripe start dates
        # (whose 2y-forward hasn't happened yet) are dropped, not flat-carried to
        # today's price — which would fabricate returns for the recent ~2 years.
        last_date = close.index.max()
        base = close.to_frame("p0").copy()
        target_dates = base.index + pd.Timedelta(days=730)
        p2 = close.reindex(
            close.index.union(target_dates)
        ).interpolate(method="time").reindex(target_dates)
        p2vals = p2.values.astype(float)
        beyond = (target_dates > last_date)
        beyond = beyond.values if hasattr(beyond, "values") else beyond
        p2vals[beyond] = float("nan")
        base["p2"] = p2vals
        base["fwd_2y"] = base["p2"] / base["p0"] - 1.0
        base = base.dropna(subset=["fwd_2y"])
        # Align P/E bucket edges (terciles) on the P/E series.
        q33, q67 = pe_series.quantile(0.33), pe_series.quantile(0.67)

        def bucket_of(pe_val):
            if pe_val <= q33:
                return "cheap"
            if pe_val >= q67:
                return "expensive"
            return "mid"

        cur_bucket = bucket_of(ref_pe) if ref_pe is not None else None
        # Attach each start day's P/E bucket, then keep only current-bucket days.
        pe_on_base = pe_series.reindex(base.index, method="ffill")
        base = base.assign(pe=pe_on_base).dropna(subset=["pe"])
        base["bucket"] = base["pe"].apply(bucket_of)
        if cur_bucket is not None:
            sub = base[base["bucket"] == cur_bucket]["fwd_2y"]
            n = int(len(sub))
            if n >= _MIN_BASE_RATE_OBS:
                out["forward_base_rate"] = {
                    "current_bucket": cur_bucket,
                    "n": n,
                    "median_return": round(float(sub.median()), 4),
                    "mean_return": round(float(sub.mean()), 4),
                    "pct_positive": round(float((sub > 0).mean() * 100.0), 1),
                    "window": window,
                    "caveat": "overlapping daily windows; small, autocorrelated sample",
                }
            else:
                out["notes"].append(
                    f"2y forward base rate for '{cur_bucket}' bucket has only "
                    f"n={n} obs (<{_MIN_BASE_RATE_OBS}); omitted as unreliable"
                )
    except Exception:
        out["notes"].append("forward base-rate computation failed")

    return out


# ---------------------------------------------------------------------------
# Price targets (2-year, bull/base/bear) — transparent scenario model
# ---------------------------------------------------------------------------
def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _price_targets(
    price: Optional[float],
    eps_ttm: Optional[float],
    base_pe: Optional[float],
    rev_ps: Optional[float],
    base_ps: Optional[float],
    growth: Optional[float],
) -> Optional[Dict[str, Any]]:
    """Project 2 years out under three scenarios. Prefers an EPS × exit-P/E model;
    falls back to a revenue-per-share × exit-P/S model for unprofitable names.
    Every scenario returns its explicit growth + exit-multiple assumptions so the
    council can attack them."""
    price = _num(price)
    # Base earnings/revenue growth: historical growth, sanity-clamped. If growth
    # is unknown, assume a modest 8%.
    g = _num(growth)
    g_base = _clamp(g if g is not None else 0.08, -0.05, 0.25)
    scenarios = {
        "bull": {"growth": _clamp(g_base + 0.05, -0.05, 0.35), "mult_factor": 1.20},
        "base": {"growth": g_base, "mult_factor": 1.00},
        "bear": {"growth": _clamp(g_base - 0.10, -0.20, 0.15), "mult_factor": 0.70},
    }

    eps_ttm, base_pe = _num(eps_ttm), _num(base_pe)
    rev_ps, base_ps = _num(rev_ps), _num(base_ps)

    method = None
    if eps_ttm is not None and eps_ttm > 0 and base_pe is not None and base_pe > 0:
        method = "eps_pe"
        anchor_mult = _clamp(base_pe, 5.0, 45.0)
    elif rev_ps is not None and rev_ps > 0 and base_ps is not None and base_ps > 0:
        method = "ps"
        anchor_mult = _clamp(base_ps, 0.5, 25.0)
    else:
        return None  # not enough to model a defensible target

    targets: Dict[str, Any] = {"method": method, "assumptions": {}}
    for name, sc in scenarios.items():
        exit_mult = anchor_mult * sc["mult_factor"]
        if method == "eps_pe":
            future = eps_ttm * (1 + sc["growth"]) ** 2
            exit_mult = _clamp(exit_mult, 4.0, 55.0)
            assump = {"eps_growth": round(sc["growth"], 3), "exit_pe": round(exit_mult, 1)}
        else:
            future = rev_ps * (1 + sc["growth"]) ** 2
            exit_mult = _clamp(exit_mult, 0.3, 30.0)
            assump = {"revenue_growth": round(sc["growth"], 3), "exit_ps": round(exit_mult, 1)}
        target_price = future * exit_mult
        entry = {"target": round(target_price, 2)}
        if price and price > 0:
            entry["implied_return"] = round(target_price / price - 1.0, 3)
            entry["implied_cagr"] = round((target_price / price) ** 0.5 - 1.0, 3)
        targets[name] = entry
        targets["assumptions"][name] = assump
    return targets


# ---------------------------------------------------------------------------
# Public: compute the full metrics bundle for a ticker (cached)
# ---------------------------------------------------------------------------
def get_stock_metrics(ticker: str) -> Optional[Dict[str, Any]]:
    """Compute (or return cached) the full metrics bundle for one ticker.
    Returns None only if the ticker is invalid / no data at all. Blocking:
    call via get_stock_metrics_async() from async code."""
    ticker = (ticker or "").strip().upper()
    if not ticker:
        return None
    cached = _cache_get(ticker)
    if cached is not None:
        return cached

    raw = _raw_fetch(ticker)
    if raw is None:
        return None
    info = raw.get("info") or {}
    income = raw.get("income")
    cashflow = raw.get("cashflow")
    history = raw.get("history")

    # --- Core primitives (raw, from info) ---
    price = _num(info.get("currentPrice"))
    prev_close = _num(info.get("previousClose"))
    name = info.get("shortName") or info.get("longName")

    # If we truly have nothing, treat as an invalid ticker.
    if price is None and (history is None or getattr(history, "empty", True)) and not name:
        return None
    if price is None and history is not None and not getattr(history, "empty", True):
        try:
            price = _num(pd.to_numeric(history["Close"], errors="coerce").dropna().iloc[-1])
        except Exception:
            price = None

    market_cap = _num(info.get("marketCap"))
    revenue_ttm = _num(info.get("totalRevenue"))
    ebitda = _num(info.get("ebitda"))
    total_debt = _num(info.get("totalDebt"))
    total_cash = _num(info.get("totalCash"))
    shares = _num(info.get("sharesOutstanding"))
    eps_ttm = _num(info.get("trailingEps"))
    ocf = _num(info.get("operatingCashflow"))

    # --- Valuation (CALCULATED, not read from info's ready-made ratios) ---
    pe = _safe_div(price, eps_ttm) if (eps_ttm and eps_ttm > 0) else None
    ps = _safe_div(market_cap, revenue_ttm)
    ev = None
    if market_cap is not None:
        ev = market_cap + (total_debt or 0.0) - (total_cash or 0.0)
    ev_ebitda = _safe_div(ev, ebitda) if (ebitda and ebitda > 0) else None

    # --- Growth (from annual statements) ---
    rev_series = _row_series(income, "Total Revenue", "Operating Revenue")
    eps_ann_series = _row_series(income, "Diluted EPS", "Basic EPS")
    rev_cagr = _cagr(rev_series, 3)
    eps_cagr = _cagr(eps_ann_series, 3)

    # --- Margins (latest annual: prefer statements, fall back to info) ---
    # Numerator and denominator are aligned to the latest fiscal date present in
    # BOTH rows, so a partial-latest column can't pair mismatched years.
    gross_margin = None
    op_margin = None
    if rev_series is not None:
        gp = _row_series(income, "Gross Profit")
        oi = _row_series(income, "Operating Income", "Total Operating Income As Reported")
        gp_v, rev_v = _latest_aligned(gp, rev_series)
        gross_margin = _safe_div(gp_v, rev_v)
        oi_v, rev_v2 = _latest_aligned(oi, rev_series)
        op_margin = _safe_div(oi_v, rev_v2)
    if gross_margin is None:
        gross_margin = _num(info.get("grossMargins"))
    if op_margin is None:
        op_margin = _num(info.get("operatingMargins"))

    # --- Free cash flow: Operating Cash Flow - CapEx (CALCULATED) ---
    # OCF and CapEx are taken from the same fiscal year (latest shared date).
    fcf = None
    ocf_row = _row_series(cashflow, "Operating Cash Flow", "Cash Flow From Continuing Operating Activities")
    capex_row = _row_series(cashflow, "Capital Expenditure")
    ocf_v, capex_v = _latest_aligned(ocf_row, capex_row)
    if ocf_v is not None and capex_v is not None:
        # CapEx is stored as a negative number, so OCF + CapEx == OCF - |CapEx|.
        fcf = ocf_v + capex_v
    if fcf is None:
        fcf = _num(info.get("freeCashflow"))
    if fcf is None and ocf is not None:
        fcf = ocf  # last resort: OCF as a coarse proxy
    fcf_margin = _safe_div(fcf, revenue_ttm)

    # --- Price targets ---
    rev_ps = _safe_div(revenue_ttm, shares)
    growth_for_targets = None
    if eps_cagr is not None:
        growth_for_targets = eps_cagr["value"]
    elif rev_cagr is not None:
        growth_for_targets = rev_cagr["value"]
    price_targets = _price_targets(
        price=price, eps_ttm=eps_ttm, base_pe=pe,
        rev_ps=rev_ps, base_ps=ps, growth=growth_for_targets,
    )

    # --- Historical context ---
    history_ctx = _historical_context(info, income, history, current_pe=pe)

    metrics = {
        "ticker": ticker,
        "name": name,
        "currency": info.get("currency"),
        "sector": info.get("sector"),
        "as_of": datetime.now().strftime("%Y-%m-%d"),
        "price": {"current": price, "previous_close": prev_close,
                  "change_pct": (round((price / prev_close - 1) * 100, 2)
                                 if price and prev_close else None)},
        "valuation": {"pe": pe, "ps": ps, "ev_ebitda": ev_ebitda, "ev": ev},
        "growth": {"revenue_cagr_3y": rev_cagr, "eps_cagr_3y": eps_cagr},
        "margins": {"gross": gross_margin, "operating": op_margin},
        "cash": {"fcf": fcf, "fcf_margin": fcf_margin},
        "fundamentals": {"market_cap": market_cap, "revenue_ttm": revenue_ttm,
                         "ebitda": ebitda, "total_debt": total_debt,
                         "total_cash": total_cash, "shares": shares, "eps_ttm": eps_ttm},
        "price_targets": price_targets,
        "history": history_ctx,
    }
    _cache_put(ticker, metrics)
    return metrics


async def get_stock_metrics_async(ticker: str) -> Optional[Dict[str, Any]]:
    """Async wrapper: runs the blocking fetch/compute in a worker thread."""
    return await asyncio.to_thread(get_stock_metrics, ticker)


async def get_many_metrics(tickers: List[str]) -> Dict[str, Dict[str, Any]]:
    """Fetch metrics for several tickers concurrently. Cached per ticker, so a
    ticker already computed this window costs nothing. Returns {ticker: metrics}
    for the ones that resolved."""
    uniq: List[str] = []
    for t in tickers:
        u = (t or "").strip().upper()
        if u and u not in uniq:
            uniq.append(u)
    if not uniq:
        return {}
    results = await asyncio.gather(*[get_stock_metrics_async(t) for t in uniq])
    return {t: m for t, m in zip(uniq, results) if m is not None}


# ---------------------------------------------------------------------------
# Formatting for prompt injection
# ---------------------------------------------------------------------------
def _pct(x: Optional[float], nd: int = 1) -> str:
    x = _num(x)
    return f"{x * 100:.{nd}f}%" if x is not None else "n/a"


def _money(x: Optional[float]) -> str:
    x = _num(x)
    if x is None:
        return "n/a"
    for div, unit in ((1e12, "T"), (1e9, "B"), (1e6, "M")):
        if abs(x) >= div:
            return f"${x / div:,.2f}{unit}"
    return f"${x:,.0f}"


def _ratio(x: Optional[float], nd: int = 2) -> str:
    x = _num(x)
    return f"{x:.{nd}f}" if x is not None else "n/a"


def format_metrics_for_prompt(m: Dict[str, Any]) -> str:
    """Render one ticker's computed metrics as a labeled block for injection into
    a model prompt. All figures are Python-computed."""
    if not m:
        return ""
    lines: List[str] = []
    head = m["ticker"] + (f" ({m['name']})" if m.get("name") else "")
    lines.append(f"### {head} — computed figures (as of {m.get('as_of')})")

    p = m.get("price", {})
    price = p.get("current")
    chg = p.get("change_pct")
    lines.append(f"- Price: {('$' + format(price, ',.2f')) if price else 'n/a'}"
                 + (f" ({chg:+.2f}% vs prev close)" if chg is not None else ""))

    v = m.get("valuation", {})
    lines.append(f"- Valuation (calculated): P/E {_ratio(v.get('pe'))}, "
                 f"P/S {_ratio(v.get('ps'))}, EV/EBITDA {_ratio(v.get('ev_ebitda'))} "
                 f"(EV {_money(v.get('ev'))})")

    g = m.get("growth", {})
    rc = g.get("revenue_cagr_3y")
    ec = g.get("eps_cagr_3y")
    rev_str = f"{_pct(rc['value'])} ({rc['years']}y)" if rc else "n/a"
    eps_str = f"{_pct(ec['value'])} ({ec['years']}y)" if ec else "n/a"
    lines.append(f"- Growth: revenue CAGR {rev_str}, EPS CAGR {eps_str}")

    mg = m.get("margins", {})
    lines.append(f"- Margins: gross {_pct(mg.get('gross'))}, operating {_pct(mg.get('operating'))}")

    c = m.get("cash", {})
    lines.append(f"- Free cash flow (OCF − CapEx): {_money(c.get('fcf'))} "
                 f"(FCF margin {_pct(c.get('fcf_margin'))})")

    pt = m.get("price_targets")
    if pt:
        a = pt.get("assumptions", {})
        lines.append("- 2-year price targets (Python scenario model, "
                     f"method={pt.get('method')}):")
        for scn in ("bear", "base", "bull"):
            if scn in pt:
                t = pt[scn]
                ret = t.get("implied_return")
                assumptions = a.get(scn, {})
                assum_str = ", ".join(f"{k}={v}" for k, v in assumptions.items())
                lines.append(
                    f"    - {scn.capitalize()}: ${_ratio(t.get('target'))}"
                    + (f" ({ret * 100:+.1f}%, ~{t.get('implied_cagr', 0) * 100:+.1f}%/yr)"
                       if ret is not None else "")
                    + (f"  [assumes {assum_str}]" if assum_str else "")
                )
    else:
        lines.append("- 2-year price targets: n/a (insufficient data to model)")

    h = m.get("history", {})
    if h.get("max_drawdown") is not None:
        lines.append(f"- Historical max drawdown (full history): {_pct(h['max_drawdown'])}")
    vp = h.get("valuation_percentile")
    if vp:
        w = vp.get("window", {})
        lines.append(
            f"- Valuation percentile: current P/E {vp['current']} (annual-EPS basis) "
            f"sits at the {vp['percentile']:.0f}/100 percentile (0=cheapest, 100=priciest) "
            f"of its own last ~{w.get('years')}y (median {vp['median']}, n={w.get('observations')})"
        )
    br = h.get("forward_base_rate")
    if br:
        lines.append(
            f"- 2y forward base rate — when P/E was in its '{br['current_bucket']}' "
            f"bucket (n={br['n']}): median 2y return {br['median_return'] * 100:+.1f}%, "
            f"{br['pct_positive']:.0f}% positive [{br['caveat']}]"
        )
    for note in h.get("notes", []) or []:
        lines.append(f"- Note: {note}")

    return "\n".join(lines)


def format_many_for_prompt(metrics_by_ticker: Dict[str, Dict[str, Any]]) -> str:
    """Render all shortlisted tickers' computed figures into one injectable block."""
    if not metrics_by_ticker:
        return ""
    blocks = [format_metrics_for_prompt(m) for m in metrics_by_ticker.values() if m]
    if not blocks:
        return ""
    date = datetime.now().strftime("%Y-%m-%d")
    body = "\n\n".join(blocks)
    return (
        f"PYTHON-COMPUTED MARKET DATA (Current data as of {date}). "
        "These figures are calculated deterministically from yfinance price and "
        "fundamental data — treat them as ground truth and interpret them; do not "
        "invent or override them:\n\n"
        f"{body}"
    )
