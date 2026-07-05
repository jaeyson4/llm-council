"""Deterministic stock metrics computed in Python (never by an LLM).

Given a ticker, this module pulls price + fundamentals + price history from
yfinance ONCE (cached), then CALCULATES valuation ratios, growth, margins, free
cash flow, per-stock 2-year scenario price targets, and historical context
(valuation percentile vs the stock's own history, max drawdown, and 2-year
forward base rates by valuation bucket).

Design principles:
- Every number here is computed in Python from raw statement/price data. The LLM
  receives these figures and interprets them; it never invents them.
- Fail safe. Any missing field yields None for that metric, never an exception,
  so the research flow continues even with partial data.
- One network fetch per ticker, cached with a TTL, so all council models and
  both stages share a single fetch instead of refetching per model.

Two DISTINCT history windows, always reported so nothing is overread:
- PRICE window: yfinance serves full daily history (`period="max"`), i.e. often
  decades. Drawdowns and the *unconditional* forward base rate use this full
  window (10y+ for any established name).
- FUNDAMENTALS window: yfinance's free annual statements only reach back ~4
  fiscal years. Anything built on reported EPS (the P/E history -> valuation
  percentile, and the valuation-BUCKETED base rate) is bounded by that ceiling.
  We surface the true fundamentals window length ("based on N.Ny of data") and
  down-weight thin samples rather than pretend to a 10y valuation history that
  the data cannot support.

Forward base rates are computed over overlapping daily windows, which are highly
autocorrelated, so we also report the number of NON-overlapping (independent)
holding periods and flag the stat "low confidence" when there are too few.

Scenario price targets are derived PER STOCK from that stock's own realized
growth plus yfinance analyst estimates (next-2-fiscal-year EPS/revenue growth,
long-term growth, and analyst dispersion), and exit multiples mean-revert toward
the stock's own historical median P/E. A mature bank and a hypergrowth chipmaker
therefore get materially different assumptions, all printed for inspection.
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

# Target lookback for the data layer. yfinance serves full price history, so this
# is comfortably met for price-based stats; fundamentals are capped by yfinance
# at ~4 annual statements regardless, and we report whatever window we actually
# got ("use max available if a stock is younger").
_TARGET_HISTORY_YEARS = 10

# Forward-return horizon for the base-rate study (2 years).
_FWD_HORIZON_DAYS = 730

# Minimum number of (overlapping) forward-return observations before a base-rate
# stat is worth reporting at all. Below this the median is too noisy to show.
_MIN_BASE_RATE_OBS = 40

# Minimum number of NON-overlapping (independent) holding periods before a base
# rate is treated as trustworthy. Fewer than this -> labeled "low confidence",
# because overlapping daily windows massively overstate the true sample size.
_MIN_INDEPENDENT_PERIODS = 5

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


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


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


def _safe_loc(df, row_label, col_label) -> Optional[float]:
    """Read a single cell from a yfinance estimates DataFrame as a float, or
    None if the frame/row/column is missing or the value isn't numeric."""
    if pd is None or df is None:
        return None
    try:
        if getattr(df, "empty", True):
            return None
        if row_label in df.index and col_label in df.columns:
            return _num(df.loc[row_label, col_label])
    except Exception:
        return None
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


def _cagr(series, target_years: int = _TARGET_HISTORY_YEARS) -> Optional[Dict[str, Any]]:
    """Compute compound annual growth rate over up to target_years from an
    oldest->newest Series. yfinance annual statements only span ~4 years, so the
    default target is set high enough that this effectively uses the FULL
    available fundamentals window (earliest->latest). Returns {'value', 'years'}
    or None. Requires positive endpoints (CAGR is undefined across a sign
    change)."""
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


def _weighted_mean(pairs) -> Optional[float]:
    """Weighted mean of [(value, weight), ...], ignoring None values. None if
    nothing usable."""
    usable = [(v, w) for v, w in pairs if v is not None and w > 0]
    if not usable:
        return None
    wsum = sum(w for _, w in usable)
    return sum(v * w for v, w in usable) / wsum if wsum > 0 else None


def _count_non_overlapping(dates, horizon_days: int) -> int:
    """Greedily count NON-overlapping holding periods among (possibly
    non-contiguous) start dates: take the earliest start, skip every later start
    that begins within `horizon_days` of it, take the next one after that, and so
    on. This is the number of statistically INDEPENDENT samples; for the
    autocorrelated overlapping daily windows we study it is far smaller than the
    raw observation count and is what should temper confidence."""
    if dates is None or len(dates) == 0:
        return 0
    try:
        ordered = sorted(pd.to_datetime(list(dates)))
    except Exception:
        return 0
    horizon = pd.Timedelta(days=horizon_days)
    count = 0
    next_ok = None
    for d in ordered:
        if next_ok is None or d >= next_ok:
            count += 1
            next_ok = d + horizon
    return count


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
    # Full available price history (decades for established names).
    try:
        raw["history"] = t.history(period="max", auto_adjust=True)
    except Exception:
        raw["history"] = None
    # Analyst estimates power the per-stock scenario growth (CHANGE 2). Each is
    # optional and can be missing/raise for ETFs or thin names -> swallow to None.
    estimates: Dict[str, Any] = {}
    for key, attr in (
        ("earnings_estimate", "earnings_estimate"),
        ("revenue_estimate", "revenue_estimate"),
        ("growth_estimates", "growth_estimates"),
    ):
        try:
            estimates[key] = getattr(t, attr)
        except Exception:
            estimates[key] = None
    raw["estimates"] = estimates
    return raw


# ---------------------------------------------------------------------------
# Analyst-estimate extraction (for per-stock scenario growth)
# ---------------------------------------------------------------------------
def _extract_analyst_growth(estimates: Dict[str, Any], info: Dict[str, Any]) -> Dict[str, Any]:
    """Pull forward growth signals out of yfinance's analyst-estimate frames.

    The `0y` (current fiscal year) and `+1y` (next fiscal year) growth rates
    together span roughly the 2-year projection horizon, which is exactly what
    the price targets need. We also capture long-term growth (LTG) and the
    analyst high/low dispersion at +1y (used to size the bull/bear spread), with
    graceful fallbacks to info's YoY growth fields when the frames are absent."""
    estimates = estimates or {}
    info = info or {}
    ee = estimates.get("earnings_estimate")
    re_ = estimates.get("revenue_estimate")
    ge = estimates.get("growth_estimates")

    out: Dict[str, Any] = {
        "eps_growth_0y": _safe_loc(ee, "0y", "growth"),
        "eps_growth_1y": _safe_loc(ee, "+1y", "growth"),
        "rev_growth_0y": _safe_loc(re_, "0y", "growth"),
        "rev_growth_1y": _safe_loc(re_, "+1y", "growth"),
        "ltg": _safe_loc(ge, "LTG", "stockTrend"),
        "range_1y": None,
        "forward_pe": _num(info.get("forwardPE")),
        "info_earnings_growth": _num(info.get("earningsGrowth")),
        "info_revenue_growth": _num(info.get("revenueGrowth")),
    }

    # Analyst dispersion at +1y: (high - low) / avg, a per-stock uncertainty gauge.
    hi = _safe_loc(ee, "+1y", "high")
    lo = _safe_loc(ee, "+1y", "low")
    avg = _safe_loc(ee, "+1y", "avg")
    if hi is not None and lo is not None and avg not in (None, 0):
        out["range_1y"] = abs(hi - lo) / abs(avg)

    # Fall back to the growth_estimates frame's stockTrend, then to info's YoY.
    if out["eps_growth_1y"] is None:
        out["eps_growth_1y"] = _safe_loc(ge, "+1y", "stockTrend")
    if out["eps_growth_0y"] is None:
        out["eps_growth_0y"] = _safe_loc(ge, "0y", "stockTrend")
    if out["eps_growth_0y"] is None and out["eps_growth_1y"] is None:
        out["eps_growth_0y"] = out["info_earnings_growth"]
    if out["rev_growth_0y"] is None and out["rev_growth_1y"] is None:
        out["rev_growth_0y"] = out["info_revenue_growth"]
    return out


# ---------------------------------------------------------------------------
# Historical context (valuation percentile, drawdown, forward base rates)
# ---------------------------------------------------------------------------
def _forward_returns(close, horizon_days: int):
    """Per start day, the realized forward return `horizon_days` later. Weekend/
    holiday target dates are time-interpolated, but any target beyond the last
    real close is MASKED (dropped) so unripe recent start dates aren't
    flat-carried to today's price — which would fabricate returns for the most
    recent ~2 years. Returns a DataFrame indexed by start date with a 'fwd'
    column, or None."""
    if close is None or len(close) < 2:
        return None
    try:
        last_date = close.index.max()
        base = close.to_frame("p0").copy()
        target_dates = base.index + pd.Timedelta(days=horizon_days)
        p_future = (
            close.reindex(close.index.union(target_dates))
            .interpolate(method="time")
            .reindex(target_dates)
        )
        vals = p_future.values.astype(float)
        beyond = target_dates > last_date
        beyond = beyond.values if hasattr(beyond, "values") else beyond
        vals[beyond] = float("nan")
        base["pf"] = vals
        base["fwd"] = base["pf"] / base["p0"] - 1.0
        return base.dropna(subset=["fwd"])
    except Exception:
        return None


def _summarize_forward_returns(returns, dates, horizon_days: int, window: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Summarize a set of forward returns into a base-rate stat, reporting BOTH
    the raw (overlapping) sample size and the number of independent
    (non-overlapping) periods, and flagging low confidence when the latter is
    small."""
    if returns is None or len(returns) < _MIN_BASE_RATE_OBS:
        return None
    n = int(len(returns))
    n_indep = _count_non_overlapping(dates, horizon_days)
    horizon_years = round(horizon_days / 365.25, 1)
    return {
        "n": n,
        "n_independent": n_indep,
        "median_return": round(float(returns.median()), 4),
        "mean_return": round(float(returns.mean()), 4),
        "pct_positive": round(float((returns > 0).mean() * 100.0), 1),
        "confidence": "low" if n_indep < _MIN_INDEPENDENT_PERIODS else "ok",
        "window": window,
        "caveat": (
            f"overlapping daily windows (autocorrelated); only {n_indep} "
            f"non-overlapping {horizon_years}y periods in the sample"
        ),
    }


def _historical_context(info, income, history) -> Dict[str, Any]:
    """Drawdown + unconditional 2y base rate use the FULL price window (decades).
    The P/E-history features (valuation percentile, valuation-bucketed base rate,
    and the historical-median-P/E exit anchor) are bounded by yfinance's ~4y of
    annual EPS; every stat carries its own window length and sample size."""
    out: Dict[str, Any] = {
        "max_drawdown": None,
        "price_window": None,
        "valuation_window": None,
        "valuation_percentile": None,
        "hist_median_pe": None,
        "forward_base_rate": None,               # valuation-bucket conditioned
        "forward_base_rate_unconditional": None,  # full-history unconditional
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
        close = close.sort_index()
    except Exception:
        return out

    # --- Price window (full available history) ---
    price_window = {
        "start": str(close.index.min().date()),
        "end": str(close.index.max().date()),
        "years": round((close.index.max() - close.index.min()).days / 365.25, 1),
        "observations": int(len(close)),
    }
    out["price_window"] = price_window

    # --- Max drawdown over full available history ---
    try:
        running_max = close.cummax()
        out["max_drawdown"] = _num((close / running_max - 1.0).min())
    except Exception:
        pass

    # --- Forward 2y returns over the full price window (built once, reused) ---
    fwd = _forward_returns(close, _FWD_HORIZON_DAYS)

    # --- Unconditional 2y forward base rate over the FULL window ---
    if fwd is not None and len(fwd) >= _MIN_BASE_RATE_OBS:
        ripe_window = {
            "start": str(fwd.index.min().date()),
            "end": str(fwd.index.max().date()),
            "years": round((fwd.index.max() - fwd.index.min()).days / 365.25, 1),
            "observations": int(len(fwd)),
        }
        out["forward_base_rate_unconditional"] = _summarize_forward_returns(
            fwd["fwd"], fwd.index, _FWD_HORIZON_DAYS, ripe_window
        )

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
            "valuation-history features unavailable (insufficient annual EPS data "
            "from yfinance)"
        )
        return out

    valuation_window = {
        "start": str(pe_series.index.min().date()),
        "end": str(pe_series.index.max().date()),
        "years": round((pe_series.index.max() - pe_series.index.min()).days / 365.25, 1),
        "observations": int(len(pe_series)),
    }
    out["valuation_window"] = valuation_window
    out["hist_median_pe"] = round(float(pe_series.median()), 2)
    if valuation_window["years"] < 5:
        out["notes"].append(
            f"valuation history spans only {valuation_window['years']}y "
            "(bounded by yfinance's ~4y of annual statements) — treat "
            "P/E-percentile and valuation-bucketed base-rate stats as indicative"
        )

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
                "window": valuation_window,
            }
        except Exception:
            pass

    # --- 2y forward base rate conditioned on the current valuation bucket ---
    if fwd is not None and ref_pe is not None:
        try:
            q33, q67 = pe_series.quantile(0.33), pe_series.quantile(0.67)

            def bucket_of(pe_val):
                if pe_val <= q33:
                    return "cheap"
                if pe_val >= q67:
                    return "expensive"
                return "mid"

            cur_bucket = bucket_of(ref_pe)
            # Attach each start day's already-reported P/E, keep current-bucket days.
            pe_on_fwd = pe_series.reindex(fwd.index, method="ffill")
            cond = fwd.assign(pe=pe_on_fwd).dropna(subset=["pe"])
            cond = cond.assign(bucket=cond["pe"].apply(bucket_of))
            sub = cond[cond["bucket"] == cur_bucket]
            stat = _summarize_forward_returns(
                sub["fwd"], sub.index, _FWD_HORIZON_DAYS, valuation_window
            )
            if stat is not None:
                stat["current_bucket"] = cur_bucket
                out["forward_base_rate"] = stat
            else:
                out["notes"].append(
                    f"valuation-bucketed 2y base rate for the '{cur_bucket}' bucket "
                    f"has <{_MIN_BASE_RATE_OBS} obs; omitted"
                )
        except Exception:
            out["notes"].append("valuation-bucketed base-rate computation failed")

    return out


# ---------------------------------------------------------------------------
# Per-stock scenario derivation (growth + exit multiple) — CHANGE 2
# ---------------------------------------------------------------------------
def _derive_scenarios(hist_growth: Optional[float], analyst: Dict[str, Any], method: str) -> Dict[str, Any]:
    """Derive bear/base/bull growth rates from THIS stock's own data — its
    realized historical growth blended with yfinance analyst forward estimates —
    and size the bull/bear spread from how much those signals disagree plus the
    analyst dispersion. Mature, low-growth, tightly-covered names collapse to a
    narrow band; hypergrowth names with wide analyst ranges get a wide band."""
    analyst = analyst or {}
    if method == "eps_pe":
        g0, g1 = analyst.get("eps_growth_0y"), analyst.get("eps_growth_1y")
    else:
        g0, g1 = analyst.get("rev_growth_0y"), analyst.get("rev_growth_1y")
    ltg = analyst.get("ltg")

    # Clamp each raw signal before blending. Recent explosive growth (e.g. a
    # chipmaker printing +200% YoY) mean-reverts hard over a 2y horizon, so we
    # cap single-signal influence rather than let it dominate the projection.
    def cl(x):
        return None if _num(x) is None else _clamp(_num(x), -0.30, 0.60)

    hist_c, g0_c, g1_c, ltg_c = cl(hist_growth), cl(g0), cl(g1), cl(ltg)

    # Analyst view over ~the 2-year projection window = avg(current yr, next yr).
    fwd_vals = [v for v in (g0_c, g1_c) if v is not None]
    analyst_2y = sum(fwd_vals) / len(fwd_vals) if fwd_vals else None

    # Base growth: weighted blend (analyst forward view weighted highest since it
    # is forward-looking and covers exactly our horizon).
    base_g = _weighted_mean([(analyst_2y, 2.0), (hist_c, 1.0), (ltg_c, 1.0)])
    if base_g is None:
        base_g = 0.08  # modest default when the stock exposes no growth signal
    base_g = _clamp(base_g, -0.10, 0.50)

    # Spread from (a) growth level, (b) disagreement among signals, (c) analyst
    # dispersion. Each pushes the band wider; all bounded so nothing runs away.
    used = [v for v in (analyst_2y, hist_c, ltg_c) if v is not None]
    disagree = (max(used) - min(used)) if len(used) >= 2 else 0.0
    arange = analyst.get("range_1y") or 0.0
    up = _clamp(0.25 * abs(base_g) + 0.40 * disagree + 0.25 * arange, 0.04, 0.20)
    down = _clamp(up * 1.25, 0.04, 0.30)  # downside a touch fatter

    bull_g = _clamp(base_g + up, -0.10, 0.65)
    bear_g = _clamp(base_g - down, -0.30, 0.40)

    parts = []
    if analyst_2y is not None:
        parts.append(f"analyst next-2y {analyst_2y * 100:.0f}%")
    if hist_c is not None:
        parts.append(f"hist {hist_c * 100:.0f}%")
    if ltg_c is not None:
        parts.append(f"LTG {ltg_c * 100:.0f}%")
    basis = ("blend of " + ", ".join(parts)) if parts else "default 8% (no growth signal available)"

    return {
        "base": base_g,
        "bull": bull_g,
        "bear": bear_g,
        "growth_basis": basis,
        "spread": {"up": round(up, 3), "down": round(down, 3)},
        "inputs": {
            "hist_growth": round(hist_c, 3) if hist_c is not None else None,
            "analyst_next_2y": round(analyst_2y, 3) if analyst_2y is not None else None,
            "ltg": round(ltg_c, 3) if ltg_c is not None else None,
            "analyst_dispersion": round(arange, 3) if arange else None,
        },
    }


def _tether_hist_median(hist_median: Optional[float], current: Optional[float]):
    """Tether the stock's historical-median multiple to a band around its CURRENT
    multiple before it anchors the exit multiple. This defangs the annual-EPS
    artifact: for a hypergrowth name whose early low-EPS years inflate the median
    (e.g. a chipmaker showing a 'median P/E' of 99 while trading at 30), an
    untethered median would drag the anchor so high that even the BEAR scenario
    expands the multiple. Returns (value_used, was_tethered). No current multiple
    -> pass the median through unchanged."""
    if hist_median is None:
        return None, False
    if current is None or current <= 0:
        return hist_median, False
    lo_b, hi_b = current * 0.6, current * 1.8
    tethered = _clamp(hist_median, lo_b, hi_b)
    return tethered, (abs(tethered - hist_median) > 1e-9)


def _exit_anchor(current: Optional[float], hist_median: Optional[float],
                 forward: Optional[float], lo: float, hi: float) -> Optional[float]:
    """Blend today's multiple, the stock's own (tethered) historical-median
    multiple, and the forward multiple into a mean-reverting exit anchor
    (historical median weighted highest so exits revert toward the stock's own
    norm rather than freezing today's multiple). Returns the clamped anchor or
    None."""
    anchor = _weighted_mean([(hist_median, 0.5), (current, 0.3), (forward, 0.2)])
    return _clamp(anchor, lo, hi) if anchor is not None else None


def _multiple_basis(anchor: float, raw_hist_median: Optional[float], used_hist_median: Optional[float],
                    tethered: bool, current: Optional[float], forward: Optional[float],
                    spread: float) -> str:
    """Human-readable basis string for the exit multiple, keeping the RAW
    historical median visible even when it was tethered, so the assumption stays
    sanity-checkable."""
    src = []
    if raw_hist_median is not None:
        seg = f"hist median {raw_hist_median:.1f}"
        if tethered and used_hist_median is not None:
            seg += f"→{used_hist_median:.1f} (tethered to current)"
        src.append(seg)
    if current is not None:
        src.append(f"current {current:.1f}")
    if forward is not None:
        src.append(f"fwd {forward:.1f}")
    inner = (" (" + ", ".join(src) + ")") if src else ""
    return f"anchor {anchor:.1f}{inner}, ±{spread * 100:.0f}% by scenario"


# ---------------------------------------------------------------------------
# Price targets (2-year, bull/base/bear) — per-stock transparent scenario model
# ---------------------------------------------------------------------------
def _price_targets(
    price: Optional[float],
    eps_ttm: Optional[float],
    current_pe: Optional[float],
    forward_pe: Optional[float],
    hist_median_pe: Optional[float],
    rev_ps: Optional[float],
    current_ps: Optional[float],
    hist_growth: Optional[float],
    analyst: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Project 2 years out under three scenarios whose growth and exit multiples
    are BOTH derived from this specific stock's data (CHANGE 2). Prefers an
    EPS × exit-P/E model; falls back to a revenue-per-share × exit-P/S model for
    unprofitable names. Every scenario returns its explicit growth + exit-multiple
    assumptions (and the basis for each) so the council can attack them."""
    price = _num(price)
    eps_ttm, current_pe, forward_pe, hist_median_pe = (
        _num(eps_ttm), _num(current_pe), _num(forward_pe), _num(hist_median_pe),
    )
    rev_ps, current_ps = _num(rev_ps), _num(current_ps)

    if eps_ttm is not None and eps_ttm > 0 and (current_pe or hist_median_pe or forward_pe):
        method = "eps_pe"
    elif rev_ps is not None and rev_ps > 0 and current_ps is not None and current_ps > 0:
        method = "ps"
    else:
        return None  # not enough to model a defensible target

    scen = _derive_scenarios(hist_growth, analyst, method)
    base_g = scen["base"]

    # Exit-multiple anchor + per-scenario expansion/compression. Higher-growth
    # names re-rate harder, so the multiple spread scales with base growth.
    mult_spread = _clamp(0.10 + 0.30 * abs(base_g), 0.10, 0.35)
    if method == "eps_pe":
        used_hm, tethered = _tether_hist_median(hist_median_pe, current_pe)
        anchor = _exit_anchor(current_pe, used_hm, forward_pe, 5.0, 50.0)
        if anchor is None:
            anchor = _clamp(current_pe or 15.0, 5.0, 50.0)
        mult_basis = _multiple_basis(anchor, hist_median_pe, used_hm, tethered,
                                     current_pe, forward_pe, mult_spread)
        exit_mult = {
            "bull": _clamp(anchor * (1 + mult_spread), 4.0, 60.0),
            "base": _clamp(anchor, 4.0, 60.0),
            "bear": _clamp(anchor * (1 - mult_spread * 1.3), 4.0, 60.0),
        }
        per_share, growth_key, mult_key = eps_ttm, "eps_growth", "exit_pe"
    else:
        anchor = _exit_anchor(current_ps, None, None, 0.3, 30.0)
        if anchor is None:
            anchor = _clamp(current_ps or 3.0, 0.3, 30.0)
        mult_basis = _multiple_basis(anchor, None, None, False,
                                     current_ps, None, mult_spread)
        exit_mult = {
            "bull": _clamp(anchor * (1 + mult_spread), 0.3, 35.0),
            "base": _clamp(anchor, 0.3, 35.0),
            "bear": _clamp(anchor * (1 - mult_spread * 1.3), 0.3, 35.0),
        }
        per_share, growth_key, mult_key = rev_ps, "revenue_growth", "exit_ps"

    targets: Dict[str, Any] = {
        "method": method,
        "assumptions": {
            "growth_basis": scen["growth_basis"],
            "multiple_basis": mult_basis,
            "inputs": scen["inputs"],
        },
    }
    for name in ("bear", "base", "bull"):
        g = scen[name]
        mult = exit_mult[name]
        future = per_share * (1 + g) ** 2
        target_price = future * mult
        entry = {"target": round(target_price, 2)}
        if price and price > 0:
            entry["implied_return"] = round(target_price / price - 1.0, 3)
            entry["implied_cagr"] = round((target_price / price) ** 0.5 - 1.0, 3)
        targets[name] = entry
        targets["assumptions"][name] = {
            growth_key: round(g, 3),
            mult_key: round(mult, 1),
        }
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
    estimates = raw.get("estimates") or {}

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

    # --- Growth (from annual statements, full available fundamentals window) ---
    rev_series = _row_series(income, "Total Revenue", "Operating Revenue")
    eps_ann_series = _row_series(income, "Diluted EPS", "Basic EPS")
    rev_cagr = _cagr(rev_series)
    eps_cagr = _cagr(eps_ann_series)

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

    # --- Historical context (computed BEFORE price targets so the exit-multiple
    #     anchor can mean-revert toward the stock's own historical median P/E) ---
    history_ctx = _historical_context(info, income, history)

    # --- Price targets (per-stock scenario assumptions) ---
    rev_ps = _safe_div(revenue_ttm, shares)
    hist_growth = None
    if eps_cagr is not None:
        hist_growth = eps_cagr["value"]
    elif rev_cagr is not None:
        hist_growth = rev_cagr["value"]
    analyst_growth = _extract_analyst_growth(estimates, info)
    price_targets = _price_targets(
        price=price,
        eps_ttm=eps_ttm,
        current_pe=pe,
        forward_pe=analyst_growth.get("forward_pe"),
        hist_median_pe=history_ctx.get("hist_median_pe"),
        rev_ps=rev_ps,
        current_ps=ps,
        hist_growth=hist_growth,
        analyst=analyst_growth,
    )

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
        "growth": {"revenue_cagr": rev_cagr, "eps_cagr": eps_cagr},
        "margins": {"gross": gross_margin, "operating": op_margin},
        "cash": {"fcf": fcf, "fcf_margin": fcf_margin},
        "fundamentals": {"market_cap": market_cap, "revenue_ttm": revenue_ttm,
                         "ebitda": ebitda, "total_debt": total_debt,
                         "total_cash": total_cash, "shares": shares, "eps_ttm": eps_ttm},
        "analyst": analyst_growth,
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


def _base_rate_line(label: str, br: Dict[str, Any]) -> str:
    """One human-readable base-rate line: overlapping n, independent-period
    count, and an explicit low-confidence flag when the independent sample is
    thin."""
    w = br.get("window", {}) or {}
    conf = " — ⚠ LOW CONFIDENCE" if br.get("confidence") == "low" else ""
    years = w.get("years")
    span = f" over ~{years}y" if years is not None else ""
    return (
        f"- 2y forward base rate ({label}{span}): "
        f"median {br['median_return'] * 100:+.1f}%, {br['pct_positive']:.0f}% positive "
        f"[n={br['n']} overlapping, but only {br['n_independent']} non-overlapping "
        f"2y period(s)]{conf}"
    )


def format_metrics_for_prompt(m: Dict[str, Any]) -> str:
    """Render one ticker's computed metrics as a labeled block for injection into
    a model prompt. All figures are Python-computed."""
    if not m:
        return ""
    lines: List[str] = []
    head = m["ticker"] + (f" ({m['name']})" if m.get("name") else "")
    lines.append(f"### {head} — computed figures (as of {m.get('as_of')})")

    # Data window banner: price history vs (thinner) fundamentals history.
    h = m.get("history", {}) or {}
    pw = h.get("price_window") or {}
    vw = h.get("valuation_window") or {}
    if pw.get("years") is not None:
        seg = f"based on {pw['years']}y of price history"
        if vw.get("years") is not None:
            seg += (f"; {vw['years']}y of fundamentals "
                    "(yfinance annual-statement ceiling ~4y)")
        else:
            seg += "; fundamentals history unavailable"
        lines.append(f"- Data window: {seg}")

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
    rc = g.get("revenue_cagr")
    ec = g.get("eps_cagr")
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
        lines.append("- 2-year price targets (Python per-stock scenario model, "
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
        if a.get("growth_basis"):
            lines.append(f"    - Growth assumptions from: {a['growth_basis']}")
        if a.get("multiple_basis"):
            lines.append(f"    - Exit-multiple basis: {a['multiple_basis']}")
    else:
        lines.append("- 2-year price targets: n/a (insufficient data to model)")

    if h.get("max_drawdown") is not None:
        span = f" over ~{pw['years']}y" if pw.get("years") is not None else " (full history)"
        lines.append(f"- Historical max drawdown{span}: {_pct(h['max_drawdown'])}")
    vp = h.get("valuation_percentile")
    if vp:
        w = vp.get("window", {})
        lines.append(
            f"- Valuation percentile: current P/E {vp['current']} (annual-EPS basis) "
            f"sits at the {vp['percentile']:.0f}/100 percentile (0=cheapest, 100=priciest) "
            f"of its own last ~{w.get('years')}y (median {vp['median']}, n={w.get('observations')})"
        )
    bru = h.get("forward_base_rate_unconditional")
    if bru:
        lines.append(_base_rate_line("unconditional, full history", bru))
    brc = h.get("forward_base_rate")
    if brc:
        lines.append(_base_rate_line(
            f"when P/E in its '{brc.get('current_bucket')}' bucket", brc))
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
        "invent or override them. Scenario growth/exit-multiple assumptions are "
        "derived per-stock from each name's own history + analyst estimates and "
        "are printed so you can sanity-check them:\n\n"
        f"{body}"
    )
