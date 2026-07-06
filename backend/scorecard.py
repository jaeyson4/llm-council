"""Pick-tracking scorecard — grades every past analysis note in the Obsidian vault.

Each stock analysis is exported as a dated note (`<TICKER>-<YYYY-MM-DD>.md`) whose
YAML frontmatter records the call: ticker, date, current_price (the price WHEN
CALLED), and bear/base/bull targets. This module:

1. Reads every analysis note in the vault (skipping screening/hub/scorecard notes).
2. Pulls current + historical prices from yfinance (one batched download).
3. Computes IN PYTHON, per pick: return since the call date, gap/progress to the
   base-case target, and the return at each elapsed 1mo/3mo/6mo/1yr checkpoint
   (directional calibration before the 2-year thesis horizon resolves).
4. Benchmarks every pick against SPY over its own holding period (alpha).
5. Writes a single, re-runnable `Scorecard.md` note: a per-pick table, a
   checkpoint table, and portfolio summary stats (avg return, hit rate, vs-SPY).

Design mirrors the rest of the backend: all math in Python (never an LLM),
fail-safe (a ticker that won't price is skipped, not fatal), and idempotent —
re-running overwrites `Scorecard.md` in place, so it always reflects today.

Run it:  python -m backend.scorecard
"""

import glob
import os
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None

try:
    import yfinance as yf
    import pandas as pd
    import logging
    logging.getLogger("yfinance").setLevel(logging.CRITICAL)
except Exception:  # pragma: no cover
    yf = None
    pd = None

from .config import OBSIDIAN_VAULT_PATH

# Benchmark and checkpoint schedule.
_BENCHMARK = "SPY"
_CHECKPOINTS = [("1mo", 1), ("3mo", 3), ("6mo", 6), ("1yr", 12)]
_HORIZON_DAYS = 730  # the 2-year thesis horizon the picks are ultimately judged on
_SCORECARD_FILENAME = "Scorecard.md"


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def _num(value: Any) -> Optional[float]:
    """Coerce to float; blank/garbage/NaN -> None."""
    if value is None:
        return None
    try:
        f = float(str(value).replace(",", "").strip())
    except (TypeError, ValueError):
        return None
    return f if f == f else None  # drop NaN


def _pct(x: Optional[float], nd: int = 1) -> str:
    return f"{x * 100:+.{nd}f}%" if x is not None else "—"


def _usd(x: Optional[float]) -> str:
    return f"${x:,.2f}" if x is not None else "—"


# ---------------------------------------------------------------------------
# Read the picks out of the vault
# ---------------------------------------------------------------------------
def _parse_frontmatter(text: str) -> Optional[Dict[str, Any]]:
    """Return the YAML frontmatter of a note as a dict, or None if there is no
    frontmatter block. Uses PyYAML when available (robust to hand-edits), else a
    minimal `key: "value"` line parser matching how the notes are written."""
    if not text.startswith("---"):
        return None
    m = re.match(r"^---\s*\n(.*?)\n---\s*(?:\n|$)", text, re.DOTALL)
    if not m:
        return None
    block = m.group(1)
    if yaml is not None:
        try:
            data = yaml.safe_load(block)
            return data if isinstance(data, dict) else None
        except Exception:
            pass  # fall back to the line parser
    out: Dict[str, Any] = {}
    for line in block.splitlines():
        lm = re.match(r'^\s*([A-Za-z0-9_]+)\s*:\s*(.*?)\s*$', line)
        if lm:
            val = lm.group(2)
            if len(val) >= 2 and val[0] == val[-1] and val[0] in "\"'":
                val = val[1:-1]
            out[lm.group(1)] = val
    return out or None


def read_picks(vault_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """Scan the vault for analysis notes and return one pick dict per note:
    {ticker, date, call_price, target_base, thesis, filename}. Skips screening,
    hub (no frontmatter), and the scorecard note itself — anything without a
    ticker+date, or carrying a `type:` key, is not a pick."""
    vault = vault_path if vault_path is not None else OBSIDIAN_VAULT_PATH
    picks: List[Dict[str, Any]] = []
    if not vault or not os.path.isdir(vault):
        return picks
    for path in sorted(glob.glob(os.path.join(vault, "*.md"))):
        try:
            with open(path, "r", encoding="utf-8") as f:
                fm = _parse_frontmatter(f.read())
        except Exception:
            continue
        if not fm or fm.get("type"):  # screening/scorecard notes carry a type
            continue
        ticker = str(fm.get("ticker") or "").strip().upper()
        date = str(fm.get("date") or "").strip()
        call_price = _num(fm.get("current_price"))
        if not ticker or not re.match(r"^\d{4}-\d{2}-\d{2}$", date) or call_price is None:
            continue
        picks.append({
            "ticker": ticker,
            "date": date,
            "call_price": call_price,
            "target_base": _num(fm.get("target_base")),
            "thesis": str(fm.get("thesis") or "").strip(),
            "filename": os.path.basename(path),
        })
    return picks


# ---------------------------------------------------------------------------
# Prices (one batched yfinance download, then as-of lookups)
# ---------------------------------------------------------------------------
def _download_closes(tickers: List[str], start: str):
    """Download daily auto-adjusted closes for `tickers` from `start` to today as a
    (dates x tickers) DataFrame (tz-naive index). Returns None on failure. Handles
    yfinance's single- vs multi-ticker column shapes."""
    if yf is None or pd is None or not tickers:
        return None
    try:
        data = yf.download(tickers, start=start, auto_adjust=True,
                           progress=False, group_by="column")
    except Exception:
        return None
    if data is None or getattr(data, "empty", True):
        return None
    close = None
    try:
        if hasattr(data.columns, "levels"):  # MultiIndex
            lvl0 = set(data.columns.get_level_values(0))
            lvl1 = set(data.columns.get_level_values(1))
            if "Close" in lvl0:
                close = data["Close"]
            elif "Close" in lvl1:
                close = data.xs("Close", axis=1, level=1)
        elif "Close" in data.columns:  # single ticker
            close = data["Close"].to_frame(tickers[0])
    except Exception:
        close = None
    if close is None or getattr(close, "empty", True):
        return None
    try:
        close = close.apply(pd.to_numeric, errors="coerce")
        close.index = pd.to_datetime(close.index).tz_localize(None)
        return close.sort_index()
    except Exception:
        return None


def _asof(series, when) -> Optional[float]:
    """Most recent non-NaN value in `series` at or before `when`, or None."""
    if series is None:
        return None
    try:
        s = series.dropna()
        s = s[s.index <= when]
        return float(s.iloc[-1]) if len(s) else None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Per-pick computation
# ---------------------------------------------------------------------------
def _compute_pick(pick: Dict[str, Any], close_frame, last_date, today) -> Optional[Dict[str, Any]]:
    """Grade one pick against the price data. Returns a row dict, or None if the
    ticker couldn't be priced. All returns are measured from the RECORDED call
    price (the price in the note), so the numbers match what was actually called."""
    ticker = pick["ticker"]
    if close_frame is None or ticker not in close_frame.columns:
        return None
    series = close_frame[ticker].dropna()
    if series.empty:
        return None

    call = pick["call_price"]
    call_date = pd.Timestamp(pick["date"])
    now = _asof(series, last_date)
    if now is None or call is None or call <= 0:
        return None

    days_held = max(0, (today - call_date).days)
    ret = now / call - 1.0

    # SPY benchmark over the SAME window (call date -> now).
    spy = close_frame[_BENCHMARK].dropna() if _BENCHMARK in close_frame.columns else None
    spy_call = _asof(spy, call_date)
    spy_now = _asof(spy, last_date)
    spy_ret = (spy_now / spy_call - 1.0) if (spy_call and spy_now and spy_call > 0) else None
    alpha = (ret - spy_ret) if spy_ret is not None else None

    # Progress / gap toward the base-case (2-year) target.
    base = pick["target_base"]
    gap_to_base = (base / now - 1.0) if (base and now > 0) else None
    status, progress = _base_status(call, now, base, days_held)

    # Checkpoints: return vs call price at each ELAPSED 1/3/6/12-month mark.
    checkpoints: Dict[str, Optional[float]] = {}
    for label, months in _CHECKPOINTS:
        cp_date = call_date + pd.DateOffset(months=months)
        if cp_date <= last_date:
            cp_price = _asof(series, cp_date)
            checkpoints[label] = (cp_price / call - 1.0) if (cp_price and call > 0) else None
        else:
            checkpoints[label] = None  # not yet reached -> pending

    return {
        "ticker": ticker,
        "date": pick["date"],
        "days_held": days_held,
        "call_price": call,
        "current_price": now,
        "return": ret,
        "spy_return": spy_ret,
        "alpha": alpha,
        "base_target": base,
        "gap_to_base": gap_to_base,
        "progress_to_base": progress,
        "status": status,
        "checkpoints": checkpoints,
        "thesis": pick["thesis"],
    }


def _base_status(call: float, now: float, base: Optional[float], days_held: int):
    """(status_label, progress_fraction) vs the base-case target. Progress is
    (now-call)/(base-call) — the fraction of the way from entry to target — and
    the label compares it to the fraction of the 2-year horizon elapsed, so a pick
    is 'ahead' only if its return is outpacing a straight-line path to base."""
    if base is None:
        return "— no base", None
    if base <= call:
        # Model's base target sits at/below the entry price (unusual for a long
        # thesis) — flag it rather than pretend the pick instantly 'hit base'.
        return "⚠️ base≤call", None
    if now >= base:
        return "✅ hit base", (now - call) / (base - call)
    progress = (now - call) / (base - call)
    time_frac = min(1.0, days_held / _HORIZON_DAYS) if days_held > 0 else 0.0
    if progress < 0:
        return "🔴 down", progress
    if progress >= time_frac:
        return "🟢 ahead", progress
    return "🟡 behind", progress


def _summary(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Portfolio-level stats computed in Python from the graded rows."""
    n = len(rows)
    rets = [r["return"] for r in rows if r["return"] is not None]
    alphas = [r["alpha"] for r in rows if r["alpha"] is not None]
    spys = [r["spy_return"] for r in rows if r["spy_return"] is not None]
    hits = sum(1 for r in rets if r > 0)
    beat = sum(1 for a in alphas if a > 0)
    return {
        "n": n,
        "avg_return": (sum(rets) / len(rets)) if rets else None,
        "hit_rate": (hits / len(rets)) if rets else None,
        "hits": hits,
        "n_returns": len(rets),
        "avg_spy": (sum(spys) / len(spys)) if spys else None,
        "avg_alpha": (sum(alphas) / len(alphas)) if alphas else None,
        "beat_spy": beat,
        "n_alpha": len(alphas),
    }


# ---------------------------------------------------------------------------
# Render the Scorecard note
# ---------------------------------------------------------------------------
def render_scorecard(rows: List[Dict[str, Any]], summary: Dict[str, Any],
                     as_of: str, skipped: List[str]) -> str:
    """Assemble the full Scorecard.md markdown (frontmatter + summary + tables).
    Rows are sorted best-return first. Pure function; no I/O."""
    rows = sorted(rows, key=lambda r: (r["return"] is None, -(r["return"] or 0)))

    hit_txt = (f"{summary['hit_rate'] * 100:.0f}% ({summary['hits']}/{summary['n_returns']})"
               if summary["hit_rate"] is not None else "—")
    beat_txt = (f"{summary['beat_spy']}/{summary['n_alpha']} "
                f"({summary['beat_spy'] / summary['n_alpha'] * 100:.0f}%)"
                if summary["n_alpha"] else "—")

    fm = [
        "---",
        'type: "scorecard"',
        f'updated: "{as_of}"',
        f'picks: {summary["n"]}',
        "---",
    ]
    head = [
        "# 📊 Pick Scorecard",
        "",
        f"_Auto-generated by `backend/scorecard.py` — re-run any time to refresh. "
        f"Call price = the price recorded in each analysis note at call time; "
        f"current price = latest yfinance daily close (auto-adjusted). All figures "
        f"computed in Python._",
        "",
        f"**As of {as_of}** · **{summary['n']}** picks · "
        f"avg return **{_pct(summary['avg_return'])}** · "
        f"hit rate **{hit_txt}** · "
        f"avg alpha vs SPY **{_pct(summary['avg_alpha'])}** · "
        f"beat SPY **{beat_txt}**",
    ]

    summary_tbl = [
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Picks tracked | {summary['n']} |",
        f"| Average return since call | {_pct(summary['avg_return'])} |",
        f"| Hit rate (% positive) | {hit_txt} |",
        f"| Average SPY return (same periods) | {_pct(summary['avg_spy'])} |",
        f"| Average alpha (pick − SPY) | {_pct(summary['avg_alpha'])} |",
        f"| Picks beating SPY | {beat_txt} |",
    ]

    picks_tbl = [
        "",
        "## Picks",
        "",
        "| Ticker | Call date | Days | Call $ | Current $ | Return | SPY | Alpha | Base $ | To base | vs base |",
        "|---|---|--:|--:|--:|--:|--:|--:|--:|--:|:--|",
    ]
    for r in rows:
        picks_tbl.append(
            f"| [[{r['ticker']}]] | {r['date']} | {r['days_held']} | "
            f"{_usd(r['call_price'])} | {_usd(r['current_price'])} | "
            f"{_pct(r['return'])} | {_pct(r['spy_return'])} | {_pct(r['alpha'])} | "
            f"{_usd(r['base_target'])} | {_pct(r['gap_to_base'])} | {r['status']} |"
        )

    cp_tbl = [
        "",
        "## Checkpoints — directional calibration toward the 2-yr horizon",
        "",
        "_Return vs call price at each elapsed checkpoint; \"—\" = not yet reached._",
        "",
        "| Ticker | Call date | 1mo | 3mo | 6mo | 1yr |",
        "|---|---|--:|--:|--:|--:|",
    ]
    for r in rows:
        c = r["checkpoints"]
        cp_tbl.append(
            f"| [[{r['ticker']}]] | {r['date']} | {_pct(c.get('1mo'))} | "
            f"{_pct(c.get('3mo'))} | {_pct(c.get('6mo'))} | {_pct(c.get('1yr'))} |"
        )

    foot = []
    if skipped:
        foot = ["", f"> ⚠️ Could not price {len(skipped)} pick(s): "
                f"{', '.join(skipped)} (skipped)."]

    return "\n".join(fm + head + summary_tbl + picks_tbl + cp_tbl + foot) + "\n"


# ---------------------------------------------------------------------------
# Orchestration (the re-runnable entry point)
# ---------------------------------------------------------------------------
def build_scorecard(vault_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Read picks, price them, grade them, and return
    {rows, summary, as_of, skipped, markdown} — or None if there's nothing to
    grade / no price backend. Pure compute + render; does NOT write the file."""
    vault = vault_path if vault_path is not None else OBSIDIAN_VAULT_PATH
    picks = read_picks(vault)
    if not picks:
        print(f"[scorecard] no analysis notes found in vault {vault!r}")
        return None
    if yf is None or pd is None:
        print("[scorecard] yfinance/pandas unavailable — cannot price picks")
        return None

    tickers = sorted({p["ticker"] for p in picks} | {_BENCHMARK})
    earliest = min(p["date"] for p in picks)
    # Small buffer so an as-of lookup at the call date always has a prior close.
    start = (pd.Timestamp(earliest) - pd.Timedelta(days=7)).strftime("%Y-%m-%d")
    close = _download_closes(tickers, start)
    if close is None:
        print("[scorecard] price download failed — no scorecard produced")
        return None

    last_date = close.index.max()
    today = pd.Timestamp(datetime.now().date())
    rows: List[Dict[str, Any]] = []
    skipped: List[str] = []
    for p in picks:
        row = _compute_pick(p, close, last_date, today)
        if row is None:
            skipped.append(f"{p['ticker']} ({p['date']})")
        else:
            rows.append(row)
    if not rows:
        print("[scorecard] no picks could be priced — no scorecard produced")
        return None

    summary = _summary(rows)
    as_of = str(last_date.date())
    markdown = render_scorecard(rows, summary, as_of, skipped)
    return {"rows": rows, "summary": summary, "as_of": as_of,
            "skipped": skipped, "markdown": markdown}


def write_scorecard(vault_path: Optional[str] = None) -> Optional[str]:
    """Build the scorecard and (over)write the single `Scorecard.md` note in the
    vault. Idempotent and re-runnable. Returns the written path, or None."""
    vault = vault_path if vault_path is not None else OBSIDIAN_VAULT_PATH
    result = build_scorecard(vault)
    if result is None:
        return None
    try:
        os.makedirs(vault, exist_ok=True)
        path = os.path.join(vault, _SCORECARD_FILENAME)
        with open(path, "w", encoding="utf-8") as f:
            f.write(result["markdown"])
        s = result["summary"]
        print(f"[scorecard] wrote {s['n']} pick(s) -> {path} "
              f"(avg {_pct(s['avg_return'])}, hit {s['hits']}/{s['n_returns']}, "
              f"as of {result['as_of']})")
        return path
    except Exception as e:
        print(f"[scorecard] failed to write scorecard: {e}")
        return None


if __name__ == "__main__":
    write_scorecard()
