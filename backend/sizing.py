"""Deterministic budget → position-sizing, computed entirely in Python.

Given the council's finalized ranked picks and a dollar budget, allocate the
budget across the names. NOTHING here is decided by an LLM: conviction is the
council's judgment (a number handed in), but every weight, cap, cluster limit
and dollar amount below is pure arithmetic so the logic is fully inspectable
and reproducible.

Fractional shares are assumed, so allocation is done purely in dollars: each
name is deployed at exactly its target weight × budget. No whole-share rounding,
no "too expensive for one share" carve-outs — the only money left in cash is the
intended risk buffer (plus any weight the hard caps genuinely couldn't place).

Weighting philosophy
--------------------
1. Each name gets a base score that BLENDS its conviction (higher = more) with
   its risk (lower drawdown/volatility = more). Concretely
       score = conviction_norm * (median_risk / risk_i) ** risk_aversion
   so a name carrying the basket's median risk gets an inverse-risk factor of
   1.0, a safer name gets >1, a riskier name gets <1.
2. A concentration exponent (kappa) set by the risk mode reshapes those scores:
   conservative flattens them toward equal weight (diversified), aggressive
   sharpens them toward the top-ranked names (concentrated).
3. Correlated names are grouped into clusters (a shared failure mode) and the
   cluster's COMBINED weight is capped, so one macro shock can't sink the book.
4. Every single position is capped, and a cash buffer is always held back.

All of steps 1-4 are logged line-by-line (see `PositionSizing.log`) and rendered
into a markdown table (ticker | % weight | dollar amount | rationale) by
`render_markdown`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Risk modes — each sets how concentrated the book is and how hard it caps.
# ---------------------------------------------------------------------------
# - kappa:         concentration exponent on the score (<1 flatten, >1 sharpen)
# - risk_aversion: exponent on the inverse-risk factor (higher = punish risk more)
# - cash_buffer:   fraction of the budget always held in cash
# - max_position:  hard cap on any single name's weight
# - cluster_cap:   hard cap on a correlated cluster's COMBINED weight
RISK_MODES: Dict[str, Dict[str, float]] = {
    "conservative": {"kappa": 0.6, "risk_aversion": 1.5, "cash_buffer": 0.10,
                     "max_position": 0.20, "cluster_cap": 0.35},
    "balanced":     {"kappa": 1.0, "risk_aversion": 1.0, "cash_buffer": 0.075,
                     "max_position": 0.25, "cluster_cap": 0.45},
    "aggressive":   {"kappa": 1.8, "risk_aversion": 0.6, "cash_buffer": 0.05,
                     "max_position": 0.35, "cluster_cap": 0.60},
}
DEFAULT_MODE = "balanced"

# Correlation at/above this pairwise threshold links two names into a cluster.
# 0.60 sits above typical market-beta co-movement (~0.3-0.5 for unrelated large
# caps) so it captures genuine shared-failure-mode pairs (e.g. two semis) without
# collapsing an entire diversified basket into one cluster.
DEFAULT_CORR_THRESHOLD = 0.60

# Fallback risk inputs when a name exposes no history-derived figure.
_DEFAULT_VOL = 0.40   # annualized
_DEFAULT_DD = 0.50    # drawdown magnitude
_CONVICTION_FLOOR = 0.05  # keep normalized conviction strictly positive


@dataclass
class Pick:
    """One ranked candidate handed to the sizer."""
    ticker: str
    conviction: Optional[float] = None       # 0-100, the council's conviction
    volatility: Optional[float] = None       # annualized, decimal
    max_drawdown: Optional[float] = None      # negative decimal (e.g. -0.55)
    sector: Optional[str] = None


@dataclass
class SizedRow:
    ticker: str
    weight: float                # final fraction of the WHOLE budget
    dollars: float               # dollars deployed (weight * budget), fractional
    conviction: float            # normalized 0-100 actually used
    risk: float                  # composite risk actually used
    cluster: Optional[int]       # cluster id (None = singleton)
    rationale: str


@dataclass
class PositionSizing:
    budget: float
    risk_mode: str
    params: Dict[str, float]
    rows: List[SizedRow]
    clusters: List[List[str]]
    cash: float                  # dollars held in cash (buffer + un-placeable weight)
    invested: float              # dollars actually deployed across the names
    log: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Clustering (correlated names share a failure mode)
# ---------------------------------------------------------------------------
def build_clusters(
    tickers: List[str],
    corr_matrix: Optional[Dict[str, Dict[str, float]]],
    sectors: Optional[Dict[str, str]] = None,
    threshold: float = DEFAULT_CORR_THRESHOLD,
) -> Tuple[List[List[str]], List[str]]:
    """Union tickers into clusters. Two names are linked when their pairwise
    return correlation is >= threshold; when a pair's correlation is unavailable
    we fall back to *same sector* as a coarse shared-failure-mode proxy. Returns
    (clusters, edge_log) where clusters is a list of ticker-lists (order-stable)
    and edge_log explains why each multi-name cluster formed."""
    sectors = sectors or {}
    parent = {t: t for t in tickers}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    edges: List[str] = []
    for i, a in enumerate(tickers):
        for b in tickers[i + 1:]:
            corr = None
            if corr_matrix and a in corr_matrix and b in corr_matrix.get(a, {}):
                corr = corr_matrix[a][b]
            linked = False
            reason = ""
            if corr is not None:
                if corr >= threshold:
                    linked, reason = True, f"corr {corr:+.2f} ≥ {threshold:.2f}"
            else:
                sa, sb = sectors.get(a), sectors.get(b)
                if sa and sb and sa == sb:
                    linked, reason = True, f"same sector ({sa}); corr unavailable"
            if linked:
                union(a, b)
                edges.append(f"{a}↔{b}: {reason}")

    groups: Dict[str, List[str]] = {}
    for t in tickers:
        groups.setdefault(find(t), []).append(t)
    # Preserve input order for determinism.
    clusters = sorted(groups.values(), key=lambda g: tickers.index(g[0]))
    return clusters, edges


# ---------------------------------------------------------------------------
# Cap projection (per-position + per-cluster), water-filling the freed weight
# ---------------------------------------------------------------------------
def _project_caps(
    weights: Dict[str, float],
    clusters: List[List[str]],
    pos_cap: float,
    cluster_cap: float,
    max_iter: int = 200,
) -> Dict[str, float]:
    """Clamp weights so no single name exceeds `pos_cap` and no cluster's
    combined weight exceeds `cluster_cap`, redistributing the freed weight to
    names that still have headroom (in BOTH their position and their cluster).
    Weight that cannot be placed anywhere is dropped (it becomes extra cash).
    A final down-only pass guarantees the caps hold exactly in the output."""
    w = dict(weights)
    tickers = list(w)
    cl_of: Dict[str, int] = {t: ci for ci, cl in enumerate(clusters) for t in cl}

    def cluster_sum(ci: int) -> float:
        return sum(w[t] for t in clusters[ci])

    pool = 0.0
    for _ in range(max_iter):
        # 1. Clamp any single position over the cap; freed weight -> pool.
        for t in tickers:
            if w[t] > pos_cap:
                pool += w[t] - pos_cap
                w[t] = pos_cap
        # 2. Scale any over-cap cluster back down; freed weight -> pool.
        for ci, cl in enumerate(clusters):
            s = cluster_sum(ci)
            if s > cluster_cap and s > 0:
                scale = cluster_cap / s
                for t in cl:
                    pool += w[t] * (1.0 - scale)
                    w[t] *= scale
        if pool <= 1e-12:
            break

        # 3. Headroom per name = min(position room, its cluster's room).
        def headroom(t: str) -> float:
            hp = pos_cap - w[t]
            ci = cl_of.get(t)
            hc = math.inf if ci is None else (cluster_cap - cluster_sum(ci))
            return max(0.0, min(hp, hc))

        total_head = sum(headroom(t) for t in tickers)
        if total_head <= 1e-12:
            break  # nowhere to place the pool -> it stays as cash
        place = min(pool, total_head)
        # Distribute proportional to headroom: each name gets <= its own room,
        # so no cap is broken by a single pass (cluster interactions are cleaned
        # up on the next loop's clamp).
        for t in tickers:
            h = headroom(t)
            if h > 0:
                w[t] += place * (h / total_head)
        pool -= place

    # Final safety projection: guarantee caps hold regardless of convergence.
    for t in tickers:
        w[t] = min(w[t], pos_cap)
    for ci, cl in enumerate(clusters):
        s = cluster_sum(ci)
        if s > cluster_cap and s > 0:
            scale = cluster_cap / s
            for t in cl:
                w[t] *= scale
    return w


# ---------------------------------------------------------------------------
# Main entry: size the positions
# ---------------------------------------------------------------------------
def size_positions(
    picks: List[Pick],
    budget: float,
    risk_mode: str = DEFAULT_MODE,
    *,
    corr_matrix: Optional[Dict[str, Dict[str, float]]] = None,
    corr_threshold: float = DEFAULT_CORR_THRESHOLD,
    overrides: Optional[Dict[str, float]] = None,
) -> PositionSizing:
    """Allocate `budget` across `picks` (ranked best->worst). Returns a fully
    populated PositionSizing (rows, clusters, cash, and a step-by-step log of the
    weighting logic). All dollar/share math is done here in Python."""
    mode = (risk_mode or DEFAULT_MODE).strip().lower()
    if mode not in RISK_MODES:
        mode = DEFAULT_MODE
    params = dict(RISK_MODES[mode])
    if overrides:
        params.update({k: v for k, v in overrides.items() if v is not None})

    log: List[str] = []
    notes: List[str] = []
    budget = float(budget)

    # De-dupe while preserving rank order.
    seen: set = set()
    picks = [p for p in picks if not (p.ticker in seen or seen.add(p.ticker))]
    if not picks or budget <= 0:
        return PositionSizing(budget=budget, risk_mode=mode, params=params, rows=[],
                              clusters=[], cash=budget, invested=0.0,
                              log=["No picks or non-positive budget — nothing to size."],
                              notes=notes)

    tickers = [p.ticker for p in picks]
    kappa = params["kappa"]
    risk_aversion = params["risk_aversion"]
    cash_buffer = params["cash_buffer"]
    pos_cap = params["max_position"]
    cluster_cap = params["cluster_cap"]

    log.append(
        f"Risk mode = {mode}: concentration κ={kappa}, risk-aversion={risk_aversion}, "
        f"cash buffer={_pct(cash_buffer)}, single-name cap={_pct(pos_cap)}, "
        f"correlated-cluster cap={_pct(cluster_cap)}."
    )

    # --- Conviction: fill any missing scores with the median of the rest ---
    conv_present = [p.conviction for p in picks if p.conviction is not None]
    conv_median = _median(conv_present) if conv_present else 50.0
    convictions: Dict[str, float] = {}
    for p in picks:
        if p.conviction is None:
            convictions[p.ticker] = conv_median
            notes.append(f"{p.ticker}: no council conviction supplied — using basket "
                         f"median {conv_median:.0f}/100.")
        else:
            convictions[p.ticker] = float(p.conviction)

    # --- Risk composite per name (0.6*vol + 0.4*|drawdown|) ---
    vols_present = [p.volatility for p in picks if p.volatility and p.volatility > 0]
    dds_present = [abs(p.max_drawdown) for p in picks if p.max_drawdown]
    vol_fallback = _median(vols_present) if vols_present else _DEFAULT_VOL
    dd_fallback = _median(dds_present) if dds_present else _DEFAULT_DD
    risk: Dict[str, float] = {}
    for p in picks:
        vol = p.volatility if (p.volatility and p.volatility > 0) else vol_fallback
        dd = abs(p.max_drawdown) if p.max_drawdown else dd_fallback
        risk[p.ticker] = 0.6 * vol + 0.4 * dd
        if not (p.volatility and p.volatility > 0):
            notes.append(f"{p.ticker}: no volatility — using {vol*100:.0f}% (basket median/default).")
        if not p.max_drawdown:
            notes.append(f"{p.ticker}: no drawdown — using −{dd*100:.0f}% (basket median/default).")
    median_risk = _median(list(risk.values())) or 1.0

    # --- Base scores: conviction blended with inverse relative risk ---
    scores: Dict[str, float] = {}
    for t in tickers:
        conv_norm = max(_CONVICTION_FLOOR, convictions[t] / 100.0)
        rel_risk = risk[t] / median_risk if median_risk else 1.0
        inv_risk = (1.0 / rel_risk) ** risk_aversion if rel_risk > 0 else 1.0
        score = conv_norm * inv_risk
        scores[t] = score
        log.append(
            f"  {t}: conviction {convictions[t]:.0f}/100 (×{conv_norm:.2f}), "
            f"risk {risk[t]*100:.0f}% = {rel_risk:.2f}× basket median → inverse-risk "
            f"factor ×{inv_risk:.2f} ⇒ score {score:.3f}"
        )

    # --- Concentration (kappa) then normalize to the invested fraction ---
    raw = {t: scores[t] ** kappa for t in tickers}
    raw_sum = sum(raw.values()) or 1.0
    invest_target = 1.0 - cash_buffer
    weights = {t: raw[t] / raw_sum * invest_target for t in tickers}
    log.append(
        f"Applied concentration κ={kappa} and normalized to {_pct(invest_target)} "
        f"invested (pre-cap weights: " +
        ", ".join(f"{t} {weights[t]*100:.1f}%" for t in tickers) + ")."
    )

    # --- Clusters + cap projection ---
    sectors = {p.ticker: p.sector for p in picks}
    clusters, edges = build_clusters(tickers, corr_matrix, sectors, corr_threshold)
    multi = [c for c in clusters if len(c) > 1]
    if multi:
        for c in multi:
            log.append(f"Correlated cluster {{{', '.join(c)}}} — combined weight capped "
                       f"at {_pct(cluster_cap)}.")
        for e in edges:
            log.append(f"    linked {e}")
    else:
        log.append("No correlated clusters detected — only single-name caps apply.")

    pre_cap = dict(weights)
    weights = _project_caps(weights, clusters, pos_cap, cluster_cap)
    for t in tickers:
        if abs(weights[t] - pre_cap[t]) > 1e-6:
            direction = "trimmed" if weights[t] < pre_cap[t] else "raised"
            log.append(f"  {t}: {pre_cap[t]*100:.1f}% → {weights[t]*100:.1f}% "
                       f"({direction} by position/cluster caps).")

    # --- Dollars (fractional shares → deploy each name at exactly its weight) ---
    cl_of = {t: ci for ci, cl in enumerate(clusters) for t in cl}
    rows: List[SizedRow] = []
    deployed = 0.0
    for p in picks:
        t = p.ticker
        w = weights[t]
        dollars = w * budget            # exact target dollars; fractional shares allowed
        deployed += dollars
        ci = cl_of.get(t)
        cluster_id = ci if (ci is not None and len(clusters[ci]) > 1) else None
        rows.append(SizedRow(
            ticker=t, weight=w, dollars=dollars,
            conviction=convictions[t], risk=risk[t], cluster=cluster_id,
            rationale=_rationale(t, convictions[t], risk[t], cluster_id, clusters,
                                 pre_cap[t], w, pos_cap, cluster_cap),
        ))

    # Fractional shares mean the deployed dollars equal the computed weights
    # exactly, so the only money left over is the intended cash buffer (plus any
    # weight the hard caps couldn't place — reported explicitly below).
    cash = budget - deployed
    placed_frac = sum(r.weight for r in rows)
    dropped = max(0.0, invest_target - placed_frac)  # weight the caps couldn't seat
    log.append(
        f"Deployed {deployed/budget*100:.1f}% (${_fmt(deployed)}) at exact target weights; "
        f"holding ${_fmt(cash)} ({cash/budget*100:.1f}%) in cash "
        f"(buffer {_pct(cash_buffer)}"
        + (f" + {dropped*100:.1f}% the caps couldn't place)." if dropped > 1e-6 else ")."
        )
    )

    return PositionSizing(budget=budget, risk_mode=mode, params=params, rows=rows,
                          clusters=clusters, cash=cash, invested=deployed,
                          log=log, notes=notes)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def render_markdown(ps: PositionSizing) -> str:
    """Render the sizing result as a markdown section: the weighting logic, the
    allocation table, and the cash/notes summary."""
    lines: List[str] = []
    lines.append("## Position Sizing")
    lines.append("")
    lines.append(f"*Budget ${_fmt(ps.budget)} · risk mode: **{ps.risk_mode}** · "
                 f"computed in Python (not by the council).*")
    lines.append("")

    if not ps.rows:
        lines.append("_No positions to size._")
        return "\n".join(lines)

    lines.append("**Weighting logic**")
    for entry in ps.log:
        lines.append(f"- {entry}")
    lines.append("")

    lines.append("| Ticker | % Weight | Dollar Amount | Rationale |")
    lines.append("|---|---:|---:|---|")
    for r in ps.rows:
        lines.append(
            f"| {r.ticker} | {r.weight*100:.1f}% | ${_fmt(r.dollars)} | {r.rationale} |"
        )
    lines.append(f"| **Cash** | **{ps.cash/ps.budget*100:.1f}%** | **${_fmt(ps.cash)}** | "
                 f"Reserve buffer + any weight the caps couldn't place |")
    lines.append("")

    lines.append(f"*Deployed ${_fmt(ps.invested)} ({ps.invested/ps.budget*100:.1f}%) at exact "
                 f"target weights (fractional shares); ${_fmt(ps.cash)} "
                 f"({ps.cash/ps.budget*100:.1f}%) held in cash as the reserve buffer.*")
    if ps.notes:
        lines.append("")
        for n in ps.notes:
            lines.append(f"> {n}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def _median(xs: List[float]) -> Optional[float]:
    vals = sorted(v for v in xs if v is not None)
    if not vals:
        return None
    n = len(vals)
    mid = n // 2
    return vals[mid] if n % 2 else (vals[mid - 1] + vals[mid]) / 2.0


def _fmt(x: float) -> str:
    """Thousands-separated dollar string; whole dollars unless sub-$100."""
    if x is None:
        return "—"
    if abs(x) >= 100:
        return f"{x:,.0f}"
    return f"{x:,.2f}"


def _pct(frac: float, places: int = 1) -> str:
    """Percentage string with trailing zeros trimmed: 7.5% not 8%, 10% not 10.0%."""
    return f"{frac * 100:.{places}f}".rstrip("0").rstrip(".") + "%"


def _rationale(ticker, conviction, risk, cluster_id, clusters, pre_w, final_w,
               pos_cap, cluster_cap) -> str:
    conv_word = "high" if conviction >= 70 else ("moderate" if conviction >= 45 else "low")
    risk_word = "low" if risk < 0.45 else ("elevated" if risk < 0.70 else "high")
    parts = [f"{conv_word} conviction ({conviction:.0f}/100)",
             f"{risk_word} risk ({risk*100:.0f}%)"]
    if cluster_id is not None:
        peers = [t for t in clusters[cluster_id] if t != ticker]
        if peers:
            parts.append("correlated w/ " + ", ".join(peers))
    if final_w < pre_w - 1e-6:
        parts.append(f"trimmed to cap {_pct(min(pos_cap, cluster_cap))}")
    return "; ".join(parts)
