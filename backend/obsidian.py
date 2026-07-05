"""Export a finished stock analysis as a markdown note into an Obsidian vault.

Each shortlisted ticker gets its own note, named `<TICKER>-<YYYY-MM-DD>.md` so
repeated analyses on different days never overwrite each other. The note opens
with YAML frontmatter (ticker, date, current_price, bear/base/bull targets, and a
one-line thesis) followed by a block of Obsidian wikilinks + tags and then the
full analysis in readable markdown.

The wikilinks (`[[TICKER]]`, `[[Sector - <name>]]`, `[[Screening <date>]]`) plus
tags (`#stock`, `#sector/<slug>`) are what wire the notes together in Obsidian's
graph view: every dated analysis of a ticker points at the same `[[TICKER]]` hub,
every stock in a sector points at the same `[[Sector - …]]` hub, and every stock
screened in one run points at that run's `[[Screening <date>]]` note (see
`export_screening_note`) so co-screened names cluster together.

Fail safe: the vault folder is created if missing, and any I/O error is logged
and swallowed so a failed export never breaks the research flow or the UI.
"""

import os
import re
import traceback
from datetime import datetime
from typing import Any, Dict, Optional

from .config import OBSIDIAN_VAULT_PATH


def _yaml_escape(value: str) -> str:
    """Make a string safe as a single-line double-quoted YAML scalar."""
    return str(value).replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ").strip()


def _target(metrics: Optional[Dict[str, Any]], scenario: str) -> str:
    """Pull a scenario price target out of the computed metrics, or '' if absent."""
    if not metrics:
        return ""
    pt = metrics.get("price_targets") or {}
    entry = pt.get(scenario) or {}
    t = entry.get("target")
    return f"{t}" if t is not None else ""


def _current_price(metrics: Optional[Dict[str, Any]]) -> str:
    if not metrics:
        return ""
    price = (metrics.get("price") or {}).get("current")
    return f"{price}" if price is not None else ""


def _sector_slug(sector: str) -> str:
    """Turn a sector name into a tag-safe slug, e.g. 'Financial Services' ->
    'financial-services'. Returns '' when there's no usable sector so callers can
    skip the sector tag/link entirely."""
    return re.sub(r"[^a-z0-9]+", "-", (sector or "").strip().lower()).strip("-")


def _links_and_tags_block(ticker: str, sector: Optional[str], date: str) -> str:
    """Build the wikilinks + tags block that connects this note into the graph.

    Emits `[[TICKER]]`, `[[Sector - <name>]]` (when a sector is known), and
    `[[Screening <date>]]`, plus the `#stock` tag and a `#sector/<slug>` tag. The
    sector link/tag are omitted when the sector is unknown (e.g. ETFs) rather than
    linking to an empty `[[Sector - ]]` hub."""
    ticker = (ticker or "").strip().upper()
    sector = (sector or "").strip()

    parts = []
    if ticker:
        parts.append(f"**Stock:** [[{ticker}]]")
    if sector:
        parts.append(f"**Sector:** [[Sector - {sector}]]")
    parts.append(f"**Screening:** [[Screening {date}]]")
    links_line = " · ".join(parts)

    tags = ["#stock"]
    slug = _sector_slug(sector)
    if slug:
        tags.append(f"#sector/{slug}")
    tags_line = "**Tags:** " + " ".join(tags)

    return links_line + "\n\n" + tags_line


def build_note(
    ticker: str,
    analysis_markdown: str,
    metrics: Optional[Dict[str, Any]] = None,
    thesis: str = "",
    date: Optional[str] = None,
) -> str:
    """Assemble the full markdown note (frontmatter + body). Pure function; no I/O.
    Price targets and current price come from the Python-computed metrics, so the
    frontmatter is reliable even if the model's prose is not."""
    date = date or (metrics.get("as_of") if metrics else None) or datetime.now().strftime("%Y-%m-%d")
    ticker = (ticker or "").strip().upper()

    frontmatter = [
        "---",
        f'ticker: "{_yaml_escape(ticker)}"',
        f'date: "{_yaml_escape(date)}"',
        f'current_price: "{_yaml_escape(_current_price(metrics))}"',
        f'target_bear: "{_yaml_escape(_target(metrics, "bear"))}"',
        f'target_base: "{_yaml_escape(_target(metrics, "base"))}"',
        f'target_bull: "{_yaml_escape(_target(metrics, "bull"))}"',
        f'thesis: "{_yaml_escape(thesis)}"',
        "---",
    ]
    title = f"# {ticker}" + (f" — {metrics.get('name')}" if metrics and metrics.get("name") else "")
    links = _links_and_tags_block(ticker, (metrics or {}).get("sector"), date)
    body = (analysis_markdown or "").strip()
    return "\n".join(frontmatter) + "\n\n" + title + "\n\n" + links + "\n\n" + body + "\n"


def _safe_filename(ticker: str, date: str) -> str:
    """`<TICKER>-<date>.md`, with anything unusual stripped to keep it filesystem safe."""
    safe_ticker = re.sub(r"[^A-Za-z0-9._-]", "_", (ticker or "TICKER").strip().upper()) or "TICKER"
    safe_date = re.sub(r"[^0-9-]", "", (date or "").strip()) or datetime.now().strftime("%Y-%m-%d")
    return f"{safe_ticker}-{safe_date}.md"


def export_analysis_note(
    ticker: str,
    analysis_markdown: str,
    metrics: Optional[Dict[str, Any]] = None,
    thesis: str = "",
    date: Optional[str] = None,
    vault_path: Optional[str] = None,
) -> Optional[str]:
    """Write one analysis note into the vault. Returns the written path, or None
    if export is disabled (no vault path) or failed (logged, never raised)."""
    vault = vault_path if vault_path is not None else OBSIDIAN_VAULT_PATH
    if not vault:
        # No vault configured -> export is intentionally disabled. Say so out loud
        # so an empty OBSIDIAN_VAULT_PATH doesn't look like a silent failure.
        print("[obsidian] export skipped: no vault path configured "
              "(OBSIDIAN_VAULT_PATH is empty)")
        return None

    date = date or (metrics.get("as_of") if metrics else None) or datetime.now().strftime("%Y-%m-%d")
    try:
        os.makedirs(vault, exist_ok=True)  # create the folder if it doesn't exist
        path = os.path.join(vault, _safe_filename(ticker, date))
        note = build_note(ticker, analysis_markdown, metrics=metrics, thesis=thesis, date=date)
        print(f"WRITING OBSIDIAN NOTE TO: {path}")
        with open(path, "w", encoding="utf-8") as f:
            f.write(note)
        print(f"[obsidian] wrote note for {ticker} ({len(note)} chars) -> {path}")
        return path
    except Exception:  # never let a bad path / permission error break the flow
        # Print the FULL traceback (not just str(e)) so the real cause -- a bad
        # path, a read-only folder, a permissions error -- is visible in the
        # terminal instead of being swallowed.
        print(f"[obsidian] export FAILED for {ticker!r} at vault {vault!r}:")
        traceback.print_exc()
        return None


def _screening_filename(date: str) -> str:
    """`Screening <date>.md` — matches the `[[Screening <date>]]` link the
    per-ticker notes point at, so the graph resolves it to a real note."""
    safe_date = re.sub(r"[^0-9-]", "", (date or "").strip()) or datetime.now().strftime("%Y-%m-%d")
    return f"Screening {safe_date}.md"


def build_screening_note(date: str, tickers) -> str:
    """Assemble a fresh screening note: frontmatter + a `[[TICKER]]` per line so
    every stock screened in this run links back here (and thus to each other via
    this shared hub). Pure function; no I/O."""
    clean = [(t or "").strip().upper() for t in tickers if (t or "").strip()]
    frontmatter = [
        "---",
        f'date: "{_yaml_escape(date)}"',
        'type: "screening"',
        "---",
    ]
    header = f"# Screening {date}"
    intro = "Stocks screened in this run (each links to its full analysis note):"
    bullets = "\n".join(f"- [[{t}]]" for t in clean) if clean else "_No tickers._"
    return "\n".join(frontmatter) + f"\n\n{header}\n\n#screening\n\n{intro}\n\n{bullets}\n"


def export_screening_note(
    tickers,
    date: Optional[str] = None,
    vault_path: Optional[str] = None,
) -> Optional[str]:
    """Write (or append to) the `Screening <date>` note listing every ticker
    screened in this run as a wikilink. If the note already exists (e.g. a second
    run on the same day), only genuinely new tickers are appended — re-running is
    idempotent and never duplicates a link. Returns the path, or None if export is
    disabled or failed (logged, never raised)."""
    vault = vault_path if vault_path is not None else OBSIDIAN_VAULT_PATH
    if not vault:
        print("[obsidian] screening note skipped: no vault path configured "
              "(OBSIDIAN_VAULT_PATH is empty)")
        return None

    date = date or datetime.now().strftime("%Y-%m-%d")
    # De-dupe within this run, preserving the shortlist order.
    seen = set()
    tickers = [
        t for t in ((x or "").strip().upper() for x in tickers)
        if t and not (t in seen or seen.add(t))
    ]

    try:
        os.makedirs(vault, exist_ok=True)
        path = os.path.join(vault, _screening_filename(date))
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                existing = f.read()
            # Which tickers are already linked? Match plain `[[X]]` links only
            # (exclude aliases/heading links) so we never re-add an existing one.
            have = {
                m.group(1).strip().upper()
                for m in re.finditer(r"\[\[([^\]|#]+?)\]\]", existing)
            }
            new = [t for t in tickers if t not in have]
            if not new:
                print(f"[obsidian] screening note already current -> {path}")
                return path
            # Append bullets contiguously with the existing list: the file already
            # ends in a newline, so the new bullets need no leading blank line.
            addition = "\n".join(f"- [[{t}]]" for t in new) + "\n"
            sep = "" if existing.endswith("\n") else "\n"
            with open(path, "a", encoding="utf-8") as f:
                f.write(sep + addition)
            print(f"[obsidian] appended {len(new)} ticker(s) to screening note -> {path}")
            return path

        note = build_screening_note(date, tickers)
        with open(path, "w", encoding="utf-8") as f:
            f.write(note)
        print(f"[obsidian] wrote screening note for {len(tickers)} ticker(s) -> {path}")
        return path
    except Exception:
        print(f"[obsidian] screening note export FAILED for {date!r} at vault {vault!r}:")
        traceback.print_exc()
        return None
