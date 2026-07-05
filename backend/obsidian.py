"""Export a finished stock analysis as a markdown note into an Obsidian vault.

Each shortlisted ticker gets its own note, named `<TICKER>-<YYYY-MM-DD>.md` so
repeated analyses on different days never overwrite each other. The note opens
with YAML frontmatter (ticker, date, current_price, bear/base/bull targets, and a
one-line thesis) followed by the full analysis in readable markdown.

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
    return "\n".join(frontmatter) + "\n\n" + title + "\n\n" + (analysis_markdown or "").strip() + "\n"


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
