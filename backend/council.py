"""3-stage LLM Council orchestration."""

import re
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional
from .openrouter import query_models_parallel, query_model
from .config import (
    COUNCIL_MODELS,
    CHAIRMAN_MODEL,
    SCREENING_MODEL,
    SHORTLIST_SIZE,
    SCREENING_MAX_TOKENS,
    DEEPDIVE_BASE_TOKENS,
    DEEPDIVE_TOKENS_PER_TICKER,
    DEEPDIVE_MAX_TOKENS,
)
from . import metrics as metrics_mod
from . import obsidian


def compute_deepdive_cap(n_tickers: int) -> int:
    """Per-call output-token budget for Stage B, scaled to the shortlist size so
    each ticker's structured analysis has room, capped by a hard ceiling."""
    n = max(1, int(n_tickers or 1))
    return min(DEEPDIVE_MAX_TOKENS, DEEPDIVE_BASE_TOKENS + DEEPDIVE_TOKENS_PER_TICKER * n)


# The structure every deep-dive analysis must follow, for every ticker covered.
OUTPUT_STRUCTURE = """REQUIRED OUTPUT STRUCTURE — for EACH ticker you cover, use exactly these six labeled sections, in this order:
1. **Macro/sector context** — the top-down backdrop (rates, cycle, sector dynamics) relevant to this name.
2. **Bull thesis** — the strongest case for owning it.
3. **Bear thesis** — the strongest case against it.
4. **Key numbers + interpretation** — reference the Python-computed figures provided above (valuation, growth, margins, free cash flow, valuation percentile vs its own history, max drawdown, forward base rates) and explain what they imply. Do NOT invent or alter these numbers.
5. **2-year price targets (base / bull / bear)** — give a 2-year target for each scenario with the key assumptions (growth, exit multiple) behind it. You may adopt or adjust the provided Python scenario targets, but justify any change.
6. **Thesis-breakers** — the specific, observable events or data points that would prove your thesis wrong."""


def _prepend_context(market_context: str, prompt: str) -> str:
    """Prepend the live market-data block to a prompt, if any data was fetched.

    market_context is an empty string when the question has no recognizable
    tickers (or data couldn't be fetched), in which case the prompt is returned
    unchanged — keeping the original flow intact.
    """
    if market_context:
        return f"{market_context}\n\n---\n\n{prompt}"
    return prompt


async def stage1_collect_responses(
    user_query: str,
    market_context: str = "",
    max_tokens: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Stage 1: Collect individual responses from all council models.

    Args:
        user_query: The user's question (or Stage B deep-dive task prompt)
        market_context: Optional live stock-data block prepended to the prompt
        max_tokens: Optional output token cap per model

    Returns:
        List of dicts with 'model' and 'response' keys
    """
    messages = [{"role": "user", "content": _prepend_context(market_context, user_query)}]

    # Query all models in parallel
    responses = await query_models_parallel(COUNCIL_MODELS, messages, max_tokens=max_tokens)

    # Format results
    stage1_results = []
    for model, response in responses.items():
        if response is not None:  # Only include successful responses
            stage1_results.append({
                "model": model,
                "response": response.get('content', '')
            })

    return stage1_results


async def stage2_collect_rankings(
    user_query: str,
    stage1_results: List[Dict[str, Any]],
    market_context: str = "",
    max_tokens: Optional[int] = None
) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    """
    Stage 2: Each model ranks the anonymized responses.

    Args:
        user_query: The original user query
        stage1_results: Results from Stage 1
        market_context: Optional live stock-data block prepended to the prompt
        max_tokens: Optional output token cap per model

    Returns:
        Tuple of (rankings list, label_to_model mapping)
    """
    # Create anonymized labels for responses (Response A, Response B, etc.)
    labels = [chr(65 + i) for i in range(len(stage1_results))]  # A, B, C, ...

    # Create mapping from label to model name
    label_to_model = {
        f"Response {label}": result['model']
        for label, result in zip(labels, stage1_results)
    }

    # Build the ranking prompt
    responses_text = "\n\n".join([
        f"Response {label}:\n{result['response']}"
        for label, result in zip(labels, stage1_results)
    ])

    ranking_prompt = f"""You are evaluating different responses to the following question:

Question: {user_query}

Here are the responses from different models (anonymized):

{responses_text}

Your task:
1. First, evaluate each response individually. For each response, explain what it does well and what it does poorly.
2. Then, at the very end of your response, provide a final ranking.

IMPORTANT: Your final ranking MUST be formatted EXACTLY as follows:
- Start with the line "FINAL RANKING:" (all caps, with colon)
- Then list the responses from best to worst as a numbered list
- Each line should be: number, period, space, then ONLY the response label (e.g., "1. Response A")
- Do not add any other text or explanations in the ranking section

Example of the correct format for your ENTIRE response:

Response A provides good detail on X but misses Y...
Response B is accurate but lacks depth on Z...
Response C offers the most comprehensive answer...

FINAL RANKING:
1. Response C
2. Response A
3. Response B

Now provide your evaluation and ranking:"""

    messages = [{"role": "user", "content": _prepend_context(market_context, ranking_prompt)}]

    # Get rankings from all council models in parallel
    responses = await query_models_parallel(COUNCIL_MODELS, messages, max_tokens=max_tokens)

    # Format results
    stage2_results = []
    for model, response in responses.items():
        if response is not None:
            full_text = response.get('content', '')
            parsed = parse_ranking_from_text(full_text)
            stage2_results.append({
                "model": model,
                "ranking": full_text,
                "parsed_ranking": parsed
            })

    return stage2_results, label_to_model


async def stage3_synthesize_final(
    user_query: str,
    stage1_results: List[Dict[str, Any]],
    stage2_results: List[Dict[str, Any]],
    market_context: str = "",
    max_tokens: Optional[int] = None,
    shortlist_tickers: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Stage 3: Chairman synthesizes final response.

    Args:
        user_query: The original user query
        stage1_results: Individual model responses from Stage 1
        stage2_results: Rankings from Stage 2
        market_context: Optional live stock-data block prepended to the prompt
        max_tokens: Optional output token cap for the chairman
        shortlist_tickers: When provided, the chairman is told to emit one
            level-2 section per ticker headed exactly "## <TICKER>", so the
            report can be split into per-ticker Obsidian notes.

    Returns:
        Dict with 'model' and 'response' keys
    """
    # Build comprehensive context for chairman
    stage1_text = "\n\n".join([
        f"Model: {result['model']}\nResponse: {result['response']}"
        for result in stage1_results
    ])

    stage2_text = "\n\n".join([
        f"Model: {result['model']}\nRanking: {result['ranking']}"
        for result in stage2_results
    ])

    chairman_prompt = f"""You are the Chairman of an LLM Council. Multiple AI models have provided responses to a user's question, and then ranked each other's responses.

Original Question: {user_query}

STAGE 1 - Individual Responses:
{stage1_text}

STAGE 2 - Peer Rankings:
{stage2_text}

Your task as Chairman is to synthesize all of this information into a single, comprehensive, accurate answer to the user's original question. Consider:
- The individual responses and their insights
- The peer rankings and what they reveal about response quality
- Any patterns of agreement or disagreement

Provide a clear, well-reasoned final answer that represents the council's collective wisdom:"""

    # In stock-research mode, require a per-ticker structure so the final report
    # can be split cleanly into one Obsidian note per ticker.
    if shortlist_tickers:
        tickers_str = ", ".join(shortlist_tickers)
        chairman_prompt += f"""

This is a stock research report on: {tickers_str}. Format your report with ONE section per ticker. Begin each ticker's section with a level-2 markdown header that is EXACTLY the ticker symbol and nothing else, e.g. "## {shortlist_tickers[0]}". Within each ticker's section, follow this structure:

{OUTPUT_STRUCTURE}

Base every figure on the Python-computed data provided above; do not invent numbers."""

    messages = [{"role": "user", "content": _prepend_context(market_context, chairman_prompt)}]

    # Query the chairman model
    response = await query_model(CHAIRMAN_MODEL, messages, max_tokens=max_tokens)

    if response is None:
        # Fallback if chairman fails
        return {
            "model": CHAIRMAN_MODEL,
            "response": "Error: Unable to generate final synthesis."
        }

    return {
        "model": CHAIRMAN_MODEL,
        "response": response.get('content', '')
    }


def parse_ranking_from_text(ranking_text: str) -> List[str]:
    """
    Parse the FINAL RANKING section from the model's response.

    Args:
        ranking_text: The full text response from the model

    Returns:
        List of response labels in ranked order
    """
    import re

    # Look for "FINAL RANKING:" section
    if "FINAL RANKING:" in ranking_text:
        # Extract everything after "FINAL RANKING:"
        parts = ranking_text.split("FINAL RANKING:")
        if len(parts) >= 2:
            ranking_section = parts[1]
            # Try to extract numbered list format (e.g., "1. Response A")
            # This pattern looks for: number, period, optional space, "Response X"
            numbered_matches = re.findall(r'\d+\.\s*Response [A-Z]', ranking_section)
            if numbered_matches:
                # Extract just the "Response X" part
                return [re.search(r'Response [A-Z]', m).group() for m in numbered_matches]

            # Fallback: Extract all "Response X" patterns in order
            matches = re.findall(r'Response [A-Z]', ranking_section)
            return matches

    # Fallback: try to find any "Response X" patterns in order
    matches = re.findall(r'Response [A-Z]', ranking_text)
    return matches


def calculate_aggregate_rankings(
    stage2_results: List[Dict[str, Any]],
    label_to_model: Dict[str, str]
) -> List[Dict[str, Any]]:
    """
    Calculate aggregate rankings across all models.

    Args:
        stage2_results: Rankings from each model
        label_to_model: Mapping from anonymous labels to model names

    Returns:
        List of dicts with model name and average rank, sorted best to worst
    """
    from collections import defaultdict

    # Track positions for each model
    model_positions = defaultdict(list)

    for ranking in stage2_results:
        ranking_text = ranking['ranking']

        # Parse the ranking from the structured format
        parsed_ranking = parse_ranking_from_text(ranking_text)

        for position, label in enumerate(parsed_ranking, start=1):
            if label in label_to_model:
                model_name = label_to_model[label]
                model_positions[model_name].append(position)

    # Calculate average position for each model
    aggregate = []
    for model, positions in model_positions.items():
        if positions:
            avg_rank = sum(positions) / len(positions)
            aggregate.append({
                "model": model,
                "average_rank": round(avg_rank, 2),
                "rankings_count": len(positions)
            })

    # Sort by average rank (lower is better)
    aggregate.sort(key=lambda x: x['average_rank'])

    return aggregate


async def generate_conversation_title(user_query: str) -> str:
    """
    Generate a short title for a conversation based on the first user message.

    Args:
        user_query: The first user message

    Returns:
        A short title (3-5 words)
    """
    title_prompt = f"""Generate a very short title (3-5 words maximum) that summarizes the following question.
The title should be concise and descriptive. Do not use quotes or punctuation in the title.

Question: {user_query}

Title:"""

    messages = [{"role": "user", "content": title_prompt}]

    # Use gemini-2.5-flash for title generation (fast and cheap)
    response = await query_model("google/gemini-2.5-flash", messages, timeout=30.0)

    if response is None:
        # Fallback to a generic title
        return "New Conversation"

    title = response.get('content', 'New Conversation').strip()

    # Clean up the title - remove quotes, limit length
    title = title.strip('"\'')

    # Truncate if too long
    if len(title) > 50:
        title = title[:47] + "..."

    return title


# ===========================================================================
# Stage A — Screening (one cheap model, no peer review)
# ===========================================================================
async def stage_a_screening(user_query: str) -> Dict[str, Any]:
    """
    Stage A: a single cheap model does a top-down (macro -> sectors -> candidates)
    pass and returns a shortlist of ~SHORTLIST_SIZE tickers, each with a one-line
    thesis. No peer review. Premium council models are NOT used here.

    Returns a dict: {'model', 'response' (raw text), 'shortlist': [{'ticker','thesis'}]}.
    """
    prompt = f"""You are a top-down equity screener. Given the user's request, reason briefly from the top down: macro backdrop -> attractive sectors -> specific candidate stocks.

User request: {user_query}

Then output a shortlist of the {SHORTLIST_SIZE} most promising, liquid, US-listed stocks to research in depth. If the user named specific tickers, include and prioritize them.

Keep any reasoning to a few sentences. Then end with the shortlist in EXACTLY this format (and nothing after it):

SHORTLIST:
1. TICKER — one-line thesis
2. TICKER — one-line thesis
(up to {SHORTLIST_SIZE} lines)

Use real, valid ticker symbols (e.g. NVDA, AAPL). One ticker per line."""

    messages = [{"role": "user", "content": prompt}]
    response = await query_model(SCREENING_MODEL, messages, max_tokens=SCREENING_MAX_TOKENS)

    raw = response.get("content", "") if response else ""
    shortlist = parse_shortlist(raw)

    # This is the hinge of the whole live-data flow: if the screening model
    # returns nothing, or its output can't be parsed into tickers, the shortlist
    # is empty and NO yfinance data gets fetched — the deep dive then answers
    # from training data only. Make that outcome loud rather than silent.
    if not response:
        print(f"[screening] model {SCREENING_MODEL!r} returned nothing (see the "
              f"[openrouter] diagnostic above) -> empty shortlist -> NO live market "
              f"data will be fetched.")
    elif not shortlist:
        print(f"[screening] model {SCREENING_MODEL!r} responded but no tickers could "
              f"be parsed from its output -> NO live market data will be fetched. "
              f"Raw screening output was:\n{raw}")
    else:
        print(f"[screening] {SCREENING_MODEL} proposed {len(shortlist)} ticker(s): "
              f"{[s['ticker'] for s in shortlist]}")

    return {"model": SCREENING_MODEL, "response": raw, "shortlist": shortlist}


def parse_shortlist(text: str) -> List[Dict[str, str]]:
    """Parse the 'SHORTLIST:' section into [{'ticker','thesis'}]. Falls back to
    scanning the whole text for 'N. TICKER — thesis' lines if the header is
    missing. De-dupes and caps at SHORTLIST_SIZE."""
    if not text:
        return []

    section = text
    if "SHORTLIST:" in text:
        section = text.split("SHORTLIST:", 1)[1]

    results: List[Dict[str, str]] = []
    seen = set()
    # Lines like: "1. NVDA — thesis" / "1. NVDA - thesis" / "- NVDA: thesis"
    line_re = re.compile(
        r'^\s*(?:\d+[.)]|[-*])?\s*\$?([A-Z][A-Z.\-]{0,5})\b\s*[—:\-–]\s*(.+?)\s*$'
    )
    for line in section.splitlines():
        m = line_re.match(line)
        if not m:
            continue
        ticker = m.group(1).upper().strip(".-")
        thesis = m.group(2).strip()
        if not ticker or ticker in seen:
            continue
        seen.add(ticker)
        results.append({"ticker": ticker, "thesis": thesis})
        if len(results) >= SHORTLIST_SIZE:
            break
    return results


# ===========================================================================
# Stage B — Deep dive preparation (Python numbers + prompt)
# ===========================================================================
async def prepare_deepdive(shortlist: List[Dict[str, str]]) -> Dict[str, Any]:
    """Fetch Python-computed metrics for every shortlisted ticker (cached, one
    fetch per ticker) and format them into an injectable context block."""
    tickers = [item["ticker"] for item in shortlist]
    metrics_by_ticker = await metrics_mod.get_many_metrics(tickers)
    context = metrics_mod.format_many_for_prompt(metrics_by_ticker)
    return {"metrics": metrics_by_ticker, "context": context}


def build_deepdive_query(
    user_query: str,
    screening: Dict[str, Any],
    shortlist: List[Dict[str, str]]
) -> str:
    """Compose the Stage B task prompt: the user's goal + screening context +
    shortlist + the required output structure. The Python numbers are injected
    separately (as the prepended market-data block)."""
    shortlist_lines = "\n".join(
        f"- {item['ticker']}: {item['thesis']}" for item in shortlist
    )
    screening_text = (screening or {}).get("response", "").strip()
    return f"""You are a member of an equity research council conducting a DEEP DIVE on a pre-screened shortlist of stocks.

Original user request: {user_query}

Screening (macro -> sector -> candidates) context from the first-pass analyst:
{screening_text}

Shortlisted tickers to analyze:
{shortlist_lines}

Analyze EVERY shortlisted ticker above. The Python-computed figures for each are provided in the data block above — cite them and interpret them; never invent numbers.

{OUTPUT_STRUCTURE}

Be concise and decisive. Cover all shortlisted tickers."""


# ===========================================================================
# Obsidian export — split the final report into one note per ticker
# ===========================================================================
def _split_report_by_ticker(
    report_text: str,
    tickers: List[str],
    names_by_ticker: Optional[Dict[str, str]] = None,
) -> Dict[str, str]:
    """Split a markdown report into {ticker: section_text}.

    A header line only starts a ticker's section if it is *ticker-shaped* — i.e.
    its FIRST word is that ticker's symbol or the first word of its company name
    ('## NVDA', '## NVDA — NVIDIA', '## NVIDIA (NVDA)', '## NVIDIA Corporation').
    This deliberately does NOT fire on:
      - a document title that names several tickers ('# Report: NVDA, AMD, AVGO'),
      - a sub-header that merely mentions a competitor ('### Competition from AMD'),
      - an incidental capital letter for single-letter tickers ('### Section V').
    When a ticker legitimately recurs, the LONGEST block is kept.
    """
    if not report_text or not tickers:
        return {}
    names_by_ticker = names_by_ticker or {}
    upper_tickers = [t.upper() for t in tickers]

    # First word of each ticker's company name, for name-only headers.
    name_first: Dict[str, str] = {}
    for t in upper_tickers:
        nm = (names_by_ticker.get(t) or "").strip()
        if nm:
            fw = re.split(r"\W+", nm.upper())[0]
            if fw:
                name_first[t] = fw

    lines = report_text.splitlines()
    header_re = re.compile(r"^#{1,4}\s+(.*)$")
    boundaries: List[tuple] = []  # (line_index, ticker)
    for i, line in enumerate(lines):
        hm = header_re.match(line)
        if not hm:
            continue
        header_text = hm.group(1).strip()
        upper = header_text.upper()

        # Skip headers that name 2+ distinct shortlisted tickers (title/overview).
        distinct = {
            t for t in upper_tickers
            if re.search(rf"(?<![A-Z0-9]){re.escape(t)}(?![A-Z0-9])", upper)
        }
        if len(distinct) >= 2:
            continue

        # The header must START with a ticker symbol or its company's first word.
        first_tok_m = re.search(r"[A-Za-z0-9.\-]+", header_text)
        if not first_tok_m:
            continue
        first_tok = first_tok_m.group(0).upper().strip(".-")
        for t in upper_tickers:
            if first_tok == t or (t in name_first and first_tok == name_first[t]):
                boundaries.append((i, t))
                break

    sections: Dict[str, str] = {}
    for idx, (line_i, ticker) in enumerate(boundaries):
        end = boundaries[idx + 1][0] if idx + 1 < len(boundaries) else len(lines)
        body = "\n".join(lines[line_i:end]).strip()
        # Keep the longest block if a ticker's header appears more than once.
        if ticker not in sections or len(body) > len(sections[ticker]):
            sections[ticker] = body
    return sections


def export_notes_from_report(
    report_text: str,
    shortlist: List[Dict[str, str]],
    metrics_by_ticker: Dict[str, Dict[str, Any]]
) -> List[Dict[str, str]]:
    """Write one Obsidian note per shortlisted ticker. The note body is that
    ticker's section of the chairman's report (falling back to its computed
    figures if no section is found). Frontmatter numbers come from Python.
    Returns [{'ticker','path'}] for notes actually written."""
    tickers = [item["ticker"] for item in shortlist]
    print(f"[obsidian] export_notes_from_report: {len(tickers)} ticker(s) to write -> {tickers}")
    thesis_by_ticker = {item["ticker"]: item.get("thesis", "") for item in shortlist}
    names_by_ticker = {
        t: (metrics_by_ticker.get(t) or {}).get("name", "") for t in tickers
    }
    sections = _split_report_by_ticker(report_text, tickers, names_by_ticker)

    # One run-level date shared by every note and the screening hub, so each
    # note's `[[Screening <date>]]` link resolves to the note we write below —
    # even if two tickers' metrics were computed on different days.
    run_date = next(
        ((metrics_by_ticker.get(t) or {}).get("as_of")
         for t in tickers if (metrics_by_ticker.get(t) or {}).get("as_of")),
        None,
    ) or datetime.now().strftime("%Y-%m-%d")

    written: List[Dict[str, str]] = []
    for ticker in tickers:
        metrics = metrics_by_ticker.get(ticker)
        body = sections.get(ticker)
        if not body:
            # No dedicated section parsed — still create a note so the ticker
            # isn't silently dropped, using its computed figures as the body.
            figures = metrics_mod.format_metrics_for_prompt(metrics) if metrics else ""
            body = (
                "> Per-ticker section could not be extracted from the council "
                "report; see the app for the full combined analysis.\n\n" + figures
            )
        path = obsidian.export_analysis_note(
            ticker=ticker,
            analysis_markdown=body,
            metrics=metrics,
            thesis=thesis_by_ticker.get(ticker, ""),
            date=run_date,
        )
        if path:
            written.append({"ticker": ticker, "path": path})
    print(f"[obsidian] export complete: wrote {len(written)} of {len(tickers)} note(s)")

    # The co-screening hub note: lists every screened ticker as a wikilink so
    # this run's stocks all connect through it in the graph view.
    obsidian.export_screening_note(tickers, date=run_date)
    return written


# ===========================================================================
# Orchestration
# ===========================================================================
async def run_full_council(user_query: str) -> Tuple[List, List, Dict, Dict]:
    """
    Run the full two-stage stock-research flow:
      Stage A  — one cheap model screens -> shortlist
      Stage B  — the full council deep-dives the shortlist with Python numbers
                 injected, peer-reviews, and the chairman synthesizes a report
                 which is exported per-ticker to Obsidian.

    Returns (stage1_results, stage2_results, stage3_result, metadata). The
    stage1/2/3 shapes are unchanged, so the existing UI and storage keep working.
    """
    # --- Stage A: screening ---
    screening = await stage_a_screening(user_query)
    shortlist = screening.get("shortlist", [])

    # --- Prepare deep-dive numbers (Python, cached per ticker) ---
    prepared = await prepare_deepdive(shortlist) if shortlist else {"metrics": {}, "context": ""}
    deepdive_context = prepared["context"]
    metrics_by_ticker = prepared["metrics"]

    # Confirm (or warn) whether real market data actually made it into the run.
    if deepdive_context:
        print(f"[deepdive] injecting live yfinance data for {len(metrics_by_ticker)} "
              f"of {len(shortlist)} shortlisted ticker(s): {list(metrics_by_ticker)}")
    else:
        print("[deepdive] WARNING: no live market data available — the council will "
              "answer from training data only. (Empty shortlist, or yfinance "
              "returned nothing for every proposed ticker.)")

    # The Stage B task prompt (falls back to the raw user query if screening
    # produced no shortlist, so the app still answers something).
    deepdive_query = build_deepdive_query(user_query, screening, shortlist) if shortlist else user_query
    cap = compute_deepdive_cap(len(shortlist))

    # --- Stage B, step 1: council deep dive ---
    stage1_results = await stage1_collect_responses(
        deepdive_query, deepdive_context, max_tokens=cap
    )
    if not stage1_results:
        return [], [], {
            "model": "error",
            "response": "All models failed to respond. Please try again."
        }, {
            "label_to_model": {},
            "aggregate_rankings": [],
            "screening": screening,
            "shortlist": shortlist,
            "metrics": metrics_by_ticker,
            "exported_notes": [],
        }

    # --- Stage B, step 2: peer rankings ---
    stage2_results, label_to_model = await stage2_collect_rankings(
        deepdive_query, stage1_results, deepdive_context, max_tokens=cap
    )
    aggregate_rankings = calculate_aggregate_rankings(stage2_results, label_to_model)

    # --- Stage B, step 3: chairman synthesis (per-ticker sections) ---
    stage3_result = await stage3_synthesize_final(
        deepdive_query, stage1_results, stage2_results, deepdive_context,
        max_tokens=cap,
        shortlist_tickers=[item["ticker"] for item in shortlist],
    )

    # --- Export each ticker's analysis to Obsidian (best-effort) ---
    exported = []
    if shortlist:
        exported = export_notes_from_report(
            stage3_result.get("response", ""), shortlist, metrics_by_ticker
        )
    else:
        print("[obsidian] export skipped: screening produced an empty shortlist, "
              "so there are no tickers to write notes for")

    metadata = {
        "label_to_model": label_to_model,
        "aggregate_rankings": aggregate_rankings,
        "screening": screening,
        "shortlist": shortlist,
        "metrics": metrics_by_ticker,
        "exported_notes": exported,
    }

    return stage1_results, stage2_results, stage3_result, metadata
