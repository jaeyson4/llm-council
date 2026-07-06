"""OpenRouter API client for making LLM requests."""

import json
import httpx
from typing import List, Dict, Any, Optional
from .config import OPENROUTER_API_KEY, OPENROUTER_API_URL, max_tokens_for


def _snippet(obj: Any, limit: int = 2000) -> str:
    """Compact, bounded string form of a JSON-ish object for logging, so a full
    error payload is visible without dumping tens of KB."""
    try:
        s = json.dumps(obj, ensure_ascii=False, default=str)
    except Exception:
        s = str(obj)
    return s if len(s) <= limit else s[:limit] + f"… [truncated, {len(s)} chars]"


async def query_model(
    model: str,
    messages: List[Dict[str, str]],
    timeout: float = 120.0,
    max_tokens: Optional[int] = None
) -> Optional[Dict[str, Any]]:
    """
    Query a single model via OpenRouter API.

    Args:
        model: OpenRouter model identifier (e.g., "openai/gpt-4o")
        messages: List of message dicts with 'role' and 'content'
        timeout: Request timeout in seconds
        max_tokens: Optional cap on output tokens. When None (default) no cap is
            sent, preserving the original behavior for existing callers.

    Returns:
        Response dict with 'content' and optional 'reasoning_details', or None if failed
    """
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "messages": messages,
    }
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                OPENROUTER_API_URL,
                headers=headers,
                json=payload
            )
            response.raise_for_status()

            data = response.json()

            # OpenRouter can return HTTP 200 with an error object in the body
            # (e.g. provider routing failures, content moderation). Surface the
            # full error so it doesn't masquerade as an empty response.
            if isinstance(data, dict) and data.get("error"):
                print(
                    f"[openrouter] model {model} returned an error body (HTTP 200):\n"
                    f"  error: {_snippet(data.get('error'))}"
                )
                return None

            choices = (data or {}).get("choices")
            if not choices:
                print(
                    f"[openrouter] model {model} returned NO choices — full payload:\n"
                    f"  {_snippet(data)}"
                )
                return None

            choice = choices[0] or {}
            message = choice.get("message") or {}
            content = message.get("content")

            # A model that "returns nothing" almost always comes back as HTTP 200
            # with empty/whitespace content. The usual culprit is a reasoning
            # model exhausting `max_tokens` on hidden reasoning
            # (finish_reason="length") — so print the finish reason and usage,
            # which make the cause self-explanatory, then treat it as a failure
            # (return None) so callers cleanly EXCLUDE it instead of silently
            # keeping a blank "successful" response.
            if content is None or (isinstance(content, str) and not content.strip()):
                print(
                    f"[openrouter] model {model} returned EMPTY content — likely the "
                    f"output token cap was consumed by reasoning, or the content was "
                    f"filtered. Diagnostics:\n"
                    f"  finish_reason: {choice.get('finish_reason')} "
                    f"(native: {choice.get('native_finish_reason')})\n"
                    f"  usage: {_snippet(data.get('usage'))}\n"
                    f"  message: {_snippet(message)}"
                )
                return None

            return {
                'content': content,
                'reasoning_details': message.get('reasoning_details')
            }

    except httpx.HTTPStatusError as e:
        # raise_for_status() only carries the status code in its message; the
        # actual reason (e.g. "not a valid model ID") lives in the response
        # body, so print the full body to make bad model IDs self-explanatory.
        print(
            f"[openrouter] error querying model {model}: HTTP {e.response.status_code} "
            f"from OpenRouter\n  response body: {e.response.text}"
        )
        return None
    except Exception as e:
        print(f"[openrouter] error querying model {model}: {type(e).__name__}: {e}")
        return None


async def query_models_parallel(
    models: List[str],
    messages: List[Dict[str, str]],
    max_tokens: Optional[int] = None
) -> Dict[str, Optional[Dict[str, Any]]]:
    """
    Query multiple models in parallel.

    Args:
        models: List of OpenRouter model identifiers
        messages: List of message dicts to send to each model
        max_tokens: Optional shared output token cap. When None (default), each
            model uses min(its real API max output, 32000) via
            config.max_tokens_for rather than a shared ceiling. Pass an int to
            force the same cap on all.

    Returns:
        Dict mapping model identifier to response dict (or None if failed)
    """
    import asyncio

    # Create tasks for all models. Resolve each model's min(real max, 32000) cap
    # when no shared cap is given.
    tasks = [
        query_model(
            model,
            messages,
            max_tokens=max_tokens if max_tokens is not None else max_tokens_for(model),
        )
        for model in models
    ]

    # Wait for all to complete
    responses = await asyncio.gather(*tasks)

    # Map models to their responses
    return {model: response for model, response in zip(models, responses)}
