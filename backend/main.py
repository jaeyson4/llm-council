"""FastAPI backend for LLM Council."""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Dict, Any
import uuid
import json
import asyncio

from . import storage
from .council import (
    run_full_council,
    generate_conversation_title,
    stage1_collect_responses,
    stage2_collect_rankings,
    stage3_synthesize_final,
    calculate_aggregate_rankings,
    stage_a_screening,
    prepare_deepdive,
    build_deepdive_query,
    export_notes_from_report,
    compute_deepdive_cap,
    parse_budget,
    run_position_sizing,
    strip_conviction_block,
)

app = FastAPI(title="LLM Council API")

# Enable CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class CreateConversationRequest(BaseModel):
    """Request to create a new conversation."""
    pass


class SendMessageRequest(BaseModel):
    """Request to send a message in a conversation."""
    content: str


class ConversationMetadata(BaseModel):
    """Conversation metadata for list view."""
    id: str
    created_at: str
    title: str
    message_count: int


class Conversation(BaseModel):
    """Full conversation with all messages."""
    id: str
    created_at: str
    title: str
    messages: List[Dict[str, Any]]


@app.get("/")
async def root():
    """Health check endpoint."""
    return {"status": "ok", "service": "LLM Council API"}


@app.get("/api/conversations", response_model=List[ConversationMetadata])
async def list_conversations():
    """List all conversations (metadata only)."""
    return storage.list_conversations()


@app.post("/api/conversations", response_model=Conversation)
async def create_conversation(request: CreateConversationRequest):
    """Create a new conversation."""
    conversation_id = str(uuid.uuid4())
    conversation = storage.create_conversation(conversation_id)
    return conversation


@app.get("/api/conversations/{conversation_id}", response_model=Conversation)
async def get_conversation(conversation_id: str):
    """Get a specific conversation with all its messages."""
    conversation = storage.get_conversation(conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conversation


@app.delete("/api/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str):
    """Permanently delete a conversation and its stored messages."""
    deleted = storage.delete_conversation(conversation_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {"status": "deleted", "id": conversation_id}


@app.post("/api/conversations/{conversation_id}/message")
async def send_message(conversation_id: str, request: SendMessageRequest):
    """
    Send a message and run the 3-stage council process.
    Returns the complete response with all stages.
    """
    # Check if conversation exists
    conversation = storage.get_conversation(conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found")

    # Check if this is the first message
    is_first_message = len(conversation["messages"]) == 0

    # Add user message
    storage.add_user_message(conversation_id, request.content)

    # If this is the first message, generate a title
    if is_first_message:
        title = await generate_conversation_title(request.content)
        storage.update_conversation_title(conversation_id, title)

    # Run the 3-stage council process
    stage1_results, stage2_results, stage3_result, metadata = await run_full_council(
        request.content
    )

    # Add assistant message with all stages
    storage.add_assistant_message(
        conversation_id,
        stage1_results,
        stage2_results,
        stage3_result
    )

    # Return the complete response with metadata
    return {
        "stage1": stage1_results,
        "stage2": stage2_results,
        "stage3": stage3_result,
        "metadata": metadata
    }


@app.post("/api/conversations/{conversation_id}/message/stream")
async def send_message_stream(conversation_id: str, request: SendMessageRequest):
    """
    Send a message and stream the 3-stage council process.
    Returns Server-Sent Events as each stage completes.
    """
    # Check if conversation exists
    conversation = storage.get_conversation(conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found")

    # Check if this is the first message
    is_first_message = len(conversation["messages"]) == 0

    async def event_generator():
        try:
            # Add user message
            storage.add_user_message(conversation_id, request.content)

            # Start title generation in parallel (don't await yet)
            title_task = None
            if is_first_message:
                title_task = asyncio.create_task(generate_conversation_title(request.content))

            # --- Stage A: screening (one cheap model, no peer review) ---
            yield f"data: {json.dumps({'type': 'screening_start'})}\n\n"
            screening = await stage_a_screening(request.content)
            shortlist = screening.get("shortlist", [])
            yield f"data: {json.dumps({'type': 'screening_complete', 'data': {'shortlist': shortlist, 'response': screening.get('response', '')}})}\n\n"

            # --- Prepare deep-dive numbers (Python-computed, cached per ticker) ---
            prepared = await prepare_deepdive(shortlist) if shortlist else {"metrics": {}, "context": ""}
            deepdive_context = prepared["context"]
            metrics_by_ticker = prepared["metrics"]
            if deepdive_context:
                print(f"[deepdive] injecting live yfinance data for {len(metrics_by_ticker)} "
                      f"of {len(shortlist)} shortlisted ticker(s): {list(metrics_by_ticker)}")
                yield f"data: {json.dumps({'type': 'market_data', 'data': deepdive_context})}\n\n"
            else:
                print("[deepdive] WARNING: no live market data available — the council will "
                      "answer from training data only. (Empty shortlist, or yfinance "
                      "returned nothing for every proposed ticker.)")

            # Stage B task prompt (fall back to the raw query if nothing was shortlisted)
            deepdive_query = build_deepdive_query(request.content, screening, shortlist) if shortlist else request.content
            cap = compute_deepdive_cap(len(shortlist))

            # --- Stage B, step 1: council deep dive ---
            yield f"data: {json.dumps({'type': 'stage1_start'})}\n\n"
            stage1_results = await stage1_collect_responses(deepdive_query, deepdive_context, max_tokens=cap)

            # If every council model failed, stop here rather than fabricating a
            # report from zero input and writing junk notes to the vault (mirrors
            # the non-streaming run_full_council guard).
            if not stage1_results:
                error_stage3 = {"model": "error", "response": "All models failed to respond. Please try again."}
                yield f"data: {json.dumps({'type': 'stage1_complete', 'data': []})}\n\n"
                yield f"data: {json.dumps({'type': 'stage3_complete', 'data': error_stage3})}\n\n"
                if title_task:
                    title = await title_task
                    storage.update_conversation_title(conversation_id, title)
                    yield f"data: {json.dumps({'type': 'title_complete', 'data': {'title': title}})}\n\n"
                storage.add_assistant_message(conversation_id, [], [], error_stage3)
                yield f"data: {json.dumps({'type': 'complete'})}\n\n"
                return

            yield f"data: {json.dumps({'type': 'stage1_complete', 'data': stage1_results})}\n\n"

            # --- Stage B, step 2: peer rankings ---
            yield f"data: {json.dumps({'type': 'stage2_start'})}\n\n"
            stage2_results, label_to_model = await stage2_collect_rankings(deepdive_query, stage1_results, deepdive_context, max_tokens=cap)
            aggregate_rankings = calculate_aggregate_rankings(stage2_results, label_to_model)
            yield f"data: {json.dumps({'type': 'stage2_complete', 'data': stage2_results, 'metadata': {'label_to_model': label_to_model, 'aggregate_rankings': aggregate_rankings}})}\n\n"

            # --- Stage B, step 3: chairman synthesis (per-ticker sections) ---
            want_sizing = parse_budget(request.content) is not None
            yield f"data: {json.dumps({'type': 'stage3_start'})}\n\n"
            stage3_result = await stage3_synthesize_final(
                deepdive_query, stage1_results, stage2_results, deepdive_context,
                max_tokens=cap,
                shortlist_tickers=[item["ticker"] for item in shortlist],
                require_conviction=want_sizing,
            )

            # --- Optional Python position sizing when a budget was given ---
            # Parse conviction from the full report, then strip the ranking block
            # so it doesn't bleed into per-ticker notes or the displayed report.
            full_report = stage3_result.get("response", "")
            report_for_export = full_report
            if want_sizing and shortlist:
                position_sizing = await run_position_sizing(
                    request.content, shortlist, metrics_by_ticker, full_report
                )
                report_for_export = strip_conviction_block(full_report)
                stage3_result["response"] = (
                    report_for_export + "\n\n" + position_sizing["markdown"]
                    if position_sizing else report_for_export
                )

            yield f"data: {json.dumps({'type': 'stage3_complete', 'data': stage3_result})}\n\n"

            # --- Export each ticker's analysis to Obsidian (best-effort) ---
            if shortlist:
                exported = export_notes_from_report(report_for_export, shortlist, metrics_by_ticker)
                if exported:
                    yield f"data: {json.dumps({'type': 'export_complete', 'data': exported})}\n\n"
            else:
                # No shortlist -> nothing to export. Log it so an empty screening
                # result doesn't get mistaken for a broken Obsidian export.
                print("[obsidian] export skipped: screening produced an empty "
                      "shortlist, so there are no tickers to write notes for")

            # Wait for title generation if it was started
            if title_task:
                title = await title_task
                storage.update_conversation_title(conversation_id, title)
                yield f"data: {json.dumps({'type': 'title_complete', 'data': {'title': title}})}\n\n"

            # Save complete assistant message
            storage.add_assistant_message(
                conversation_id,
                stage1_results,
                stage2_results,
                stage3_result
            )

            # Send completion event
            yield f"data: {json.dumps({'type': 'complete'})}\n\n"

        except Exception as e:
            # Send error event
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
