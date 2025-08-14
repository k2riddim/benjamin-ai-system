from __future__ import annotations

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Any, Dict
import json as _json
import re
from datetime import datetime

import typer
from fastapi import FastAPI, Request
import asyncio
import time as _time
from fastapi.responses import JSONResponse
from pydantic import BaseModel

sys.path.append(str(Path(__file__).parent.parent.parent))

from agentic_app.config.settings import settings, LOGS_DIR
from agentic_app.agents.project_manager import ProjectManagerRouter
from agentic_app.agents.contextualizer import QueryContextualizer
from agentic_app.agents.tools import data_api, memory_store, short_memory
from agentic_app.src.db_logging import log_agent_discussion
from agentic_app.agents.session_agent import SessionAgent
from shared.database.connection import create_all_tables, get_db_session
from shared.database.models import AgentDiscussions
from langchain_core.tracers.context import tracing_v2_enabled


def setup_logging() -> logging.Logger:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
        handlers=[
            logging.handlers.RotatingFileHandler(LOGS_DIR / settings.log_file, maxBytes=2_000_000, backupCount=5),
            logging.StreamHandler(sys.stdout),
        ],
    )
    # Reduce noisy HTTP logs that may include sensitive URLs
    try:
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("urllib3").setLevel(logging.WARNING)
    except Exception:
        pass
    return logging.getLogger("agentic_app")


logger = setup_logging()
router = ProjectManagerRouter()
contextualizer = QueryContextualizer()
session_agent = SessionAgent()

def _get_session_log_path() -> str:
    """Return a rotating log file path for agent discussions."""
    try:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"agent_discussion_{ts}.log"
        path = str(LOGS_DIR / filename)
        # Touch file
        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write("")
        except Exception:
            pass
        return path
    except Exception:
        # Best-effort fallback
        return str(LOGS_DIR / "agent_discussion_default.log")


def _append_discussion_log(user_text: str, result: Dict[str, Any]) -> None:
    """Append one JSON line capturing context and model answers to the session log file."""
    try:
        path = _get_session_log_path()
        payload: Dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "user_text": user_text,
            "classification": (result.get("logs", {}) or {}).get("classification"),
            "agents": result.get("agents"),
            "context": result.get("context"),
            "agent_outputs": (result.get("logs", {}) or {}).get("agent_outputs"),
            "reply_text": result.get("reply_text"),
            "formatted_message": result.get("formatted_message"),
            "duration_seconds": result.get("duration_seconds"),
        }
        with open(path, "a", encoding="utf-8") as f:
            f.write(_json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.warning(f"Failed to append discussion log: {e}")


class DailyDiscussionResponse(BaseModel):
    telegram_message: str
    agents_involved: list[str]
    duration_seconds: float
    logs: Dict[str, Any]


app = FastAPI(title=settings.app_name, version=settings.version)


@app.on_event("startup")
async def on_startup():
    try:
        create_all_tables()
        logger.info("Ensured database tables exist for Agentic App")
    except Exception as e:
        logger.error(f"Failed to ensure DB tables: {e}")
    # Inactivity watchdog removed with session memory deprecation


@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok", "version": settings.version}


if settings.request_logging_enabled:
    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        import time, uuid
        request_id = request.headers.get("X-Request-Id") or str(uuid.uuid4())
        start = time.perf_counter()
        try:
            body = await request.body()
            body_len = len(body or b"")
        except Exception:
            body_len = -1
        logger.info(f"HTTP {request.method} {request.url.path} rid={request_id} body_len={body_len}")
        try:
            response = await call_next(request)
        except Exception:
            duration = (time.perf_counter() - start) * 1000
            logger.exception(f"HTTP EXC {request.method} {request.url.path} rid={request_id} dur_ms={duration:.1f}")
            raise
        duration = (time.perf_counter() - start) * 1000
        try:
            response.headers["X-Request-Id"] = request_id
        except Exception:
            pass
        logger.info(f"HTTP {request.method} {request.url.path} -> {getattr(response, 'status_code', '?')} rid={request_id} dur_ms={duration:.1f}")
        return response


@app.post("/daily-discussion", response_model=DailyDiscussionResponse)
async def daily_discussion(request: Request) -> Any:
    latest = {}
    try:
        latest = data_api.get_latest_health()
    except Exception as e:
        logger.warning(f"Could not fetch latest health: {e}")

    with tracing_v2_enabled(project_name=settings.langsmith_project):
        result = router.route("Generate best workout for today", forced_intent="workout_of_the_day")

    # Use PM's formatted message if present in logs or reply
    msg = result.get("formatted_message") or result.get("reply_text", "")
    telegram_message = msg[:3000]

    response = DailyDiscussionResponse(
        telegram_message=telegram_message,
        agents_involved=result.get("agents", []),
        duration_seconds=float(result.get("duration_seconds", 0.0)),
        logs=result.get("logs", {}),
    )

    try:
        discussion_id = log_agent_discussion(
            discussion_type="daily",
            topic="daily_best_workout",
            summary="Daily best workout recommendation",
            message_content=response.telegram_message,
            agents_involved=response.agents_involved,
            agent_inputs=None,
            agent_analyses=None,
            agent_recommendations=None,
            final_recommendations=None,
            benjamin_context=result.get("context"),
            data_sources_used={"latest_health": bool(latest)},
            duration_seconds=response.duration_seconds,
        )
        if discussion_id:
            response.logs = {**response.logs, "discussion_id": discussion_id}
    except Exception:
        pass

    jr = JSONResponse(status_code=200, content=response.model_dump())
    try:
        if response.logs.get("discussion_id"):
            jr.headers["X-Discussion-Id"] = response.logs["discussion_id"]
    except Exception:
        pass
    return jr


class RouteRequest(BaseModel):
    text: str
    intent: str | None = None
    metric: str | None = None
    as_assistant: bool | None = None
    debug: bool | None = None
    session_id: str | None = None


@app.post("/route")
async def route_message(payload: RouteRequest):
    # Optional assistant seeding: record an assistant message into short-term memory
    if bool(payload.as_assistant):
        try:
            sid = payload.session_id or "default"
            short_memory.add_agent_reply(sid, payload.text, agent_outputs=None)
        except Exception:
            pass
        return JSONResponse(status_code=200, content={"seeded": True})

    # Record user turn into short-term memory (if session_id provided)
    try:
        if payload.session_id:
            short_memory.add_user_message(payload.session_id, payload.text)
    except Exception:
        pass

    # Build short-term history for contextualizer
    try:
        history = short_memory.get_context(payload.session_id, 8) if payload.session_id else []
    except Exception:
        history = []

    with tracing_v2_enabled(project_name=settings.langsmith_project):
        # Step 1: build unified context packet
        packet = contextualizer.build(payload.text, short_history=history, memories_top_k=5)
        rewritten_text = packet.get("contextualized_query") or payload.text
        # Step 2: route using rewritten text
        result = router.route(
            rewritten_text,
            forced_intent=payload.intent,
            forced_metric=payload.metric,
            session_id=payload.session_id,
        )
        # Keep both raw and contextualized texts in logs
        try:
            lg = result.setdefault("logs", {})
            lg["raw_user_text"] = payload.text
            lg["contextualized_text"] = rewritten_text
            lg["context_packet"] = packet
        except Exception:
            pass
    # Append context and LLM answer into log
    try:
        _append_discussion_log(payload.text, result)
    except Exception:
        pass
    # Optional context debug (log)
    if settings.context_debug_enabled or bool(payload.debug):
        try:
            compact_ctx = {
                "latest_health_keys": list((result.get("context", {}).get("latest_health") or {}).keys()),
                "has_trends": bool(result.get("context", {}).get("trends")),
                "has_baselines": bool(result.get("context", {}).get("baselines")),
                "long_term_preview": (result.get("context", {}).get("long_term_summary") or "")[:300],
                "short_memory_turns": len(short_memory.get_context(payload.session_id, 99)) if payload.session_id else 0,
            }
            logger.info(f"CTX_DEBUG compact={compact_ctx}")
        except Exception:
            pass

    # Build response payload (always)
    payload_content = {
        "reply": result.get("formatted_message") or result["reply_text"],
        "agents": result.get("agents", []),
        "duration_seconds": result.get("duration_seconds", 0.0),
        "logs": result.get("logs", {}),
    }
    if settings.context_debug_enabled or bool(payload.debug):
        try:
            payload_content["context_debug"] = {
                "latest_health_keys": list((result.get("context", {}).get("latest_health") or {}).keys()),
                "has_trends": bool(result.get("context", {}).get("trends")),
                "has_baselines": bool(result.get("context", {}).get("baselines")),
                "long_term": result.get("context", {}).get("long_term"),
                "short_memory": short_memory.get_context(payload.session_id, 8) if payload.session_id else [],
            }
        except Exception:
            pass
    return JSONResponse(status_code=200, content=payload_content)


# ---------------- Event management endpoints ----------------
class AddEventPayload(BaseModel):
    text: str | None = None
    title: str | None = None
    date: str | None = None  # ISO date
    status: str | None = None  # planned|active|completed|canceled


@app.post("/events/add")
async def add_event(payload: AddEventPayload):
    """Add a new event. If text is provided, parse it; else use title/date fields."""
    try:
        title = payload.title
        date = payload.date
        if payload.text and (not title or not date):
            parsed = router._parse_event_intent(payload.text, intent="event_add")
            title = title or parsed.get("title")
            date = date or parsed.get("date")
        if not title or not date:
            return JSONResponse(status_code=400, content={"error": "title and date required"})
        metadata = {"status": payload.status} if payload.status else None
        eid = memory_store.add_event(title, date, metadata=metadata)
        return {"id": eid, "title": title, "date": date, "status": (payload.status or "planned")}
    except Exception as e:
        logger.error(f"add_event failed: {e}")
        return JSONResponse(status_code=500, content={"error": "failed"})


@app.get("/events/list")
async def list_events():
    try:
        items = memory_store.list_events(upcoming_only=False)
        return {"events": items}
    except Exception as e:
        logger.error(f"list_events failed: {e}")
        return JSONResponse(status_code=500, content={"error": "failed"})


class DeleteEventPayload(BaseModel):
    id: str | None = None
    title: str | None = None


@app.post("/events/delete")
async def delete_event(payload: DeleteEventPayload):
    try:
        deleted = 0
        if payload.id:
            deleted = memory_store.delete_event(event_id=payload.id)
        elif payload.title:
            deleted = memory_store.delete_event(title_match=payload.title)
        return {"deleted": deleted}
    except Exception as e:
        logger.error(f"delete_event failed: {e}")
        return JSONResponse(status_code=500, content={"error": "failed"})
class UpdateEventStatusPayload(BaseModel):
    id: str
    status: str  # planned|active|completed|canceled


@app.post("/events/update-status")
async def update_event_status(payload: UpdateEventStatusPayload):
    try:
        ok = memory_store.update_event_status(payload.id, payload.status)
        return {"ok": ok}
    except Exception as e:
        logger.error(f"update_event_status failed: {e}")
        return JSONResponse(status_code=500, content={"error": "failed"})


class EndSessionPayload(BaseModel):
    reason: str | None = None
    session_id: str | None = None


@app.post("/end-session")
async def end_session(payload: EndSessionPayload):
    """Summarize and persist durable insights from short-term session, then clear it."""
    sid = payload.session_id or "default"
    try:
        transcript = short_memory.get_full_transcript(sid)
    except Exception:
        transcript = ""
    preferences_count = insights_count = patterns_count = 0
    if transcript:
        try:
            # Strict provenance: keep only USER utterances
            try:
                user_only_lines = []
                for line in (transcript or "").split("\n"):
                    if line.lower().startswith("user:"):
                        # strip leading role label
                        user_only_lines.append(line.split(":", 1)[1].strip())
                user_only_transcript = "\n".join(user_only_lines).strip()
            except Exception:
                user_only_transcript = transcript
            extraction = session_agent.extract(user_only_transcript)
            for pref in extraction.preferences:
                try:
                    memory_store.add_memory(pref, metadata={"type": "preference", "source": "session_agent"})
                    preferences_count += 1
                except Exception:
                    pass
            for ins in extraction.insights:
                try:
                    memory_store.add_memory(ins, metadata={"type": "insight", "source": "session_agent"})
                    insights_count += 1
                except Exception:
                    pass
            for hp in extraction.health_patterns:
                try:
                    memory_store.add_memory(hp, metadata={"type": "health_pattern", "source": "session_agent"})
                    patterns_count += 1
                except Exception:
                    pass
        except Exception as e:
            logger.warning(f"Session extraction failed: {e}")
    # Clear short-term memory for session
    try:
        short_memory.clear(sid)
    except Exception:
        pass
    return {
        "status": "ok",
        "preferences_logged": preferences_count,
        "insights_logged": insights_count,
        "health_patterns_logged": patterns_count,
    }


# Inactivity watchdog removed


cli = typer.Typer(name="agentic-app")


@cli.command()
def serve(host: str = "0.0.0.0", port: int = 8012):
    import uvicorn

    uvicorn.run("agentic_app.src.main:app", host=host, port=port, reload=False, log_level=settings.log_level.lower())


# Deprecated: run-daily-discussion was replaced by the Telegram app's daily_message command


if __name__ == "__main__":
    cli()


