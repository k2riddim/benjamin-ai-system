from __future__ import annotations

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Any, Dict, List
import json as _json
import re
from datetime import datetime
import threading
from collections import defaultdict

import typer
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
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

# Global status tracking for thinking status
class StatusTracker:
    """Track thinking status for each session"""
    def __init__(self):
        self._statuses: Dict[str, Dict[str, Any]] = defaultdict(dict)
        self._lock = threading.Lock()
    
    def set_status(self, session_id: str, status: str, details: str = "", agents: List[str] = None):
        """Set the current status for a session"""
        with self._lock:
            self._statuses[session_id] = {
                "status": status,
                "details": details,
                "agents": agents or [],
                "timestamp": datetime.now().isoformat(),
            }
    
    def get_status(self, session_id: str) -> Dict[str, Any]:
        """Get the current status for a session"""
        with self._lock:
            return self._statuses.get(session_id, {
                "status": "idle",
                "details": "",
                "agents": [],
                "timestamp": datetime.now().isoformat(),
            })
    
    def clear_status(self, session_id: str):
        """Clear status for a session"""
        with self._lock:
            self._statuses.pop(session_id, None)

status_tracker = StatusTracker()

# Session inactivity watchdog
class SessionWatchdog:
    """Monitor sessions for inactivity and automatically end them after 1 hour"""
    
    def __init__(self, check_interval_minutes: int = 10):
        self.check_interval_minutes = check_interval_minutes
        self.is_running = False
        self._task: asyncio.Task | None = None
    
    async def start(self):
        """Start the inactivity watchdog background task"""
        if self.is_running:
            return
        self.is_running = True
        self._task = asyncio.create_task(self._watchdog_loop())
        logger.info(f"Session inactivity watchdog started (check every {self.check_interval_minutes} minutes)")
    
    async def stop(self):
        """Stop the inactivity watchdog"""
        self.is_running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        logger.info("Session inactivity watchdog stopped")
    
    async def _watchdog_loop(self):
        """Main watchdog loop that periodically checks for inactive sessions"""
        while self.is_running:
            try:
                await self._check_and_end_inactive_sessions()
                # Wait for next check interval
                await asyncio.sleep(self.check_interval_minutes * 60)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"Session watchdog error: {e}")
                # Continue running even if there's an error
                await asyncio.sleep(60)  # Wait 1 minute before retrying
    
    async def _check_and_end_inactive_sessions(self):
        """Check for inactive sessions and end them"""
        try:
            inactive_sessions = short_memory.get_inactive_sessions(timeout_seconds=3600)  # 1 hour
            
            if not inactive_sessions:
                return
            
            logger.info(f"Found {len(inactive_sessions)} inactive sessions to end: {inactive_sessions}")
            
            for session_id in inactive_sessions:
                try:
                    await self._end_session_internally(session_id, "inactivity_timeout")
                    logger.info(f"Ended inactive session: {session_id}")
                except Exception as e:
                    logger.warning(f"Failed to end inactive session {session_id}: {e}")
        
        except Exception as e:
            logger.warning(f"Error checking inactive sessions: {e}")
    
    async def _end_session_internally(self, session_id: str, reason: str = "inactivity_timeout"):
        """Internal method to end a session (reuses logic from /end-session endpoint)"""
        try:
            transcript = short_memory.get_full_transcript(session_id)
        except Exception:
            transcript = ""
        
        preferences_count = insights_count = patterns_count = 0
        
        if transcript:
            try:
                # Strict provenance: keep only USER utterances
                user_only_lines = []
                for line in (transcript or "").split("\n"):
                    if line.lower().startswith("user:"):
                        user_only_lines.append(line.split(":", 1)[1].strip())
                user_only_transcript = "\n".join(user_only_lines).strip()
                
                if user_only_transcript:  # Only extract if there's actual user content
                    extraction = session_agent.extract(user_only_transcript)
                    
                    for pref in extraction.preferences:
                        try:
                            memory_store.add_memory(pref, metadata={"type": "preference", "source": "session_agent", "ended_by": reason})
                            preferences_count += 1
                        except Exception:
                            pass
                    
                    for ins in extraction.insights:
                        try:
                            memory_store.add_memory(ins, metadata={"type": "insight", "source": "session_agent", "ended_by": reason})
                            insights_count += 1
                        except Exception:
                            pass
                    
                    for hp in extraction.health_patterns:
                        try:
                            memory_store.add_memory(hp, metadata={"type": "health_pattern", "source": "session_agent", "ended_by": reason})
                            patterns_count += 1
                        except Exception:
                            pass
            except Exception as e:
                logger.warning(f"Session extraction failed for {session_id}: {e}")
        
        # Clear short-term memory and status for session
        try:
            short_memory.clear(session_id)
            status_tracker.clear_status(session_id)
        except Exception:
            pass
        
        logger.info(f"Session {session_id} ended by {reason}: {preferences_count} preferences, {insights_count} insights, {patterns_count} patterns extracted")

# Initialize watchdog
session_watchdog = SessionWatchdog(check_interval_minutes=10)

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

# Add CORS middleware to allow requests from frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # React dev server
        "http://127.0.0.1:3000",
        "http://localhost:8080",  # Test page server
        "http://127.0.0.1:8080",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def on_startup():
    try:
        create_all_tables()
        logger.info("Ensured database tables exist for Agentic App")
    except Exception as e:
        logger.error(f"Failed to ensure DB tables: {e}")
    
    # Start session inactivity watchdog
    try:
        await session_watchdog.start()
    except Exception as e:
        logger.error(f"Failed to start session watchdog: {e}")


@app.on_event("shutdown")
async def on_shutdown():
    """Cleanup on app shutdown"""
    try:
        await session_watchdog.stop()
    except Exception as e:
        logger.error(f"Failed to stop session watchdog: {e}")


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
    # Generate a session ID for daily discussion tracking
    import uuid
    session_id = f"daily-{uuid.uuid4()}"
    
    # Set thinking status
    status_tracker.set_status(session_id, "thinking", "Preparing daily workout recommendation...", [])
    
    try:
        latest = {}
        try:
            latest = data_api.get_latest_health()
        except Exception as e:
            logger.warning(f"Could not fetch latest health: {e}")

        status_tracker.set_status(session_id, "thinking", "Generating workout plan...", [])
        
        with tracing_v2_enabled(project_name=settings.langsmith_project):
            result = router.route(
                "Generate best workout for today", 
                forced_intent="workout_of_the_day",
                session_id=session_id,
                status_tracker=status_tracker
            )
        
        # Set completion status
        status_tracker.set_status(session_id, "complete", "Daily workout ready", [])
        
    except Exception as e:
        # Set error status
        status_tracker.set_status(session_id, "error", f"Error generating daily workout: {str(e)}", [])
        raise

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
        # Add session ID to headers for status tracking
        jr.headers["X-Session-Id"] = session_id
    except Exception:
        pass
    return jr


@app.get("/status/{session_id}")
async def get_session_status(session_id: str):
    """Get the current thinking status for a session"""
    status = status_tracker.get_status(session_id)
    
    # Update activity when status is checked (user is actively monitoring)
    try:
        if session_id and session_id in short_memory._transcripts:
            short_memory._update_activity(session_id)
    except Exception:
        pass
    
    return JSONResponse(status_code=200, content=status)


@app.get("/watchdog/status")
async def get_watchdog_status():
    """Get the current session watchdog status and active sessions"""
    try:
        active_sessions = short_memory.get_all_active_sessions()
        inactive_sessions = short_memory.get_inactive_sessions(timeout_seconds=3600)
        
        return {
            "watchdog_running": session_watchdog.is_running,
            "check_interval_minutes": session_watchdog.check_interval_minutes,
            "total_active_sessions": len(active_sessions),
            "sessions_to_timeout": len(inactive_sessions),
            "active_sessions": active_sessions[:10],  # Limit to first 10 for brevity
            "sessions_near_timeout": inactive_sessions[:5],  # Limit to first 5
        }
    except Exception as e:
        logger.warning(f"Error getting watchdog status: {e}")
        return {
            "watchdog_running": session_watchdog.is_running,
            "check_interval_minutes": session_watchdog.check_interval_minutes,
            "error": str(e)
        }


@app.post("/watchdog/test-cleanup")
async def test_watchdog_cleanup():
    """Manually trigger a watchdog cleanup check (for testing)"""
    try:
        await session_watchdog._check_and_end_inactive_sessions()
        return {"status": "ok", "message": "Watchdog cleanup check completed"}
    except Exception as e:
        logger.warning(f"Manual watchdog test failed: {e}")
        return {"status": "error", "error": str(e)}


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

    session_id = payload.session_id or "default"
    
    # Set thinking status before processing
    status_tracker.set_status(session_id, "thinking", "Processing your request...", [])
    
    try:
        with tracing_v2_enabled(project_name=settings.langsmith_project):
            # Step 1: build unified context packet
            status_tracker.set_status(session_id, "thinking", "Building context...", [])
            packet = contextualizer.build(payload.text, short_history=history, memories_top_k=5)
            rewritten_text = packet.get("contextualized_query") or payload.text
            
            # Step 2: route using rewritten text
            status_tracker.set_status(session_id, "thinking", "Routing to AI specialists...", [])
            result = router.route(
                rewritten_text,
                forced_intent=payload.intent,
                forced_metric=payload.metric,
                session_id=payload.session_id,
                status_tracker=status_tracker,
            )
            
        # Keep both raw and contextualized texts in logs
        try:
            lg = result.setdefault("logs", {})
            lg["raw_user_text"] = payload.text
            lg["contextualized_text"] = rewritten_text
            lg["context_packet"] = packet
        except Exception:
            pass
        
        # Set completion status
        status_tracker.set_status(session_id, "complete", "Response ready", [])
        
    except Exception as e:
        # Set error status
        status_tracker.set_status(session_id, "error", f"Error processing request: {str(e)}", [])
        raise
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
    final_reply = result.get("formatted_message") or result.get("reply_text", "")
    
    # DEBUGGING: Log what we're sending to Telegram
    logger.info(f"TELEGRAM_DEBUG: reply_length={len(final_reply)}, formatted_message={bool(result.get('formatted_message'))}, reply_text={bool(result.get('reply_text'))}, agents={result.get('agents', [])}")
    if not final_reply:
        logger.error(f"TELEGRAM_DEBUG: Empty reply! Result keys: {list(result.keys())}, logs keys: {list((result.get('logs', {}) or {}).keys())}")
    
    payload_content = {
        "reply": final_reply,
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
    reason = payload.reason or "manual"
    
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
                    memory_store.add_memory(pref, metadata={"type": "preference", "source": "session_agent", "ended_by": reason})
                    preferences_count += 1
                except Exception:
                    pass
            for ins in extraction.insights:
                try:
                    memory_store.add_memory(ins, metadata={"type": "insight", "source": "session_agent", "ended_by": reason})
                    insights_count += 1
                except Exception:
                    pass
            for hp in extraction.health_patterns:
                try:
                    memory_store.add_memory(hp, metadata={"type": "health_pattern", "source": "session_agent", "ended_by": reason})
                    patterns_count += 1
                except Exception:
                    pass
        except Exception as e:
            logger.warning(f"Session extraction failed: {e}")
    # Clear short-term memory and status for session
    try:
        short_memory.clear(sid)
        status_tracker.clear_status(sid)
    except Exception:
        pass
    
    logger.info(f"Session {sid} ended by {reason}: {preferences_count} preferences, {insights_count} insights, {patterns_count} patterns")
    
    return {
        "status": "ok",
        "session_id": sid,
        "ended_by": reason,
        "preferences_logged": preferences_count,
        "insights_logged": insights_count,
        "health_patterns_logged": patterns_count,
    }


# Session inactivity watchdog is now active and integrated above


cli = typer.Typer(name="agentic-app")


@cli.command()
def serve(host: str = "0.0.0.0", port: int = 8012):
    import uvicorn

    uvicorn.run("agentic_app.src.main:app", host=host, port=port, reload=False, log_level=settings.log_level.lower())


# Deprecated: run-daily-discussion was replaced by the Telegram app's daily_message command


if __name__ == "__main__":
    cli()


