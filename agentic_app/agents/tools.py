from __future__ import annotations

import requests
from typing import Any, Dict, Optional

from agentic_app.config.settings import settings
from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    Range,
)
from langchain_openai import OpenAIEmbeddings
import logging
from uuid import uuid4


class DataApiClient:
    def __init__(self, base_url: Optional[str] = None, timeout: int = 30):
        self.base_url = base_url or settings.data_api_base_url
        self.timeout = timeout

    def get_latest_health(self) -> Dict[str, Any]:
        resp = requests.get(f"{self.base_url}/health-data/latest", timeout=self.timeout)
        resp.raise_for_status()
        return resp.json() or {}

    def get_recent_readiness(self, days: int = 7) -> Any:
        resp = requests.get(f"{self.base_url}/readiness/recent", params={"days": days}, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def get_recent_training_load(self, days: int = 28) -> Any:
        resp = requests.get(f"{self.base_url}/training-load/recent", params={"days": days}, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def get_recent_activities(self, days: int = 7, limit: int = 10) -> Any:
        resp = requests.get(f"{self.base_url}/activities/recent", params={"days": days, "limit": limit}, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def get_rich_context(self) -> Dict[str, Any]:
        # Prefer consolidated dashboard that includes trend packs
        try:
            dash = requests.get(f"{self.base_url}/dashboard/agent-context", timeout=self.timeout).json()
            if isinstance(dash, dict) and dash:
                return self._sanitize_remove_stress(dash)
        except Exception:
            pass
        # Fallback: stitch from individual endpoints
        ctx: Dict[str, Any] = {}
        try:
            ctx["latest_health"] = self.get_latest_health()
        except Exception:
            ctx["latest_health"] = {}
        try:
            ctx["readiness_recent"] = self.get_recent_readiness(7)
        except Exception:
            ctx["readiness_recent"] = []
        try:
            ctx["training_load"] = self.get_recent_training_load(28)
        except Exception:
            ctx["training_load"] = []
        try:
            ctx["weight_recent"] = requests.get(f"{self.base_url}/weight/recent", params={"days": 60}, timeout=self.timeout).json()
        except Exception:
            ctx["weight_recent"] = []
        try:
            ctx["activities_recent"] = self.get_recent_activities(14, 10)
        except Exception:
            ctx["activities_recent"] = []
        return self._sanitize_remove_stress(ctx)

    # --------- Weather (Vincennes, France) ---------
    def get_weather_vincennes(self) -> Dict[str, Any]:
        """Fetch current and near-term weather for Vincennes, France using Open-Meteo (no key).

        Returns a compact dict with current conditions and hourly summaries for today.
        """
        try:
            # Vincennes approx: 48.847N, 2.433E
            url = (
                "https://api.open-meteo.com/v1/forecast?latitude=48.847&longitude=2.433"
                "&current=temperature_2m,relative_humidity_2m,apparent_temperature,wind_speed_10m,precipitation"
                "&hourly=temperature_2m,apparent_temperature,precipitation_probability,precipitation,wind_speed_10m"
                "&forecast_days=1&timezone=Europe%2FParis"
            )
            r = requests.get(url, timeout=10)
            r.raise_for_status()
            data = r.json() or {}
            out: Dict[str, Any] = {"location": {"name": "Vincennes, France", "lat": 48.847, "lon": 2.433}}
            cur = (data.get("current") or {})
            out["current"] = {
                "temperature_c": cur.get("temperature_2m"),
                "apparent_c": cur.get("apparent_temperature"),
                "humidity_pct": cur.get("relative_humidity_2m"),
                "wind_speed_kmh": cur.get("wind_speed_10m"),
                "precip_mm": cur.get("precipitation"),
            }
            # Summaries from hourly next few hours
            hourly = data.get("hourly") or {}
            out["today"] = {
                "max_temp_c": (max(hourly.get("temperature_2m", []) or [None]) if hourly.get("temperature_2m") else None),
                "max_precip_prob_pct": (max(hourly.get("precipitation_probability", []) or [0]) if hourly.get("precipitation_probability") else 0),
                "max_wind_kmh": (max(hourly.get("wind_speed_10m", []) or [0]) if hourly.get("wind_speed_10m") else 0),
            }
            # Derive simple environmental flags
            try:
                max_temp = out["today"]["max_temp_c"]
                out["flags"] = {
                    "heat_warning": bool(max_temp is not None and float(max_temp) >= 30.0 or (out.get("current", {}).get("temperature_c") or 0) >= 30.0),
                    "high_wind": bool((out["today"].get("max_wind_kmh") or 0) >= 35),
                    "rain_likely": bool((out["today"].get("max_precip_prob_pct") or 0) >= 60),
                }
            except Exception:
                out["flags"] = {"heat_warning": False, "high_wind": False, "rain_likely": False}
            return out
        except Exception:
            return {"location": {"name": "Vincennes, France"}, "current": {}, "today": {}}

    def _sanitize_remove_stress(self, obj: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively remove any keys containing 'stress' (case-insensitive) from dictionaries and lists."""
        try:
            def cleanse(value):
                if isinstance(value, dict):
                    return {k: cleanse(v) for k, v in value.items() if "stress" not in str(k).lower()}
                if isinstance(value, list):
                    return [cleanse(v) for v in value]
                return value
            return cleanse(obj)
        except Exception:
            # Best-effort: return original if something goes wrong
            return obj


data_api = DataApiClient(timeout=settings.http_timeout_seconds)


class MemoryStore:
    """Qdrant-backed long-term memory for agent context (RAG-style)."""

    def __init__(self):
        self.enabled = bool(settings.qdrant_url)
        if not self.enabled:
            self.client = None
            self.embeddings = None
            return
        self.client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key or None)
        self.embeddings = OpenAIEmbeddings(model=settings.embedding_model, api_key=settings.openai_api_key)
        # Ensure collection exists
        try:
            self.client.get_collection(collection_name="benjamin_agent_memory")
        except Exception:
            self.client.recreate_collection(
                collection_name="benjamin_agent_memory",
                vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
            )

    def add_memory(self, text: str, metadata: Optional[Dict[str, Any]] = None):
        if not self.enabled or not text:
            return
        try:
            vec = self.embeddings.embed_query(text)
            point = PointStruct(id=str(uuid4()), vector=vec, payload={"text": text, **(metadata or {})})
            self.client.upsert(collection_name="benjamin_agent_memory", points=[point])
        except Exception as e:
            logging.getLogger("agentic_app").warning(f"Qdrant add_memory failed: {e}")

    def search(self, query: str, top_k: int = 5) -> list[Dict[str, Any]]:
        if not self.enabled or not query:
            return []
        vec = self.embeddings.embed_query(query)
        res = self.client.search(collection_name="benjamin_agent_memory", query_vector=vec, limit=top_k)
        return [{"text": r.payload.get("text", ""), "score": r.score, **(r.payload or {})} for r in res]

    # --------- Event management in long-term memory ---------
    def add_event(self, title: str, date_iso: str, metadata: Optional[Dict[str, Any]] = None) -> str | None:
        """Store a user-managed event (e.g., race) with ISO date and title.

        Returns the point id.
        """
        if not self.enabled or not title or not date_iso:
            return None
        try:
            payload = {"type": "event", "title": title, "date": date_iso, "status": (metadata or {}).get("status", "planned")}
            if metadata:
                payload.update(metadata)
            vec = self.embeddings.embed_query(f"event: {title} on {date_iso}")
            pid = str(uuid4())
            self.client.upsert(
                collection_name="benjamin_agent_memory",
                points=[PointStruct(id=pid, vector=vec, payload=payload)],
            )
            return pid
        except Exception as e:
            logging.getLogger("agentic_app").warning(f"Qdrant add_event failed: {e}")
            return None

    def list_events(self, upcoming_only: bool = True, limit: int = 50) -> list[Dict[str, Any]]:
        if not self.enabled:
            return []
        try:
            conditions = [FieldCondition(key="type", match=MatchValue(value="event"))]
            if upcoming_only:
                try:
                    from datetime import datetime

                    today = datetime.utcnow().date().isoformat()
                    conditions.append(FieldCondition(key="date", range=Range(gte=today)))
                except Exception:
                    pass
            flt = Filter(must=conditions)
            points, _, _ = self.client.scroll(
                collection_name="benjamin_agent_memory",
                limit=limit,
                with_payload=True,
                with_vectors=False,
                scroll_filter=flt,
            )
            out: list[Dict[str, Any]] = []
            from datetime import datetime

            for p in points or []:
                payload = p.payload or {}
                date_s = payload.get("date")
                days_until = None
                try:
                    if date_s:
                        d = datetime.fromisoformat(date_s)
                        days_until = (d.date() - datetime.utcnow().date()).days
                except Exception:
                    pass
                out.append({"id": str(p.id), **payload, "days_until": days_until})
            # Sort by date
            out.sort(key=lambda e: e.get("date") or "")
            return out
        except Exception as e:
            logging.getLogger("agentic_app").warning(f"Qdrant list_events failed: {e}")
            return []

    def delete_event(self, event_id: str | None = None, title_match: str | None = None) -> int:
        """Delete events by id or by fuzzy title match.

        Returns number of deleted points.
        """
        if not self.enabled:
            return 0
        try:
            if event_id:
                self.client.delete(collection_name="benjamin_agent_memory", points_selector=[event_id])
                return 1
            if title_match:
                # Scroll and match in Python for simplicity
                points, _, _ = self.client.scroll(
                    collection_name="benjamin_agent_memory",
                    limit=100,
                    with_payload=True,
                    with_vectors=False,
                    scroll_filter=Filter(must=[FieldCondition(key="type", match=MatchValue(value="event"))]),
                )
                to_delete: list[str] = []
                for p in points or []:
                    title = (p.payload or {}).get("title", "")
                    if title_match.lower() in title.lower():
                        to_delete.append(str(p.id))
                if to_delete:
                    self.client.delete(collection_name="benjamin_agent_memory", points_selector=to_delete)
                return len(to_delete)
        except Exception as e:
            logging.getLogger("agentic_app").warning(f"Qdrant delete_event failed: {e}")
        return 0

    def update_event_status(self, event_id: str, status: str) -> bool:
        if not self.enabled or not event_id or status not in {"planned", "active", "completed", "canceled"}:
            return False
        try:
            # Merge payload: set status
            self.client.set_payload(
                collection_name="benjamin_agent_memory",
                payload={"status": status},
                points=[event_id],
            )
            return True
        except Exception as e:
            logging.getLogger("agentic_app").warning(f"Qdrant update_event_status failed: {e}")
            return False


memory_store = MemoryStore()


# ---------------- Short-term memory (chat-scoped buffer) -----------------
from collections import deque
from typing import Deque, Tuple


class ShortTermMemory:
    """Session-scoped memory keyed by chat_id.

    - Keeps a full session transcript (until session end)
    - Also maintains a rolling window for compact context
    - Tracks last agent outputs and last classification
    - Tracks session activity for inactivity watchdog
    """

    def __init__(self, max_turns: int = 20) -> None:
        self.max_turns = max_turns
        self._messages_rolling: dict[str, Deque[Tuple[str, str]]] = {}
        self._transcripts: dict[str, list[Tuple[str, str]]] = {}
        self._last_agent_outputs: dict[str, dict] = {}
        self._last_classification: dict[str, dict] = {}
        self._last_activity: dict[str, float] = {}  # Track last activity timestamp per session

    def add_user_message(self, chat_id: str, text: str) -> None:
        if not chat_id or not text:
            return
        q = self._messages_rolling.setdefault(chat_id, deque(maxlen=self.max_turns))
        q.append(("user", text))
        self._transcripts.setdefault(chat_id, []).append(("user", text))
        self._update_activity(chat_id)

    def add_agent_reply(self, chat_id: str, reply_text: str, agent_outputs: dict | None = None) -> None:
        if not chat_id or not reply_text:
            return
        q = self._messages_rolling.setdefault(chat_id, deque(maxlen=self.max_turns))
        q.append(("assistant", reply_text))
        self._transcripts.setdefault(chat_id, []).append(("assistant", reply_text))
        if agent_outputs is not None:
            self._last_agent_outputs[chat_id] = agent_outputs
        self._update_activity(chat_id)

    def get_context(self, chat_id: str, turns: int = 8) -> list[dict[str, str]]:
        if not chat_id or chat_id not in self._messages_rolling:
            return []
        q = self._messages_rolling[chat_id]
        last = list(q)[-turns:]
        return [{"role": r, "text": t} for r, t in last]

    def get_last_agent_outputs(self, chat_id: str) -> dict:
        return self._last_agent_outputs.get(chat_id, {})

    def set_last_classification(self, chat_id: str, classification: dict) -> None:
        if chat_id:
            self._last_classification[chat_id] = classification or {}

    def get_last_classification(self, chat_id: str) -> dict:
        return self._last_classification.get(chat_id, {})

    def clear(self, chat_id: str) -> None:
        for store in (self._messages_rolling, self._transcripts, self._last_agent_outputs, self._last_classification, self._last_activity):
            if chat_id in store:
                del store[chat_id]

    def get_plain_conversation(self, chat_id: str, turns: int | None = None) -> str:
        msgs = self._transcripts.get(chat_id)
        if not msgs:
            return ""
        items = msgs[-turns:] if isinstance(turns, int) else msgs
        return "\n".join([f"{r}: {t}" for r, t in items])

    def get_full_transcript(self, chat_id: str) -> str:
        return self.get_plain_conversation(chat_id)
    
    def _update_activity(self, chat_id: str) -> None:
        """Update last activity timestamp for a session"""
        import time
        if chat_id:
            self._last_activity[chat_id] = time.time()
    
    def get_inactive_sessions(self, timeout_seconds: int = 3600) -> list[str]:
        """Get list of session IDs that have been inactive for longer than timeout_seconds (default 1 hour)"""
        import time
        current_time = time.time()
        inactive_sessions = []
        
        for chat_id, last_activity in self._last_activity.items():
            if current_time - last_activity > timeout_seconds:
                # Only consider sessions with actual content as inactive
                if chat_id in self._transcripts and self._transcripts[chat_id]:
                    inactive_sessions.append(chat_id)
        
        return inactive_sessions
    
    def get_all_active_sessions(self) -> list[str]:
        """Get all session IDs that have content"""
        return [chat_id for chat_id in self._transcripts.keys() if self._transcripts[chat_id]]


short_memory = ShortTermMemory()


