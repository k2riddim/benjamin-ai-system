from __future__ import annotations

from typing import Dict, Any, List
import logging

from openai import OpenAI
from agentic_app.config.settings import settings
from agentic_app.agents.llm_utils import complete_text_with_guards
from agentic_app.agents.tools import memory_store
from langsmith.run_helpers import traceable


class QueryContextualizer:
    """Rewrites a user query into a self-contained prompt by resolving references
    using the short-term conversation history.

    Returns a dict with keys:
      - text: rewritten user text (string)
      - notes: optional brief explanation of what was resolved
    """

    def __init__(self) -> None:
        self.client = OpenAI(api_key=settings.openai_api_key) if settings.openai_api_key else None

    @traceable(name="contextualizer.rewrite", run_type="chain")
    def rewrite(self, user_text: str, short_history: List[Dict[str, str]] | None = None) -> Dict[str, Any]:
        short_history = short_history or []
        if not user_text:
            return {"text": "", "notes": "empty input"}

        # If no LLM available, perform a minimal heuristic fallback
        if not self.client:
            # Build a compact inline context from last few user turns
            last_user_mentions: List[str] = []
            for msg in reversed(short_history):
                if msg.get("role") == "user":
                    text = (msg.get("text") or "").strip()
                    if text:
                        last_user_mentions.append(text)
                    if len(last_user_mentions) >= 3:
                        break
            if last_user_mentions:
                context_str = " | prior: " + "; ".join(reversed(last_user_mentions))
            else:
                context_str = ""
            return {
                "text": f"{user_text}" + (context_str if context_str else ""),
                "notes": "fallback: appended brief prior user mentions",
            }

        # LLM-powered contextual rewrite
        try:
            # Limit to the last 6 user/assistant turns to keep prompt lean
            recent = (short_history or [])[-6:]
            condensed = "\n".join([f"{m.get('role')}: {m.get('text')[:400]}" for m in recent])
            system = (
                "You are a Query Contextualizer for an AI coaching system. "
                "Rewrite the latest user message into a clear, self-contained prompt by resolving pronouns and references "
                "to the provided conversation history. Inject specific values explicitly (e.g., numbers, distances) when referenced. "
                "Return STRICT JSON with keys: rewritten (string), notes (string, brief)."
            )
            user = (
                f"Conversation history (most recent last, trimmed):\n{condensed or 'none'}\n\n"
                f"Latest user message: {user_text[:600]}\n"
                "Task: Produce a single rewritten prompt capturing what the user is asking now."
            )
            text, diag = complete_text_with_guards(
                self.client,
                model=(settings.openai_context_model or settings.openai_model),
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                max_completion_tokens=3500,
                temperature=0,
            )
            try:
                logging.getLogger("agentic_app").debug(f"gpt5.diag.contextualizer.rewrite: {diag}")
            except Exception:
                pass
            try:
                # Surface diagnostics inside the notes field if parsing fails later
                self._last_diag = diag  # type: ignore[attr-defined]
            except Exception:
                pass
            import json as _json
            parsed = _json.loads(text)
            rewritten = (parsed.get("rewritten") or user_text).strip()
            notes = (parsed.get("notes") or "").strip()
            return {"text": rewritten, "notes": notes}
        except Exception:
            # Robust fallback to original text
            # Include last diagnostics if available to aid debugging in the UI/logs
            try:
                diag_note = f" llm_diag={getattr(self, '_last_diag', {})}"  # type: ignore[attr-defined]
            except Exception:
                diag_note = ""
            return {"text": user_text, "notes": ("llm_failure_fallback" + diag_note).strip()}

    @traceable(name="contextualizer.build_packet", run_type="chain")
    def build(self, user_text: str, short_history: List[Dict[str, str]] | None = None, memories_top_k: int = 5) -> Dict[str, Any]:
        """Build a unified context packet with conversational rewrite, long-term memory hits, and key metrics.

        Output keys:
          - original_query: str
          - contextualized_query: str
          - retrieved_memories: List[str]
          - extracted_metrics: List[str]
          - notes: str (brief)
        """
        short_history = short_history or []
        # Step 1: conversational rewrite
        rw = self.rewrite(user_text, short_history=short_history)
        contextualized = rw.get("text") or user_text
        notes = rw.get("notes") or ""
        # Step 2: long-term memory search (Qdrant)
        mem_hits = []
        try:
            results = memory_store.search(contextualized, top_k=memories_top_k)
            for r in (results or []):
                text = r.get("text") or ""
                if text:
                    mem_hits.append(text)
        except Exception:
            mem_hits = []
        # Step 3: metric extraction
        extracted = self._extract_metrics_llm(contextualized, short_history) or []
        return {
            "original_query": user_text,
            "contextualized_query": contextualized,
            "retrieved_memories": mem_hits,
            "extracted_metrics": extracted,
            "notes": notes,
        }

    def _extract_metrics_llm(self, contextualized: str, short_history: List[Dict[str, str]] | None) -> List[str] | None:
        if not self.client:
            return None
        try:
            system = (
                "Extract key metrics, events, and time spans mentioned in the user's request. "
                "Return STRICT JSON with key: metrics (array of short strings)."
            )
            user = f"Text: {contextualized}"
            text, diag2 = complete_text_with_guards(
                self.client,
                model=(settings.openai_context_model or settings.openai_model),
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                max_completion_tokens=3200,
                temperature=0,
            )
            try:
                logging.getLogger("agentic_app").debug(f"gpt5.diag.contextualizer.extract: {diag2}")
            except Exception:
                pass
            import json as _json
            parsed = _json.loads(text)
            arr = parsed.get("metrics") or []
            return [str(x)[:64] for x in arr if isinstance(x, str) and x.strip()][:12]
        except Exception:
            return None

    


