from __future__ import annotations

from dataclasses import dataclass
from typing import List

from openai import OpenAI
from agentic_app.config.settings import settings


@dataclass
class SessionExtraction:
    preferences: List[str]
    insights: List[str]
    health_patterns: List[str]


class SessionAgent:
    """Extracts durable long-term signals from a session transcript.

    Only returns preferences, insights, and health_patterns as requested.
    """

    def __init__(self) -> None:
        self.client = OpenAI(api_key=settings.openai_api_key) if settings.openai_api_key else None

    def extract(self, transcript: str) -> SessionExtraction:
        if not transcript or not self.client:
            return SessionExtraction(preferences=[], insights=[], health_patterns=[])
        system = (
            "You are an expert coach's session summarizer. Only consider USER statements; ignore assistant/AI text. "
            "From the USER transcript, extract three lists: "
            "preferences (stable training likes/dislikes stated by the user), insights (coach-relevant learnings about the user), and health_patterns "
            "(repeated behaviors like overeating triggers, poor sleep patterns, overtraining). "
            "Strict provenance: do NOT infer or invent items that the user did not say or that were only suggested by the AI. "
            "Return strict JSON with keys: preferences (string[]), insights (string[]), health_patterns (string[]). Items must be short (<= 120 chars)."
        )
        user = transcript[:12000]
        resp = self.client.chat.completions.create(
            model=settings.openai_model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0,
            max_tokens=500,
        )
        content = (resp.choices[0].message.content or "").strip()
        try:
            import json

            obj = json.loads(content)
            return SessionExtraction(
                preferences=list(obj.get("preferences", []) or []),
                insights=list(obj.get("insights", []) or []),
                health_patterns=list(obj.get("health_patterns", []) or []),
            )
        except Exception:
            return SessionExtraction(preferences=[], insights=[], health_patterns=[])


