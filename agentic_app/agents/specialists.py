from __future__ import annotations

from typing import Dict, Any, List
from dataclasses import dataclass

from openai import OpenAI
from agentic_app.config.settings import settings
from langsmith.run_helpers import traceable


def _get_client() -> OpenAI:
    if not settings.openai_api_key:
        raise RuntimeError("AGENTIC_APP_OPENAI_API_KEY is not configured")
    return OpenAI(api_key=settings.openai_api_key)


@traceable(name="specialist.chat", run_type="llm")
def _chat(messages: List[Dict[str, str]], temperature: float = 0.4, max_tokens: int = 600) -> str:
    client = _get_client()
    resp = client.chat.completions.create(
        model=settings.openai_model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    content = resp.choices[0].message.content.strip()
    return content


@dataclass
class AgentReply:
    role: str
    content: str
    reasoning: str | None = None


class RunningCoachAgent:
    @traceable(name="running_coach.generate_workout", run_type="chain")
    def generate_workout(self, context: Dict[str, Any], preference: str | None = None, mode: str | None = None, query: str | None = None) -> AgentReply:
        return self.execute_task(context=context, task="GENERATE_WORKOUT_JSON", query=query, preference=preference)

    @traceable(name="running_coach.execute_task", run_type="chain")
    def execute_task(self, context: Dict[str, Any], task: str, query: str | None = None, preference: str | None = None) -> AgentReply:
        now = context.get("now") or ""
        events = context.get("upcoming_events") or []
        events_hint = ", ".join([f"{e.get('title')} in {e.get('days_until','?')}d" for e in events[:3] if isinstance(e, dict)])
        system = (
            "You are an expert Running Coach on a multidisciplinary AI coaching team. "
            "Provide safe, context-aware guidance aligned with readiness, sleep, HRV, training load, preferences, and upcoming events. "
            "Adapt your response to the assigned task."
        )
        instructions: List[str] = []
        if task == "GENERATE_WORKOUT_JSON":
            instructions.append("Generate a single running workout for today as strict JSON with keys: title, summary, details, duration_min, intensity, sport='running'.")
        elif task == "ANSWER_USER_QUESTION":
            instructions.append("Answer in concise natural language suitable for the athlete. No JSON.")
        elif task == "PROVIDE_EXPERT_OPINION":
            instructions.append("Provide a brief expert opinion with clear rationale. No JSON.")
        elif task == "ANALYZE_DATA_AND_SUMMARIZE":
            instructions.append("Summarize what the current context implies for running today. No JSON.")
        else:
            instructions.append("Provide a concise, helpful response appropriate to the task. No JSON.")
        if preference:
            instructions.append(f"Preference: {preference}.")
        if query:
            instructions.append(f"User query: {query}.")
        user = (
            f"Today: {now}. Upcoming events: {events_hint or 'none'}.\n"
            f"Assigned task: {task}.\n"
            f"Context: {context}.\n"
            + " ".join(instructions)
        )
        content = _chat([{"role": "system", "content": system}, {"role": "user", "content": user}], max_tokens=600)
        return AgentReply(role="running_coach", content=content)

    @traceable(name="running_coach.critique", run_type="chain")
    def critique(self, context: Dict[str, Any], colleague_proposals: Dict[str, str]) -> AgentReply:
        """Critique others from a running perspective, without numeric anchoring."""
        system = (
            "You are the running coach on a multidisciplinary staff. Provide a concise critique of colleagues' daily proposals. "
            "Focus on: appropriateness of running intensity/volume relative to readiness, injury risk, scheduling conflicts with other sessions, and coherence with long-term running goals. "
            "Suggest qualitative adjustments only when clearly warranted (avoid arbitrary percentages). No JSON. Max 6 short bullets."
        )
        stitched = "\n\n".join([f"[{role}]\n{content}" for role, content in (colleague_proposals or {}).items()])
        user = f"Context: {context}.\n\nColleagues' proposals (non-running):\n{stitched}"
        content = _chat([{"role": "system", "content": system}, {"role": "user", "content": user}], max_tokens=280)
        return AgentReply(role="running_coach", content=content)


class CyclingCoachAgent:
    @traceable(name="cycling_coach.generate_workout", run_type="chain")
    def generate_workout(self, context: Dict[str, Any], preference: str | None = None, mode: str | None = None, query: str | None = None) -> AgentReply:
        return self.execute_task(context=context, task="GENERATE_WORKOUT_JSON", query=query, preference=preference)

    @traceable(name="cycling_coach.execute_task", run_type="chain")
    def execute_task(self, context: Dict[str, Any], task: str, query: str | None = None, preference: str | None = None) -> AgentReply:
        now = context.get("now") or ""
        events = context.get("upcoming_events") or []
        events_hint = ", ".join([f"{e.get('title')} in {e.get('days_until','?')}d" for e in events[:3] if isinstance(e, dict)])
        system = (
            "You are an expert Cycling Coach on a multidisciplinary AI coaching team. "
            "Provide context-aware guidance based on readiness, recent load, terrain preferences, and upcoming events. "
            "Adapt your response to the assigned task."
        )
        instructions: List[str] = []
        if task == "GENERATE_WORKOUT_JSON":
            instructions.append("Generate a single cycling session for today as strict JSON with keys: title, summary, details, duration_min, intensity, sport='cycling'.")
        elif task == "ANSWER_USER_QUESTION":
            instructions.append("Answer concisely in natural language. No JSON.")
        elif task == "PROVIDE_EXPERT_OPINION":
            instructions.append("Provide a brief expert opinion with rationale. No JSON.")
        elif task == "ANALYZE_DATA_AND_SUMMARIZE":
            instructions.append("Summarize what the context implies for cycling today. No JSON.")
        else:
            instructions.append("Provide a concise, helpful response appropriate to the task. No JSON.")
        if preference:
            instructions.append(f"Preference: {preference}.")
        if query:
            instructions.append(f"User query: {query}.")
        user = (
            f"Today: {now}. Upcoming events: {events_hint or 'none'}.\n"
            f"Assigned task: {task}.\n"
            f"Context: {context}.\n"
            + " ".join(instructions)
        )
        content = _chat([{"role": "system", "content": system}, {"role": "user", "content": user}], max_tokens=600)
        return AgentReply(role="cycling_coach", content=content)

    @traceable(name="cycling_coach.critique", run_type="chain")
    def critique(self, context: Dict[str, Any], colleague_proposals: Dict[str, str]) -> AgentReply:
        system = (
            "You are the cycling coach on a multidisciplinary staff. Critique the colleagues' daily proposals. "
            "Focus on: cycling-relevant fatigue interactions (e.g., high-intensity run vs. ride), terrain/skill suitability, and recovery implications. "
            "Offer qualitative, non-numeric suggestions when needed. No JSON. Max 6 bullets."
        )
        stitched = "\n\n".join([f"[{role}]\n{content}" for role, content in (colleague_proposals or {}).items()])
        user = f"Context: {context}.\n\nColleagues' proposals (non-cycling):\n{stitched}"
        content = _chat([{"role": "system", "content": system}, {"role": "user", "content": user}], max_tokens=260)
        return AgentReply(role="cycling_coach", content=content)


class StrengthCoachAgent:
    @traceable(name="strength_coach.generate_session", run_type="chain")
    def generate_session(self, context: Dict[str, Any], mode: str | None = None, query: str | None = None) -> AgentReply:
        return self.execute_task(context=context, task="GENERATE_WORKOUT_JSON", query=query)

    @traceable(name="strength_coach.execute_task", run_type="chain")
    def execute_task(self, context: Dict[str, Any], task: str, query: str | None = None) -> AgentReply:
        now = context.get("now") or ""
        events = context.get("upcoming_events") or []
        events_hint = ", ".join([f"{e.get('title')} in {e.get('days_until','?')}d" for e in events[:3] if isinstance(e, dict)])
        system = (
            "You are an expert Strength Coach on a multidisciplinary AI coaching team. "
            "Tailor sessions to support endurance without undue fatigue and adapt to the assigned task."
        )
        instructions: List[str] = []
        if task == "GENERATE_WORKOUT_JSON":
            instructions.append("Generate a short full-body routine as strict JSON with keys: title, summary, blocks (each with exercises, sets, reps), duration_min.")
        elif task == "ANSWER_USER_QUESTION":
            instructions.append("Answer concisely in natural language. No JSON.")
        elif task == "PROVIDE_EXPERT_OPINION":
            instructions.append("Provide a brief expert opinion with rationale. No JSON.")
        elif task == "ANALYZE_DATA_AND_SUMMARIZE":
            instructions.append("Summarize what the context implies for strength today. No JSON.")
        else:
            instructions.append("Provide a concise, helpful response appropriate to the task. No JSON.")
        if query:
            instructions.append(f"User query: {query}.")
        user = (
            f"Today: {now}. Upcoming events: {events_hint or 'none'}.\n"
            f"Assigned task: {task}.\n"
            f"Context: {context}.\n"
            + " ".join(instructions)
        )
        content = _chat([{"role": "system", "content": system}, {"role": "user", "content": user}], max_tokens=500)
        return AgentReply(role="strength_coach", content=content)

    @traceable(name="strength_coach.critique", run_type="chain")
    def critique(self, context: Dict[str, Any], colleague_proposals: Dict[str, str]) -> AgentReply:
        system = (
            "You are the strength coach on a multidisciplinary staff. Critique the colleagues' daily proposals. "
            "Focus on: interference effects with strength work, excessive eccentric load, joint/tendon stress, and whether mobility or core support would be preferable. "
            "Provide qualitative suggestions and sequencing notes. No JSON. Max 6 bullets."
        )
        stitched = "\n\n".join([f"[{role}]\n{content}" for role, content in (colleague_proposals or {}).items()])
        user = f"Context: {context}.\n\nColleagues' proposals (non-strength):\n{stitched}"
        content = _chat([{"role": "system", "content": system}, {"role": "user", "content": user}], max_tokens=260)
        return AgentReply(role="strength_coach", content=content)


class NutritionistAgent:
    @traceable(name="nutritionist.advise", run_type="chain")
    def advise(self, context: Dict[str, Any], concern: str | None = None) -> AgentReply:
        return self.execute_task(context=context, task="PROVIDE_EXPERT_OPINION", query=concern or None)

    @traceable(name="nutritionist.execute_task", run_type="chain")
    def execute_task(self, context: Dict[str, Any], task: str, query: str | None = None) -> AgentReply:
        now = context.get("now") or ""
        system = (
            "You are an expert Sports Nutritionist on a multidisciplinary AI coaching team. "
            "Provide practical, compassionate, context-aware guidance and adapt to the assigned task. "
            "Understand eating disorders and their impact on motivation and performance."
        )
        instructions: List[str] = []
        if task == "ANSWER_USER_QUESTION":
            instructions.append("Answer concisely in natural language. No JSON.")
        elif task == "PROVIDE_EXPERT_OPINION":
            instructions.append("Provide 3-5 concise bullets or a short paragraph with clear, actionable guidance. No JSON.")
        elif task == "ANALYZE_DATA_AND_SUMMARIZE":
            instructions.append("Summarize what the context implies for nutrition today. No JSON.")
        else:
            instructions.append("Provide a concise, helpful response appropriate to the task. No JSON.")
        if query:
            instructions.append(f"User concern: {query}.")
        user = (
            f"Today: {now}. Context: {context}. Assigned task: {task}. "
            + " ".join(instructions)
        )
        content = _chat([{"role": "system", "content": system}, {"role": "user", "content": user}], max_tokens=350)
        return AgentReply(role="nutritionist", content=content)

    @traceable(name="nutritionist.critique", run_type="chain")
    def critique(self, context: Dict[str, Any], colleague_proposals: Dict[str, str]) -> AgentReply:
        system = (
            "You are the sports nutritionist on a multidisciplinary staff. Critique the colleagues' daily proposals. "
            "Focus on: fueling timing relative to intensity/volume, hydration, GI tolerance, recovery support, and behaviorally safe guidance. "
            "Suggest practical adjustments without fixed numeric anchors unless supported by context. No JSON. Max 6 bullets."
        )
        stitched = "\n\n".join([f"[{role}]\n{content}" for role, content in (colleague_proposals or {}).items()])
        user = f"Context: {context}.\n\nColleagues' proposals:\n{stitched}"
        content = _chat([{"role": "system", "content": system}, {"role": "user", "content": user}], max_tokens=240)
        return AgentReply(role="nutritionist", content=content)


class PsychologistAgent:
    @traceable(name="psychologist.advise", run_type="chain")
    def advise(self, context: Dict[str, Any], concern: str | None = None) -> AgentReply:
        return self.execute_task(context=context, task="PROVIDE_EXPERT_OPINION", query=concern or None)

    @traceable(name="psychologist.execute_task", run_type="chain")
    def execute_task(self, context: Dict[str, Any], task: str, query: str | None = None) -> AgentReply:
        now = context.get("now") or ""
        system = (
            "You are an expert Sports Psychologist on a multidisciplinary AI coaching team. "
            "Use a supportive tone and adapt your response to the assigned task. "
            "Understand eating disorders and their impact on motivation and performance."
        )
        instructions: List[str] = []
        if task == "ANSWER_USER_QUESTION":
            instructions.append("Answer concisely in natural language. No JSON.")
        elif task == "PROVIDE_EXPERT_OPINION":
            instructions.append("Provide a brief, empathetic response with 2-3 coping strategies and one actionable next step. No JSON.")
        elif task == "ANALYZE_DATA_AND_SUMMARIZE":
            instructions.append("Summarize what the context implies for mindset today. No JSON.")
        else:
            instructions.append("Provide a concise, helpful response appropriate to the task. No JSON.")
        if query:
            instructions.append(f"User concern: {query}.")
        user = (
            f"Today: {now}. Context: {context}. Assigned task: {task}. "
            + " ".join(instructions)
        )
        content = _chat([{"role": "system", "content": system}, {"role": "user", "content": user}], max_tokens=300)
        return AgentReply(role="psychologist", content=content)

    @traceable(name="psychologist.critique", run_type="chain")
    def critique(self, context: Dict[str, Any], colleague_proposals: Dict[str, str]) -> AgentReply:
        system = (
            "You are the sports psychologist on a multidisciplinary staff. Critique the colleagues' daily proposals. "
            "Focus on: psychological readiness, stress and recovery balance, motivation, self-efficacy, and sustainable habits. "
            "Offer supportive, non-judgmental suggestions; keep concise. No JSON. Max 6 bullets."
        )
        stitched = "\n\n".join([f"[{role}]\n{content}" for role, content in (colleague_proposals or {}).items()])
        user = f"Context: {context}.\n\nColleagues' proposals:\n{stitched}"
        content = _chat([{"role": "system", "content": system}, {"role": "user", "content": user}], max_tokens=220)
        return AgentReply(role="psychologist", content=content)


class DataAnalystAgent:
    @traceable(name="data_analyst.answer_metric", run_type="chain")
    def answer_metric(self, context: Dict[str, Any], metric: str) -> AgentReply:
        """Answer a metric query with robust synonym handling.

        Accepts many user-level synonyms and normalizes to the keys used by
        the data API (`latest_health`). When appropriate, supports modality
        disambiguation for VO2max (running vs cycling).
        """
        latest = context.get("latest_health", {})

        # Normalize metric name
        metric_lc = (metric or "").strip().lower()

        # Canonical key mapping with common synonyms
        canonical_map: Dict[str, str] = {
            # Resting heart rate
            "resting_heart_rate": "resting_heart_rate",
            "resting hr": "resting_heart_rate",
            "rhr": "resting_heart_rate",
            "restingheartrate": "resting_heart_rate",
            # HRV
            "hrv": "hrv_score",
            "hrv score": "hrv_score",
            # Sleep
            "sleep score": "sleep_score",
            "sleep": "sleep_score",
            "sleep hours": "sleep_hours",
            # Stress/body battery
            "stress": "stress_score",
            "stress score": "stress_score",
            "body battery": "body_battery",
            # Steps
            "steps": "steps",
            # Weight
            "weight": "body_weight_kg",
            "body weight": "body_weight_kg",
            # Training readiness
            "training readiness": "training_readiness",
            "readiness": "training_readiness",
            # Fitness age
            "fitness age": "fitness_age",
        }

        # VO2max: allow modality selection and common spellings
        vo2_aliases = {"vo2max", "vo2 max", "vo2", "vo2max running", "vo2max cycling"}
        is_vo2 = any(a in metric_lc for a in vo2_aliases)

        if is_vo2:
            # Disambiguate based on hints present in the metric string
            if "cycl" in metric_lc or "bike" in metric_lc:
                key = "vo2max_cycling"
            elif "run" in metric_lc:
                key = "vo2max_running"
            else:
                # Fallback: prefer cycling if running is missing, else running
                key = "vo2max_cycling" if latest.get("vo2max_cycling") is not None else "vo2max_running"
        else:
            key = canonical_map.get(metric_lc, metric)

        value = latest.get(key)
        if value is None:
            # Trend-style answers
            if metric_lc in {"trend", "overall fitness trend", "fitness trend", "overall fitness"}:
                summary = self._summarize_trend(context)
                content = summary or "I could not find enough data to compute an overall fitness trend."
            else:
                content = f"I could not find {key.replace('_', ' ')} in the latest data."
        else:
            label = key.replace("_", " ")
            # Add units for common metrics
            if key.startswith("vo2max"):
                content = f"Latest {label}: {value} ml/kg/min"
            elif key == "resting_heart_rate":
                content = f"Latest {label}: {value} bpm"
            elif key == "body_weight_kg":
                content = f"Latest body weight: {value} kg"
            else:
                content = f"Latest {label}: {value}"

        return AgentReply(role="data_analyst", content=content)

    def _summarize_trend(self, context: Dict[str, Any]) -> str | None:
        """Build a compact trend summary from recent context arrays.

        Uses: readiness_recent, training_load, weight_recent, activities_recent, and latest_health.
        """
        try:
            latest = context.get("latest_health", {}) or {}
            tl = context.get("training_load", []) or []
            wt = context.get("weight_recent", []) or []
            acts = context.get("activities_recent", []) or []

            parts: list[str] = []
            if isinstance(latest.get("vo2max_running"), (int, float)) or isinstance(latest.get("vo2max_cycling"), (int, float)):
                vr = latest.get("vo2max_running")
                vc = latest.get("vo2max_cycling")
                if vr:
                    parts.append(f"VO2max(run): {vr:.1f}")
                if vc:
                    parts.append(f"VO2max(bike): {vc:.1f}")
            # Load trend
            if tl and isinstance(tl, list):
                try:
                    last = tl[-1]
                    parts.append(f"Load 7d/28d: {last.get('acute_load_7d','?')}/{last.get('chronic_load_28d','?')}")
                except Exception:
                    pass
            # Weight trend
            if wt and isinstance(wt, list):
                try:
                    start = wt[0].get("weight_kg")
                    end = wt[-1].get("weight_kg")
                    if isinstance(start, (int, float)) and isinstance(end, (int, float)):
                        delta = end - start
                        parts.append(f"Weight 60d Δ: {delta:+.1f} kg")
                except Exception:
                    pass
            # Volume proxy
            if acts and isinstance(acts, list):
                try:
                    weekly = len([a for a in acts if a])
                    parts.append(f"Activities last 14d: {weekly}")
                except Exception:
                    pass
            return ", ".join(parts) if parts else None
        except Exception:
            return None


class RecoveryAdvisorAgent:
    @traceable(name="recovery_advisor.advise", run_type="chain")
    def advise(self, context: Dict[str, Any], reason: str | None = None) -> AgentReply:
        now = context.get("now") or ""
        system = (
            "You are a recovery advisor on a multidisciplinary AI coaching team. "
            "When a rest day is warranted, provide a concise day plan focused on recovery with optional gentle movement, mobility, sleep targets, and readiness checks. "
            "Keep it athlete-friendly."
        )
        user = (
            f"Today: {now}. Context (condensed): { {k: context.get(k) for k in ['latest_health','training_load','readiness_recent','weather_vincennes']} }. "
            f"Reason for rest: {reason or 'automatic threshold'}."
        )
        content = _chat([{"role": "system", "content": system}, {"role": "user", "content": user}], max_tokens=320)
        return AgentReply(role="recovery_advisor", content=content)


