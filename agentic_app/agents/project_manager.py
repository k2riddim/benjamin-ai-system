from __future__ import annotations

from typing import Dict, Any, Tuple, List
import json
import time

from openai import OpenAI

from agentic_app.config.settings import settings
from agentic_app.agents.specialists import (
    RunningCoachAgent,
    CyclingCoachAgent,
    StrengthCoachAgent,
    NutritionistAgent,
    PsychologistAgent,
    DataAnalystAgent,
    AgentReply,
    RecoveryAdvisorAgent,
)
from agentic_app.agents.tools import data_api, memory_store, short_memory
from agentic_app.agents.contextualizer import QueryContextualizer
from langsmith.run_helpers import traceable
from agentic_app.agents.graph import build_pm_router_graph


class ProjectManagerRouter:
    """Routes user requests to the appropriate specialist agents."""

    def __init__(self) -> None:
        self.running = RunningCoachAgent()
        self.cycling = CyclingCoachAgent()
        self.strength = StrengthCoachAgent()
        self.nutrition = NutritionistAgent()
        self.psychology = PsychologistAgent()
        self.analyst = DataAnalystAgent()
        self.recovery = RecoveryAdvisorAgent()
        self.client = OpenAI(api_key=settings.openai_api_key) if settings.openai_api_key else None
        self.contextualizer = QueryContextualizer()
        # Compile LangGraph router once (if available)
        try:
            self._graph_router = build_pm_router_graph(self)
        except Exception:
            self._graph_router = None

    @traceable(name="pm.classify", run_type="chain")
    def classify(self, user_text: str) -> Dict[str, Any]:
        """Semantic routing using LLM JSON output; no keyword heuristics."""
        if not self.client:
            return {"intent": "general"}

        system = (
            "You are a project manager agent for an AI coaching system. Determine the user's intent and which specialist agents to involve. "
            "Return strict JSON with keys: intent (one of: training for today, metric_query, workout_change, plan_request, nutrition_psych_support, performance_forecast, event_add, event_delete, event_modify, mixed_support, general), "
            "agents (array of strings from: running_coach, cycling_coach, strength_coach, nutritionist, psychologist, data_analyst), and optional fields: metric, sport, intensity, days, race, preference, event_title, event_date, delete_target."
        )
        user = f"User message: {user_text}"
        resp = self.client.chat.completions.create(
            model=settings.openai_model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0,
            max_tokens=200,
        )
        try:
            parsed = json.loads(resp.choices[0].message.content)
            return parsed
        except Exception:
            return {"intent": "general", "agents": ["running_coach"]}

    @traceable(name="pm.route", run_type="chain")
    def route(self, user_text: str, forced_intent: str | None = None, forced_metric: str | None = None, session_id: str | None = None) -> Dict[str, Any]:
        """Route to agents using the LangGraph state machine and build a response. Returns a structured dict with logs."""
        t0 = time.time()
        # Short history for contextualizer/graph
        try:
            short_history = short_memory.get_context(session_id, turns=8) if session_id else []
        except Exception:
            short_history = []

        # If graph is available, run it; else fall back to legacy flow
        if getattr(self, "_graph_router", None) is not None:
            try:
                initial_state = {
                    "user_text": user_text,
                    "session_id": session_id,
                    "forced_intent": forced_intent,
                    "forced_metric": forced_metric,
                    "short_history": short_history,
                }
                graph_result = self._graph_router.invoke(initial_state)  # type: ignore[attr-defined]
                # Ensure logs carry agent_outputs
                logs = graph_result.get("logs", {}) or {}
                if graph_result.get("agent_outputs") is not None:
                    ao = graph_result.get("agent_outputs")
                    try:
                        logs.setdefault("agent_outputs", {}).update(ao if isinstance(ao, dict) else {})
                    except Exception:
                        logs["agent_outputs"] = ao
                result: Dict[str, Any] = {
                    "reply_text": graph_result.get("reply_text", ""),
                    "formatted_message": graph_result.get("formatted_message"),
                    "agents": graph_result.get("agents", []),
                    "context": graph_result.get("context", {}),
                    "logs": logs,
                }
                # Persist user-derived facts (heuristics preserved)
                try:
                    intent = (graph_result.get("intent") or (graph_result.get("classification", {}) or {}).get("intent") or "").lower()
                    lower = (user_text or "").lower()
                    if intent == "workout_change" and ("gravel" in lower or "bike" in lower) and ("tired" in lower or "fatigue" in lower or "recovery" in lower):
                        memory_store.add_memory("Preference: prefers light gravel when tired", metadata={"type": "preference", "source": "user_text"})
                    if intent == "nutrition_psych_support" and ("overeat" in lower or "bloat" in lower or "guilty" in lower):
                        memory_store.add_memory("Health pattern: possible emotional eating after high stress", metadata={"type": "health_pattern", "source": "user_text"})
                except Exception:
                    pass
                # Store turn into short-term memory for continuity
                try:
                    if session_id:
                        classification = graph_result.get("classification") or {}
                        short_memory.set_last_classification(session_id, classification)
                        reply_for_memory = result.get("formatted_message") or result.get("reply_text") or ""
                        short_memory.add_agent_reply(session_id, reply_for_memory, agent_outputs=result.get("logs", {}).get("agent_outputs"))
                except Exception:
                    pass
                result["duration_seconds"] = time.time() - t0
                return result
            except Exception:
                # Fallback to legacy routing if graph execution fails
                pass

        # Legacy fallback
        return self._legacy_route(user_text, forced_intent, forced_metric, session_id, t0)

    # Legacy routing preserved as a fallback path
    def _legacy_route(self, user_text: str, forced_intent: str | None, forced_metric: str | None, session_id: str | None, t0: float) -> Dict[str, Any]:
        try:
            short_history = short_memory.get_context(session_id, turns=8) if session_id else []
        except Exception:
            short_history = []
        try:
            packet = self.contextualizer.build(user_text, short_history=short_history, memories_top_k=5)
            rewritten_text = packet.get("contextualized_query") or user_text
        except Exception:
            packet = {
                "original_query": user_text,
                "contextualized_query": user_text,
                "retrieved_memories": [],
                "extracted_metrics": [],
                "notes": "contextualizer_error",
            }
            rewritten_text = user_text
        from datetime import datetime
        context = data_api.get_rich_context()
        try:
            context["weather_vincennes"] = data_api.get_weather_vincennes()
        except Exception:
            context["weather_vincennes"] = {}
        now = datetime.now()
        context["now"] = now.isoformat()
        context["today"] = now.date().isoformat()
        context["weekday"] = now.strftime("%A")
        try:
            context["contextualized_query"] = rewritten_text
            context["extracted_metrics"] = packet.get("extracted_metrics", [])
            context["retrieved_memories"] = packet.get("retrieved_memories", [])
            context["query_context"] = packet
        except Exception:
            pass
        try:
            events = memory_store.list_events(upcoming_only=True)
        except Exception:
            events = []
        context["upcoming_events"] = events
        context["goals_status"] = {e.get("title"): e.get("status", "planned") for e in (events or []) if e.get("title")}
        memory_hits = memory_store.search(rewritten_text, top_k=3)
        try:
            for t in (packet.get("retrieved_memories") or [])[:5]:
                memory_hits.append({"text": t, "source": "contextualizer"})
        except Exception:
            pass
        context["memory_hits"] = memory_hits
        context["short_memory"] = short_history or []
        is_followup = (len(user_text.split()) <= 12) or any(k in user_text.lower() for k in ["that", "this", "it", "how so", "why", "how did", "explain", "justify"]) 
        classification = {"intent": "general"}
        if forced_intent == "workout_of_the_day":
            classification = {"intent": "workout_of_the_day", "agents": ["running_coach", "cycling_coach", "strength_coach", "nutritionist", "psychologist", "data_analyst"]}
        else:
            classification = self.classify(rewritten_text)
        if forced_intent:
            classification["intent"] = forced_intent
        if forced_metric:
            classification["metric"] = forced_metric
        if session_id and is_followup:
            try:
                prev_cls = short_memory.get_last_classification(session_id)
                if prev_cls and (classification.get("intent") in {None, "general"}):
                    classification = {**prev_cls, **{k: v for k, v in classification.items() if v is not None}}
            except Exception:
                pass
        logs = {
            "classification": classification,
            "raw_user_text": user_text,
            "contextualized_text": rewritten_text,
            "context_packet": packet,
        }
        context.setdefault("latest_health", context.get("latest_health", {}))
        intent = classification.get("intent")
        if is_followup and any(w in user_text.lower() for w in ["explain", "justify", "how", "why"]):
            intent = intent or "general"
            classification["intent"] = "workout_explanation"
        result = self._route_from_classification(rewritten_text, classification, context, logs)
        try:
            agent_outputs = result.get("logs", {}).get("agent_outputs") or {}
            if (intent or "").startswith("metric") and isinstance(agent_outputs, dict):
                value = agent_outputs.get("data_analyst") or result.get("reply_text", "")
                compact = self._metric_to_telegram(value, context)
                result["formatted_message"] = compact
            else:
                formatted = self._summarize_to_telegram(
                    agent_outputs=agent_outputs if isinstance(agent_outputs, dict) else {},
                    intent=intent or "general",
                    context=context,
                )
                result["formatted_message"] = formatted
        except Exception:
            pass
        try:
            lower = (user_text or "").lower()
            if intent == "workout_change" and ("gravel" in lower or "bike" in lower) and ("tired" in lower or "fatigue" in lower or "recovery" in lower):
                memory_store.add_memory("Preference: prefers light gravel when tired", metadata={"type": "preference", "source": "user_text"})
            if intent == "nutrition_psych_support" and ("overeat" in lower or "bloat" in lower or "guilty" in lower):
                memory_store.add_memory("Health pattern: possible emotional eating after high stress", metadata={"type": "health_pattern", "source": "user_text"})
            if intent == "event_add" and result.get("logs", {}).get("event"):
                ev = result["logs"]["event"]
                memory_store.add_event(ev.get("title"), ev.get("date"))
        except Exception:
            pass
        try:
            if session_id:
                short_memory.set_last_classification(session_id, classification)
                reply_for_memory = result.get("formatted_message") or result.get("reply_text") or ""
                short_memory.add_agent_reply(session_id, reply_for_memory, agent_outputs=result.get("logs", {}).get("agent_outputs"))
        except Exception:
            pass
        result["duration_seconds"] = time.time() - t0
        return result

    # Internal helper so graph wrapper can reuse routing logic
    @traceable(name="pm.route_from_classification", run_type="chain")
    def _route_from_classification(
        self,
        user_text: str,
        classification: Dict[str, Any],
        context: Dict[str, Any] | None = None,
        logs: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        context = context or {}
        logs = logs or {"classification": classification}

        intent = classification.get("intent")
        if intent == "metric_query":
            metric = classification.get("metric", "vo2max_running")
            reply = self.analyst.answer_metric(context, metric)
            agents = [reply.role]
            final_text = reply.content
            logs["agent_outputs"] = {reply.role: reply.content}
        elif intent in {"workout_of_the_day", "daily_workout"}:
            # Auto rest-day detector: if readiness is poor, sleep low, or high acute:chronic load, propose recovery
            try:
                latest = (context.get("latest_health") or {})
                readiness = float(latest.get("training_readiness") or 0)
            except Exception:
                readiness = 0.0
            try:
                sleep_score = float(latest.get("sleep_score") or 0)
            except Exception:
                sleep_score = 0.0
            try:
                tl = context.get("training_load") or []
                last = tl[-1] if isinstance(tl, list) and tl else {}
                acute = float(last.get("acute_load_7d") or 0)
                chronic = float(last.get("chronic_load_28d") or 1)
                ratio = (acute / chronic) if chronic > 0 else 0
            except Exception:
                ratio = 0.0

            rest_trigger = (
                readiness and readiness <= 25
            ) or (
                sleep_score and sleep_score <= 50
            ) or (
                ratio and ratio >= 1.5
            )

            if rest_trigger:
                rec = self.recovery.advise(context, reason=f"readiness={readiness}, sleep={sleep_score}, a/c={ratio:.2f}")
                agents = [rec.role]
                final_text = rec.content
                logs["agent_outputs"] = {rec.role: rec.content}
                return {
                    "reply_text": final_text,
                    "agents": agents,
                    "context": context,
                    "logs": logs,
                    "duration_seconds": 0.0,
                }
            # Choose one sport only (default running) unless user specified
            sport = classification.get("sport") or "running"
            proposals = {}
            if sport == "cycling":
                c = self.cycling.execute_task(context, task="GENERATE_WORKOUT_JSON")
                agents = [c.role]
                proposals = {c.role: c.content}
            elif sport == "strength":
                s = self.strength.execute_task(context, task="GENERATE_WORKOUT_JSON")
                agents = [s.role]
                proposals = {s.role: s.content}
            else:
                r = self.running.execute_task(context, task="GENERATE_WORKOUT_JSON")
                agents = [r.role]
                proposals = {r.role: r.content}
            # Optionally include nutrition/psych short guidance as notes in consolidation, but do not switch sport
            try:
                n = self.nutrition.execute_task(context, task="ANALYZE_DATA_AND_SUMMARIZE", query=None)
                proposals["nutritionist"] = n.content
            except Exception:
                pass
            try:
                p = self.psychology.execute_task(context, task="ANALYZE_DATA_AND_SUMMARIZE", query=None)
                proposals["psychologist"] = p.content
            except Exception:
                pass
            # Finalize single-sport plan
            final_text = self._consolidate(proposals, context, goal="single_workout")
            logs["agent_outputs"] = {**proposals, "consolidated": final_text}
        elif intent in {"event_add", "event_delete", "event_modify"}:
            # Natural language event management
            parsed = self._parse_event_intent(user_text, intent)
            logs["event"] = parsed
            if intent == "event_add" and parsed.get("title") and parsed.get("date"):
                try:
                    eid = memory_store.add_event(parsed["title"], parsed["date"])
                    logs["event"]["id"] = eid
                    final_text = f"Added event: {parsed['title']} on {parsed['date']}"
                except Exception:
                    final_text = "Could not add event."
                agents = []
            elif intent == "event_delete":
                deleted = 0
                try:
                    if parsed.get("id"):
                        deleted = memory_store.delete_event(event_id=parsed["id"])
                    elif parsed.get("delete_target"):
                        deleted = memory_store.delete_event(title_match=parsed["delete_target"]) 
                except Exception:
                    pass
                final_text = f"Deleted {deleted} event(s)." if deleted else "No matching events to delete."
                agents = []
            else:
                final_text = "Event update not implemented."
                agents = []
            logs["agent_outputs"] = {}
        elif intent == "workout_explanation":
            # Honor classifier-suggested agents: gather explanations/opinions from all
            agent_list = classification.get("agents") or []
            outputs: dict[str, str] = {}
            agents: list[str] = []
            try:
                if "nutritionist" in agent_list:
                    n = self.nutrition.execute_task(context, task="PROVIDE_EXPERT_OPINION", query=user_text)
                    outputs[n.role] = n.content
                    agents.append(n.role)
            except Exception:
                pass
            try:
                if "psychologist" in agent_list:
                    p = self.psychology.execute_task(context, task="PROVIDE_EXPERT_OPINION", query=user_text)
                    outputs[p.role] = p.content
                    agents.append(p.role)
            except Exception:
                pass
            try:
                if "running_coach" in agent_list:
                    r = self.running.execute_task(context, task="ANALYZE_DATA_AND_SUMMARIZE", query=user_text)
                    outputs[r.role] = r.content
                    agents.append(r.role)
            except Exception:
                pass
            try:
                if "cycling_coach" in agent_list:
                    c = self.cycling.execute_task(context, task="ANALYZE_DATA_AND_SUMMARIZE", query=user_text)
                    outputs[c.role] = c.content
                    agents.append(c.role)
            except Exception:
                pass
            try:
                if "strength_coach" in agent_list:
                    s = self.strength.execute_task(context, task="ANALYZE_DATA_AND_SUMMARIZE", query=user_text)
                    outputs[s.role] = s.content
                    agents.append(s.role)
            except Exception:
                pass
            if outputs:
                consolidated = self._consolidate(outputs, context, goal="support_synthesis")
                final_text = consolidated or "\n\n".join(v for v in outputs.values() if v)
                logs["agent_outputs"] = {**outputs, "consolidated": final_text}
            else:
                # Fallback single-agent explanation
                preference_hint = ", ".join((context.get("long_term", {}) or {}).get("preferences", [])[:5])
                system = "You are a running coach. Explain briefly why today's plan makes sense given readiness, sleep, load, trends, and preferences."
                try:
                    from .specialists import _chat  # reuse helper
                    summary = {
                        "latest_health": context.get("latest_health", {}),
                        "trends": context.get("trends", {}),
                        "long_term_summary": context.get("long_term_summary", ""),
                    }
                    user = f"Context: {summary}. Prior user: {user_text}. Preferences: {preference_hint}. Keep under 8 lines."
                    content = _chat([{"role": "system", "content": system}, {"role": "user", "content": user}], max_tokens=220)
                except Exception:
                    content = "Based on your recent sleep, readiness, and load, today's session is set at low intensity to support recovery and consistency."
                agents = ["running_coach"]
                final_text = content
                logs["agent_outputs"] = {"running_coach": content}
        elif intent == "workout_change":
            # If the user hints at fatigue, allow rest recommendation
            lower = (user_text or "").lower()
            if any(w in lower for w in ["too tired", "exhausted", "no energy", "rest day", "skip today", "fatigued"]):
                rec = self.recovery.advise(context, reason="user fatigue request")
                agents = [rec.role]
                final_text = rec.content
                logs["agent_outputs"] = {rec.role: rec.content}
                return {
                    "reply_text": final_text,
                    "agents": agents,
                    "context": context,
                    "logs": logs,
                    "duration_seconds": 0.0,
                }
            sport = classification.get("sport", "running")
            pref = classification.get("intensity", "easy")
            if sport == "cycling":
                reply = self.cycling.execute_task(context, task="GENERATE_WORKOUT_JSON", query=user_text, preference=pref)
            else:
                reply = self.running.execute_task(context, task="GENERATE_WORKOUT_JSON", query=user_text, preference=pref)
            # Single workout recommendation
            agents = [reply.role]
            # Normalize to a clean Telegram summary using consolidator for consistency
            consolidated = self._consolidate({reply.role: reply.content}, context, goal="single_workout")
            final_text = consolidated or reply.content
            logs["agent_outputs"] = {reply.role: reply.content, "consolidated": final_text}
        elif intent == "nutrition_psych_support":
            n = self.nutrition.execute_task(context, task="PROVIDE_EXPERT_OPINION", query=user_text)
            p = self.psychology.execute_task(context, task="PROVIDE_EXPERT_OPINION", query=user_text)
            agents = [n.role, p.role]
            consolidated = self._consolidate({n.role: n.content, p.role: p.content}, context, goal="support_synthesis")
            final_text = consolidated or (f"Nutrition:\n{n.content}\n\nPsychology:\n{p.content}")
            logs["agent_outputs"] = {n.role: n.content, p.role: p.content, "consolidated": final_text}
        elif intent == "performance_forecast":
            d = self.analyst.answer_metric(context, metric="trend")
            r = self.running.execute_task(context, task="PROVIDE_EXPERT_OPINION", query=user_text)
            agents = [d.role, r.role]
            consolidated = self._consolidate({d.role: d.content, r.role: r.content}, context, goal="performance_forecast")
            final_text = consolidated or (d.content + "\n\n" + r.content)
            logs["agent_outputs"] = {d.role: d.content, r.role: r.content, "consolidated": final_text}
        elif intent == "plan_request":
            # Collaborative plan synthesis via head coach consolidator
            r = self.running.execute_task(context, task="GENERATE_WORKOUT_JSON")
            s = self.strength.execute_task(context, task="GENERATE_WORKOUT_JSON")
            agents = [r.role, s.role]
            consolidated_json = self._consolidate({r.role: r.content, s.role: s.content}, context, goal="weekly_plan")
            days = classification.get("days") or "N/A"
            final_text = consolidated_json or (f"Training plan for {days} days.\n\n" + r.content + "\n\n" + s.content)
            logs["agent_outputs"] = {r.role: r.content, s.role: s.content, "consolidated": final_text}
        else:
            # General: provide a concise helpful reply using running coach persona
            # If classification suggested multiple agents, honor them and consolidate
            agent_list = classification.get("agents") or []
            outputs: dict[str, str] = {}
            agents: list[str] = []
            if "running_coach" in agent_list:
                rr = self.running.execute_task(context, task="ANALYZE_DATA_AND_SUMMARIZE", query=user_text)
                outputs[rr.role] = rr.content
                agents.append(rr.role)
            if "cycling_coach" in agent_list:
                cc = self.cycling.execute_task(context, task="ANALYZE_DATA_AND_SUMMARIZE", query=user_text)
                outputs[cc.role] = cc.content
                agents.append(cc.role)
            if "strength_coach" in agent_list:
                ss = self.strength.execute_task(context, task="ANALYZE_DATA_AND_SUMMARIZE", query=user_text)
                outputs[ss.role] = ss.content
                agents.append(ss.role)
            if "nutritionist" in agent_list:
                try:
                    nn = self.nutrition.execute_task(context, task="PROVIDE_EXPERT_OPINION", query=user_text)
                    outputs[nn.role] = nn.content
                    agents.append(nn.role)
                except Exception:
                    pass
            if "psychologist" in agent_list:
                try:
                    pp = self.psychology.execute_task(context, task="PROVIDE_EXPERT_OPINION", query=user_text)
                    outputs[pp.role] = pp.content
                    agents.append(pp.role)
                except Exception:
                    pass
            if "data_analyst" in agent_list:
                try:
                    dd = self.analyst.answer_metric(context, metric=classification.get("metric", "trend"))
                    outputs[dd.role] = dd.content
                    agents.append(dd.role)
                except Exception:
                    pass
            if outputs:
                consolidated = self._consolidate(outputs, context, goal="support_synthesis")
                final_text = consolidated or next(iter(outputs.values()))
                logs["agent_outputs"] = {**outputs, "consolidated": final_text}
            else:
                r = self.running.execute_task(context, task="ANALYZE_DATA_AND_SUMMARIZE", query=user_text)
                agents = [r.role]
                consolidated = self._consolidate({r.role: r.content}, context, goal="support_synthesis")
                final_text = consolidated or r.content
                logs["agent_outputs"] = {r.role: r.content, "consolidated": final_text}

        return {
            "reply_text": final_text,
            # formatted_message will be added by caller using _summarize_to_telegram
            "agents": agents,
            "context": context,
            "logs": logs,
            "duration_seconds": 0.0,
        }

    # ---------------- Consensus workflow helpers ----------------
    @traceable(name="pm.consensus_initial_proposals", run_type="chain")
    def consensus_initial_proposals(self, user_text: str) -> Dict[str, Any]:
        from datetime import datetime
        context = data_api.get_rich_context()
        now = datetime.now()
        context["now"] = now.isoformat()
        context["today"] = now.date().isoformat()
        context["weekday"] = now.strftime("%A")
        try:
            events = memory_store.list_events(upcoming_only=True)
        except Exception:
            events = []
        context["upcoming_events"] = events
        context["memory_hits"] = memory_store.search(user_text, top_k=5)
        r = self.running.generate_workout(context)
        c = self.cycling.generate_workout(context)
        s = self.strength.generate_session(context)
        n = self.nutrition.advise(context, concern="support pre/during/post workout fueling and hydration")
        p = self.psychology.advise(context, concern="motivation and mindset for today's training")
        d = self.analyst.answer_metric(context, metric="trend")
        proposals = {r.role: r.content, c.role: c.content, s.role: s.content, n.role: n.content, p.role: p.content, d.role: d.content}
        return {"context": context, "proposals": proposals}

    @traceable(name="pm.consensus_cross_critique", run_type="chain")
    def consensus_cross_critique(self, context: Dict[str, Any], proposals: Dict[str, str]) -> Dict[str, Any]:
        critiques = self._cross_critique(context, proposals)
        return {"critiques": critiques}

    @traceable(name="pm.finalize_daily_consensus", run_type="chain")
    def finalize_daily_consensus(self, context: Dict[str, Any], proposals: Dict[str, str], critiques: Dict[str, str]) -> str:
        return self._head_coach_finalize(context, proposals, critiques)

    def _cross_critique(self, context: Dict[str, Any], proposals: Dict[str, str]) -> Dict[str, str]:
        """Have each specialist critique using its own domain-specific critique method (no generic bias)."""
        critiques: Dict[str, str] = {}
        try:
            # Running critiques cycling/strength/nutrition/psych proposals
            try:
                non_running = {k: v for k, v in proposals.items() if k != "running_coach"}
                cr = self.running.critique(context, non_running)
                critiques[cr.role] = cr.content
            except Exception:
                pass
            # Cycling
            try:
                non_cycling = {k: v for k, v in proposals.items() if k != "cycling_coach"}
                cc = self.cycling.critique(context, non_cycling)
                critiques[cc.role] = cc.content
            except Exception:
                pass
            # Strength
            try:
                non_strength = {k: v for k, v in proposals.items() if k != "strength_coach"}
                sc = self.strength.critique(context, non_strength)
                critiques[sc.role] = sc.content
            except Exception:
                pass
            # Nutrition
            try:
                nn = self.nutrition.critique(context, proposals)
                critiques[nn.role] = nn.content
            except Exception:
                pass
            # Psychology
            try:
                pp = self.psychology.critique(context, proposals)
                critiques[pp.role] = pp.content
            except Exception:
                pass
        except Exception:
            return {}
        return critiques

    def _head_coach_finalize(self, context: Dict[str, Any], proposals: Dict[str, str], critiques: Dict[str, str]) -> str:
        """Head Coach merges proposals and critiques into a single coherent daily plan."""
        # Reuse consolidator but include critiques as additional inputs
        stitched_inputs = {**(proposals or {})}
        for role, text in (critiques or {}).items():
            stitched_inputs[f"critique_{role}"] = text
        return self._consolidate(stitched_inputs, context, goal="single_workout")

    def _format_for_markdown(self, content: str) -> str:
        """Normalize mixed content into concise Telegram Markdown (no HTML, no JSON)."""
        try:
            import re
            # Strip code fences if present
            stripped = content.strip()
            fence_match = re.search(r"```(?:json)?\n([\s\S]*?)```", stripped)
            if fence_match:
                stripped = fence_match.group(1).strip()
            # If looks like JSON, compress to a few lines without braces
            if stripped.startswith("{") or stripped.startswith("["):
                stripped = re.sub(r"[{}\[\]]", "", stripped)
                stripped = re.sub(r"\n+", "\n", stripped)
            # Remove backticks entirely (avoid code blocks in Telegram)
            stripped = stripped.replace("`", "")
            # Hard trim to safe length
            return stripped[:3500]
        except Exception:
            return content[:3500]

    @traceable(name="pm.summarize_to_telegram", run_type="chain")
    def _summarize_to_telegram(self, agent_outputs: Dict[str, str], intent: str, context: Dict[str, Any]) -> str:
        """Summarize multiple agent outputs into a concise natural-language Telegram message.

        - Natural language only; avoid JSON/code blocks
        - Telegram Markdown only (e.g., *bold*, _italic_), emojis allowed
        - Keep length under 3500 chars
        """
        try:
            # Prepare structured hints by parsing any JSON-like agent outputs
            import json as _json
            structured_hints: Dict[str, Dict[str, Any]] = {}
            for role, raw in (agent_outputs or {}).items():
                try:
                    parsed = _json.loads(raw)
                    if isinstance(parsed, dict):
                        keep_keys = {k: parsed.get(k) for k in [
                            "title", "summary", "details", "duration_min", "intensity", "sport"
                        ] if k in parsed}
                        if keep_keys:
                            structured_hints[role] = keep_keys
                except Exception:
                    continue

            if not self.client:
                # Fallback without LLM: choose format by intent
                is_workout_like = intent in {"workout_of_the_day", "daily_workout", "plan_request", "workout_change"}
                if is_workout_like and structured_hints:
                    title = "*Daily Recommendation*"
                    workout_line_items: list[str] = []
                    for preferred in ("running_coach", "cycling_coach", "strength_coach"):
                        if preferred in structured_hints:
                            h = structured_hints[preferred]
                            workout_title = h.get("title") or "Workout of the day"
                            duration = h.get("duration_min")
                            intensity = h.get("intensity")
                            key_bits = []
                            if duration is not None:
                                key_bits.append(f"~{duration} min")
                            if intensity:
                                key_bits.append(str(intensity))
                            header = f"*{workout_title}*" + (f" — {' · '.join(key_bits)}" if key_bits else "")
                            workout_line_items.append(header)
                            details = h.get("details") or h.get("summary")
                            if details:
                                workout_line_items.append(f"- {self._format_for_markdown(details)[:300]}")
                            break
                    msg = title
                    if workout_line_items:
                        msg += "\n\n" + "\n".join(workout_line_items)
                    # Add 1-3 short notes
                    extras: list[str] = []
                    for role, content in (agent_outputs or {}).items():
                        if len(extras) >= 3:
                            break
                        snippet = self._format_for_markdown(content).replace("\n", " ")
                        extras.append(f"- {role.replace('_', ' ').title()}: {snippet[:200]}")
                    if extras:
                        msg += "\n\n" + "\n".join(extras)
                    return msg[:3500]
                # Conversational fallback
                bits: list[str] = []
                for role, content in (agent_outputs or {}).items():
                    snippet = self._format_for_markdown(content).strip()
                    if snippet:
                        bits.append(snippet)
                return ("\n\n".join(bits))[:3500]

            system = (
                "You are a project manager preparing a Telegram answer for an athlete. "
                "Choose the response style based on the user's intent: \n"
                "- If intent is a workout/plan (workout_of_the_day, daily_workout, plan_request, workout_change): format like a Training of the Day (title, ~duration, intensity, main set).\n"
                "- Otherwise (nutrition, psychology, performance forecast, metric_query, general): write a natural, conversational chatbot reply that synthesizes insights. No plan template.\n"
                "Hard constraints: NO JSON, NO code blocks, NO HTML. Use Telegram Markdown only (*bold*, _italic_). Keep under 3500 characters."
            )
            stitched_raw = "\n\n".join([f"[{role}]\n{content}" for role, content in (agent_outputs or {}).items()])
            hints_str = "\n".join(
                [
                    f"{role}: " + ", ".join(
                        [
                            f"{k}={v}" for k, v in fields.items() if v is not None
                        ]
                    )
                    for role, fields in structured_hints.items()
                ]
            )
            user = (
                f"Intent: {intent}.\n"
                f"Parsed hints (if any):\n{hints_str or 'none'}\n\n"
                f"Raw agent outputs:\n{stitched_raw}"
            )
            resp = self.client.chat.completions.create(
                model=settings.openai_model,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                temperature=0.25,
                max_tokens=900,
            )
            msg = resp.choices[0].message.content.strip()
            # Remove backticks / code fences and hard trim
            msg = msg.replace("```", "").replace("`", "")
            if len(msg) > 3500:
                msg = msg[:3500] + "..."
            return msg
        except Exception:
            combined = "\n\n".join(f"{k}: {v}" for k, v in (agent_outputs or {}).items())
            return self._format_for_markdown(combined)

    @traceable(name="pm.metric_to_telegram", run_type="chain")
    def _metric_to_telegram(self, analyst_text: str, context: Dict[str, Any]) -> str:
        """Normalize a metric value reply into a compact message."""
        latest = context.get("latest_health", {})
        def fmt(val):
            try:
                return f"{float(val):.1f}"
            except Exception:
                return str(val)
        lower = analyst_text.lower()
        if "vo2" in lower:
            # Respect modality if clearly indicated in the analyst text
            if "cycl" in lower or "bike" in lower:
                val = latest.get("vo2max_cycling")
                if val is not None:
                    return f"*VO2max (cycling):* {fmt(val)} ml/kg/min"
            if "run" in lower:
                val = latest.get("vo2max_running")
                if val is not None:
                    return f"*VO2max (running):* {fmt(val)} ml/kg/min"
            # Fallback preference: cycling if available else running
            val = latest.get("vo2max_cycling") or latest.get("vo2max_running")
            if val is not None:
                label = "cycling" if latest.get("vo2max_cycling") == val else "running"
                return f"*VO2max ({label}):* {fmt(val)} ml/kg/min"
        # Fallback to analyst text
        return self._format_for_markdown(analyst_text)

    @traceable(name="pm.consolidate", run_type="chain")
    def _consolidate(self, agent_outputs: Dict[str, str], context: Dict[str, Any], goal: str = "single_workout") -> str:
        """Head Coach consolidation: synthesize multiple agent outputs into one coherent plan/message.

        goal: one of "single_workout", "support_synthesis", "performance_forecast", "weekly_plan"
        Returns a natural-language message suitable for Telegram.
        """
        try:
            if not self.client:
                combined = "\n\n".join(f"{k}: {v}" for k, v in (agent_outputs or {}).items())
                return self._format_for_markdown(combined)[:3000]
            system = (
                "You are the Head Coach. Combine specialist inputs into one coherent, non-redundant plan. "
                "Always produce ONE workout for the day when goal=single_workout. When goal=weekly_plan, produce a concise week-by-week plan (bulleted). "
                "When goal=support_synthesis, give 3-5 compact bullets that integrate nutrition + psychology. "
                "When goal=performance_forecast, provide a single clear forecast with rationale. No JSON, Telegram Markdown only."
            )
            stitched = "\n\n".join([f"[{role}]\n{content}" for role, content in (agent_outputs or {}).items()])
            events_hint = ", ".join([f"{e.get('title')} in {e.get('days_until','?')}d" for e in (context.get('upcoming_events') or [])[:3] if isinstance(e, dict)])
            user = (
                f"Goal: {goal}. Today: {context.get('today')}. Weekday: {context.get('weekday')}. Upcoming: {events_hint or 'none'}.\n\n"
                f"Specialists say:\n{stitched}"
            )
            resp = self.client.chat.completions.create(
                model=settings.openai_model,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                temperature=0.2,
                max_tokens=900,
            )
            msg = (resp.choices[0].message.content or "").strip()
            msg = msg.replace("```", "").replace("`", "")
            return msg[:3000]
        except Exception:
            return ""

    def _parse_event_intent(self, user_text: str, intent: str) -> Dict[str, Any]:
        """LLM-assisted event parsing to extract title/date/id from natural language."""
        try:
            if not self.client:
                return {}
            system = (
                "Extract event management parameters from the user's message. Return strict JSON with keys: "
                "title (string), date (ISO date YYYY-MM-DD if present), id (string if provided), delete_target (string fuzzy title if deleting)."
            )
            user = f"Intent: {intent}. Text: {user_text}"
            resp = self.client.chat.completions.create(
                model=settings.openai_model,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                temperature=0,
                max_tokens=180,
            )
            import json as _json
            return _json.loads(resp.choices[0].message.content)
        except Exception:
            return {}


