from __future__ import annotations

from typing import TypedDict, Dict, Any, List

try:
    from langgraph.graph import StateGraph, START, END
except Exception:  # pragma: no cover
    StateGraph = None  # type: ignore
    START = "START"  # type: ignore
    END = "END"  # type: ignore

from typing import TYPE_CHECKING
from agentic_app.agents.contextualizer import QueryContextualizer
from agentic_app.agents.tools import data_api, memory_store, short_memory

if TYPE_CHECKING:  # avoid runtime circular import
    from agentic_app.agents.project_manager import ProjectManagerRouter  # noqa: F401


class BenjaminRouterState(TypedDict, total=False):
    user_text: str
    session_id: str | None
    forced_intent: str | None
    forced_metric: str | None
    short_history: List[Dict[str, str]]
    context_packet: Dict[str, Any]
    contextualized_text: str
    classification: Dict[str, Any]
    intent: str
    context: Dict[str, Any]
    agents: List[str]
    agent_outputs: Dict[str, str]
    reply_text: str
    formatted_message: str
    final_response: str
    logs: Dict[str, Any]
    duration_seconds: float


def build_pm_router_graph(pm: ProjectManagerRouter):
    if StateGraph is None:
        return None

    graph = StateGraph(BenjaminRouterState)
    contextualizer = QueryContextualizer()

    def contextualize_node(state: BenjaminRouterState) -> BenjaminRouterState:
        text = state.get("user_text", "")
        short_hist = state.get("short_history") or []
        packet = contextualizer.build(text, short_history=short_hist, memories_top_k=5)
        state["context_packet"] = packet
        state["contextualized_text"] = packet.get("contextualized_query") or text
        lg = state.setdefault("logs", {})
        lg["raw_user_text"] = text
        lg["contextualized_text"] = state["contextualized_text"]
        lg["context_packet"] = packet
        return state

    def enrich_context_node(state: BenjaminRouterState) -> BenjaminRouterState:
        from datetime import datetime
        try:
            context = data_api.get_rich_context()
        except Exception:
            context = {}
        try:
            context["weather_vincennes"] = data_api.get_weather_vincennes()
        except Exception:
            context["weather_vincennes"] = {}
        now = datetime.now()
        context["now"] = now.isoformat()
        context["today"] = now.date().isoformat()
        context["weekday"] = now.strftime("%A")
        packet = state.get("context_packet", {}) or {}
        try:
            context["contextualized_query"] = state.get("contextualized_text") or state.get("user_text", "")
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
        memory_hits = memory_store.search(state.get("contextualized_text", state.get("user_text", "")), top_k=3)
        try:
            for t in (packet.get("retrieved_memories") or [])[:5]:
                memory_hits.append({"text": t, "source": "contextualizer"})
        except Exception:
            pass
        context["memory_hits"] = memory_hits
        context["short_memory"] = state.get("short_history") or []
        context.setdefault("latest_health", context.get("latest_health", {}))
        state["context"] = context
        return state

    def classify_node(state: BenjaminRouterState) -> BenjaminRouterState:
        user_text = state.get("contextualized_text") or state.get("user_text", "")
        forced_intent = state.get("forced_intent")
        forced_metric = state.get("forced_metric")
        is_followup = (len((user_text or '').split()) <= 12) or any(k in (user_text or '').lower() for k in ["that", "this", "it", "how so", "why", "how did", "explain", "justify"])
        if forced_intent == "workout_of_the_day":
            classification = {"intent": "workout_of_the_day", "agents": ["running_coach", "cycling_coach", "strength_coach", "nutritionist", "psychologist", "data_analyst"]}
        else:
            classification = pm.classify(user_text)
        if forced_intent:
            classification["intent"] = forced_intent
        if forced_metric:
            classification["metric"] = forced_metric
        sid = state.get("session_id")
        if sid and is_followup:
            try:
                prev = short_memory.get_last_classification(sid)
                if prev and (classification.get("intent") in {None, "general"}):
                    classification = {**prev, **{k: v for k, v in classification.items() if v is not None}}
            except Exception:
                pass
        intent = classification.get("intent") or "general"
        if is_followup and any(w in (user_text or '').lower() for w in ["explain", "justify", "how", "why"]):
            intent = "workout_explanation"
            classification["intent"] = intent
        state["classification"] = classification
        state["intent"] = intent
        state.setdefault("logs", {})["classification"] = classification
        return state

    def handle_metric_query(state: BenjaminRouterState) -> BenjaminRouterState:
        metric = state.get("classification", {}).get("metric", "vo2max_running")
        reply = pm.analyst.answer_metric(state.get("context", {}), metric)
        state["agents"] = [reply.role]
        state.setdefault("agent_outputs", {})[reply.role] = reply.content
        state["reply_text"] = reply.content
        return state

    def handle_wod(state: BenjaminRouterState) -> BenjaminRouterState:
        ctx = state.get("context", {})
        latest = (ctx.get("latest_health") or {})
        try:
            readiness = float(latest.get("training_readiness") or 0)
        except Exception:
            readiness = 0.0
        try:
            sleep_score = float(latest.get("sleep_score") or 0)
        except Exception:
            sleep_score = 0.0
        try:
            tl = ctx.get("training_load") or []
            last = tl[-1] if isinstance(tl, list) and tl else {}
            acute = float(last.get("acute_load_7d") or 0)
            chronic = float(last.get("chronic_load_28d") or 1)
            ratio = (acute / chronic) if chronic > 0 else 0
        except Exception:
            ratio = 0.0
        rest_trigger = (readiness and readiness <= 25) or (sleep_score and sleep_score <= 50) or (ratio and ratio >= 1.5)
        if rest_trigger:
            rec = pm.recovery.advise(ctx, reason=f"readiness={readiness}, sleep={sleep_score}, a/c={ratio:.2f}")
            state["agents"] = [rec.role]
            state.setdefault("agent_outputs", {})[rec.role] = rec.content
            state["reply_text"] = rec.content
            return state
        sport = state.get("classification", {}).get("sport") or "running"
        proposals: Dict[str, str] = {}
        agents: List[str] = []
        if sport == "cycling":
            c = pm.cycling.execute_task(ctx, task="GENERATE_WORKOUT_JSON")
            proposals[c.role] = c.content
            agents = [c.role]
        elif sport == "strength":
            s = pm.strength.execute_task(ctx, task="GENERATE_WORKOUT_JSON")
            proposals[s.role] = s.content
            agents = [s.role]
        else:
            r = pm.running.execute_task(ctx, task="GENERATE_WORKOUT_JSON")
            proposals[r.role] = r.content
            agents = [r.role]
        try:
            n = pm.nutrition.execute_task(ctx, task="ANALYZE_DATA_AND_SUMMARIZE", query=None)
            proposals["nutritionist"] = n.content
        except Exception:
            pass
        try:
            p = pm.psychology.execute_task(ctx, task="ANALYZE_DATA_AND_SUMMARIZE", query=None)
            proposals["psychologist"] = p.content
        except Exception:
            pass
        state["agents"] = agents
        state.setdefault("agent_outputs", {}).update(proposals)
        return state

    def handle_event(state: BenjaminRouterState) -> BenjaminRouterState:
        intent = state.get("intent")
        user_text = state.get("contextualized_text") or state.get("user_text", "")
        parsed = pm._parse_event_intent(user_text, intent)
        state.setdefault("logs", {})["event"] = parsed
        reply_text = ""
        if intent == "event_add" and parsed.get("title") and parsed.get("date"):
            try:
                eid = memory_store.add_event(parsed["title"], parsed["date"])
                state["logs"]["event"]["id"] = eid
                reply_text = f"Added event: {parsed['title']} on {parsed['date']}"
            except Exception:
                reply_text = "Could not add event."
        elif intent == "event_delete":
            deleted = 0
            try:
                if parsed.get("id"):
                    deleted = memory_store.delete_event(event_id=parsed["id"])
                elif parsed.get("delete_target"):
                    deleted = memory_store.delete_event(title_match=parsed["delete_target"])
            except Exception:
                pass
            reply_text = f"Deleted {deleted} event(s)." if deleted else "No matching events to delete."
        else:
            reply_text = "Event update not implemented."
        state["reply_text"] = reply_text
        state["agents"] = []
        state.setdefault("agent_outputs", {})
        return state

    def handle_explanation(state: BenjaminRouterState) -> BenjaminRouterState:
        ctx = state.get("context", {})
        user_text = state.get("contextualized_text") or state.get("user_text", "")
        agent_list = state.get("classification", {}).get("agents") or []
        outputs: Dict[str, str] = {}
        agents: List[str] = []
        try:
            if "nutritionist" in agent_list:
                n = pm.nutrition.execute_task(ctx, task="PROVIDE_EXPERT_OPINION", query=user_text)
                outputs[n.role] = n.content
                agents.append(n.role)
        except Exception:
            pass
        try:
            if "psychologist" in agent_list:
                p = pm.psychology.execute_task(ctx, task="PROVIDE_EXPERT_OPINION", query=user_text)
                outputs[p.role] = p.content
                agents.append(p.role)
        except Exception:
            pass
        try:
            if "running_coach" in agent_list:
                r = pm.running.execute_task(ctx, task="ANALYZE_DATA_AND_SUMMARIZE", query=user_text)
                outputs[r.role] = r.content
                agents.append(r.role)
        except Exception:
            pass
        try:
            if "cycling_coach" in agent_list:
                c = pm.cycling.execute_task(ctx, task="ANALYZE_DATA_AND_SUMMARIZE", query=user_text)
                outputs[c.role] = c.content
                agents.append(c.role)
        except Exception:
            pass
        try:
            if "strength_coach" in agent_list:
                s = pm.strength.execute_task(ctx, task="ANALYZE_DATA_AND_SUMMARIZE", query=user_text)
                outputs[s.role] = s.content
                agents.append(s.role)
        except Exception:
            pass
        if outputs:
            state["agent_outputs"] = {**(state.get("agent_outputs") or {}), **outputs}
            state["agents"] = agents
            return state
        try:
            preference_hint = ", ".join((ctx.get("long_term", {}) or {}).get("preferences", [])[:5])
        except Exception:
            preference_hint = ""
        system = "You are a running coach. Explain briefly why today's plan makes sense given readiness, sleep, load, trends, and preferences."
        try:
            from .specialists import _chat
            summary = {
                "latest_health": ctx.get("latest_health", {}),
                "trends": ctx.get("trends", {}),
                "long_term_summary": ctx.get("long_term_summary", ""),
            }
            user = f"Context: {summary}. Prior user: {user_text}. Preferences: {preference_hint}. Keep under 8 lines."
            content = _chat([{"role": "system", "content": system}, {"role": "user", "content": user}], max_tokens=220)
        except Exception:
            content = "Based on your recent sleep, readiness, and load, today's session is set at low intensity to support recovery and consistency."
        state["agents"] = ["running_coach"]
        state.setdefault("agent_outputs", {})["running_coach"] = content
        state["reply_text"] = content
        return state

    def handle_workout_change(state: BenjaminRouterState) -> BenjaminRouterState:
        ctx = state.get("context", {})
        user_text = state.get("contextualized_text") or state.get("user_text", "")
        lower = (user_text or "").lower()
        if any(w in lower for w in ["too tired", "exhausted", "no energy", "rest day", "skip today", "fatigued"]):
            rec = pm.recovery.advise(ctx, reason="user fatigue request")
            state["agents"] = [rec.role]
            state.setdefault("agent_outputs", {})[rec.role] = rec.content
            state["reply_text"] = rec.content
            return state
        sport = state.get("classification", {}).get("sport", "running")
        pref = state.get("classification", {}).get("intensity", "easy")
        if sport == "cycling":
            reply = pm.cycling.execute_task(ctx, task="GENERATE_WORKOUT_JSON", query=user_text, preference=pref)
        else:
            reply = pm.running.execute_task(ctx, task="GENERATE_WORKOUT_JSON", query=user_text, preference=pref)
        state["agents"] = [reply.role]
        state.setdefault("agent_outputs", {})[reply.role] = reply.content
        return state

    def handle_nutrition_psych(state: BenjaminRouterState) -> BenjaminRouterState:
        ctx = state.get("context", {})
        user_text = state.get("contextualized_text") or state.get("user_text", "")
        n = pm.nutrition.execute_task(ctx, task="PROVIDE_EXPERT_OPINION", query=user_text)
        p = pm.psychology.execute_task(ctx, task="PROVIDE_EXPERT_OPINION", query=user_text)
        state["agents"] = [n.role, p.role]
        state.setdefault("agent_outputs", {})[n.role] = n.content
        state.setdefault("agent_outputs", {})[p.role] = p.content
        return state

    def handle_performance_forecast(state: BenjaminRouterState) -> BenjaminRouterState:
        ctx = state.get("context", {})
        d = pm.analyst.answer_metric(ctx, metric="trend")
        r = pm.running.execute_task(ctx, task="PROVIDE_EXPERT_OPINION", query=state.get("contextualized_text") or state.get("user_text", ""))
        state["agents"] = [d.role, r.role]
        state.setdefault("agent_outputs", {})[d.role] = d.content
        state.setdefault("agent_outputs", {})[r.role] = r.content
        return state

    def handle_plan_request(state: BenjaminRouterState) -> BenjaminRouterState:
        ctx = state.get("context", {})
        r = pm.running.execute_task(ctx, task="GENERATE_WORKOUT_JSON")
        s = pm.strength.execute_task(ctx, task="GENERATE_WORKOUT_JSON")
        state["agents"] = [r.role, s.role]
        state.setdefault("agent_outputs", {})[r.role] = r.content
        state.setdefault("agent_outputs", {})[s.role] = s.content
        return state

    def handle_general(state: BenjaminRouterState) -> BenjaminRouterState:
        ctx = state.get("context", {})
        user_text = state.get("contextualized_text") or state.get("user_text", "")
        agent_list = state.get("classification", {}).get("agents") or []
        outputs: Dict[str, str] = {}
        agents: List[str] = []
        if "running_coach" in agent_list:
            rr = pm.running.execute_task(ctx, task="ANALYZE_DATA_AND_SUMMARIZE", query=user_text)
            outputs[rr.role] = rr.content
            agents.append(rr.role)
        if "cycling_coach" in agent_list:
            cc = pm.cycling.execute_task(ctx, task="ANALYZE_DATA_AND_SUMMARIZE", query=user_text)
            outputs[cc.role] = cc.content
            agents.append(cc.role)
        if "strength_coach" in agent_list:
            ss = pm.strength.execute_task(ctx, task="ANALYZE_DATA_AND_SUMMARIZE", query=user_text)
            outputs[ss.role] = ss.content
            agents.append(ss.role)
        if "nutritionist" in agent_list:
            try:
                nn = pm.nutrition.execute_task(ctx, task="PROVIDE_EXPERT_OPINION", query=user_text)
                outputs[nn.role] = nn.content
                agents.append(nn.role)
            except Exception:
                pass
        if "psychologist" in agent_list:
            try:
                pp = pm.psychology.execute_task(ctx, task="PROVIDE_EXPERT_OPINION", query=user_text)
                outputs[pp.role] = pp.content
                agents.append(pp.role)
            except Exception:
                pass
        if "data_analyst" in agent_list:
            try:
                dd = pm.analyst.answer_metric(ctx, metric=state.get("classification", {}).get("metric", "trend"))
                outputs[dd.role] = dd.content
                agents.append(dd.role)
            except Exception:
                pass
        if outputs:
            state["agent_outputs"] = {**(state.get("agent_outputs") or {}), **outputs}
            state["agents"] = agents
            return state
        r = pm.running.execute_task(ctx, task="ANALYZE_DATA_AND_SUMMARIZE", query=user_text)
        state["agents"] = [r.role]
        state.setdefault("agent_outputs", {})[r.role] = r.content
        return state

    def consolidate_node(state: BenjaminRouterState) -> BenjaminRouterState:
        outs = state.get("agent_outputs") or {}
        goal = "single_workout"
        intent = state.get("intent") or "general"
        if intent in {"support_synthesis", "nutrition_psych_support", "general", "workout_explanation"}:
            goal = "support_synthesis"
        elif intent == "performance_forecast":
            goal = "performance_forecast"
        elif intent == "plan_request":
            goal = "weekly_plan"
        consolidated = pm._consolidate(outs if isinstance(outs, dict) else {}, state.get("context", {}), goal=goal)
        if consolidated:
            state["reply_text"] = consolidated
        return state

    def summarize_node(state: BenjaminRouterState) -> BenjaminRouterState:
        intent = state.get("intent") or "general"
        if intent.startswith("metric"):
            msg = pm._metric_to_telegram(state.get("reply_text") or "", state.get("context", {}))
        else:
            outs = state.get("agent_outputs") or {}
            msg = pm._summarize_to_telegram(outs if isinstance(outs, dict) else {}, intent=intent, context=state.get("context", {}))
        state["formatted_message"] = msg
        state["final_response"] = msg
        return state

    graph.add_node("contextualize", contextualize_node)
    graph.add_node("enrich_context", enrich_context_node)
    graph.add_node("classify", classify_node)
    graph.add_node("handle_metric_query", handle_metric_query)
    graph.add_node("handle_wod", handle_wod)
    graph.add_node("handle_event", handle_event)
    graph.add_node("handle_explanation", handle_explanation)
    graph.add_node("handle_workout_change", handle_workout_change)
    graph.add_node("handle_nutrition_psych", handle_nutrition_psych)
    graph.add_node("handle_performance_forecast", handle_performance_forecast)
    graph.add_node("handle_plan_request", handle_plan_request)
    graph.add_node("handle_general", handle_general)
    graph.add_node("consolidate", consolidate_node)
    graph.add_node("summarize", summarize_node)

    graph.add_edge(START, "contextualize")
    graph.add_edge("contextualize", "enrich_context")
    graph.add_edge("enrich_context", "classify")

    def route_intent(state: BenjaminRouterState) -> str:
        it = (state.get("intent") or "general").lower()
        if it == "metric_query":
            return "handle_metric_query"
        if it in {"workout_of_the_day", "daily_workout"}:
            return "handle_wod"
        if it in {"event_add", "event_delete", "event_modify"}:
            return "handle_event"
        if it == "workout_explanation":
            return "handle_explanation"
        if it == "workout_change":
            return "handle_workout_change"
        if it == "nutrition_psych_support":
            return "handle_nutrition_psych"
        if it == "performance_forecast":
            return "handle_performance_forecast"
        if it == "plan_request":
            return "handle_plan_request"
        return "handle_general"

    graph.add_conditional_edges("classify", route_intent, {
        "handle_metric_query": "handle_metric_query",
        "handle_wod": "handle_wod",
        "handle_event": "handle_event",
        "handle_explanation": "handle_explanation",
        "handle_workout_change": "handle_workout_change",
        "handle_nutrition_psych": "handle_nutrition_psych",
        "handle_performance_forecast": "handle_performance_forecast",
        "handle_plan_request": "handle_plan_request",
        "handle_general": "handle_general",
    })

    def needs_consolidation(state: BenjaminRouterState) -> str:
        intent = state.get("intent") or "general"
        if intent in {"metric_query", "event_add", "event_delete", "event_modify"}:
            return "summarize"
        outs = state.get("agent_outputs") or {}
        if not outs or (len(outs) <= 1 and intent in {"workout_change", "workout_of_the_day", "daily_workout"} and (state.get("reply_text") or "")):
            return "summarize"
        return "consolidate"

    for handler in [
        "handle_metric_query", "handle_wod", "handle_event", "handle_explanation",
        "handle_workout_change", "handle_nutrition_psych", "handle_performance_forecast",
        "handle_plan_request", "handle_general",
    ]:
        graph.add_conditional_edges(handler, needs_consolidation, {
            "summarize": "summarize",
            "consolidate": "consolidate",
        })

    graph.add_edge("consolidate", "summarize")
    graph.add_edge("summarize", END)

    return graph.compile()

class RouterState(TypedDict, total=False):
    user_text: str
    contextualized_text: str
    classification: Dict[str, Any]
    reply_text: str
    agents: List[str]
    logs: Dict[str, Any]
    duration_seconds: float
    context: Dict[str, Any]
    proposals: Dict[str, str]
    critiques: Dict[str, str]


def build_router_graph(pm: ProjectManagerRouter):
    # New LangGraph-based router with explicit state and intent nodes
    if StateGraph is None:
        return None

    graph = StateGraph(BenjaminRouterState)  # type: ignore[name-defined]
    contextualizer = QueryContextualizer()

    def contextualize_node(state: "BenjaminRouterState") -> "BenjaminRouterState":  # type: ignore[name-defined]
        text = state.get("user_text", "")
        short_hist = state.get("short_history") or []
        packet = contextualizer.build(text, short_history=short_hist, memories_top_k=5)
        state["context_packet"] = packet
        state["contextualized_text"] = packet.get("contextualized_query") or text
        lg = state.setdefault("logs", {})
        lg["raw_user_text"] = text
        lg["contextualized_text"] = state["contextualized_text"]
        lg["context_packet"] = packet
        return state

    def enrich_context_node(state: "BenjaminRouterState") -> "BenjaminRouterState":  # type: ignore[name-defined]
        from datetime import datetime
        try:
            context = data_api.get_rich_context()
        except Exception:
            context = {}
        try:
            context["weather_vincennes"] = data_api.get_weather_vincennes()
        except Exception:
            context["weather_vincennes"] = {}
        now = datetime.now()
        context["now"] = now.isoformat()
        context["today"] = now.date().isoformat()
        context["weekday"] = now.strftime("%A")
        packet = state.get("context_packet", {}) or {}
        try:
            context["contextualized_query"] = state.get("contextualized_text") or state.get("user_text", "")
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
        memory_hits = memory_store.search(state.get("contextualized_text", state.get("user_text", "")), top_k=3)
        try:
            for t in (packet.get("retrieved_memories") or [])[:5]:
                memory_hits.append({"text": t, "source": "contextualizer"})
        except Exception:
            pass
        context["memory_hits"] = memory_hits
        context["short_memory"] = state.get("short_history") or []
        context.setdefault("latest_health", context.get("latest_health", {}))
        state["context"] = context
        return state

    def classify_node(state: "BenjaminRouterState") -> "BenjaminRouterState":  # type: ignore[name-defined]
        user_text = state.get("contextualized_text") or state.get("user_text", "")
        forced_intent = state.get("forced_intent")
        forced_metric = state.get("forced_metric")
        is_followup = (len((user_text or '').split()) <= 12) or any(k in (user_text or '').lower() for k in ["that", "this", "it", "how so", "why", "how did", "explain", "justify"])
        if forced_intent == "workout_of_the_day":
            classification = {"intent": "workout_of_the_day", "agents": ["running_coach", "cycling_coach", "strength_coach", "nutritionist", "psychologist", "data_analyst"]}
        else:
            classification = pm.classify(user_text)
        if forced_intent:
            classification["intent"] = forced_intent
        if forced_metric:
            classification["metric"] = forced_metric
        sid = state.get("session_id")
        if sid and is_followup:
            try:
                prev = short_memory.get_last_classification(sid)
                if prev and (classification.get("intent") in {None, "general"}):
                    classification = {**prev, **{k: v for k, v in classification.items() if v is not None}}
            except Exception:
                pass
        intent = classification.get("intent") or "general"
        if is_followup and any(w in (user_text or '').lower() for w in ["explain", "justify", "how", "why"]):
            intent = "workout_explanation"
            classification["intent"] = intent
        state["classification"] = classification
        state["intent"] = intent
        state.setdefault("logs", {})["classification"] = classification
        return state

    # Intent handler and utility nodes will be defined below; we first wire the scaffolding
    graph.add_node("contextualize", contextualize_node)
    graph.add_node("enrich_context", enrich_context_node)
    graph.add_node("classify", classify_node)
    # Placeholder nodes to be referenced later
    def _noop(state):
        return state
    for n in [
        "handle_metric_query", "handle_wod", "handle_event", "handle_explanation",
        "handle_workout_change", "handle_nutrition_psych", "handle_performance_forecast",
        "handle_plan_request", "handle_general", "consolidate", "summarize",
    ]:
        graph.add_node(n, _noop)

    graph.add_edge(START, "contextualize")
    graph.add_edge("contextualize", "enrich_context")
    graph.add_edge("enrich_context", "classify")
    # Conditional edges will be added later after real node defs are in place
    # We'll return a partially built graph for now; a second builder below will override nodes
    return graph


def build_collaboration_graph(pm: ProjectManagerRouter):
    """Graph that, given a classification with multiple agents, runs them and consolidates."""
    if StateGraph is None:
        return None

    graph = StateGraph(RouterState)

    def classify_node(state: RouterState) -> RouterState:
        cls = pm.classify(state.get("user_text", ""))
        state["classification"] = cls
        state.setdefault("logs", {})["classification"] = cls
        return state

    def run_agents_node(state: RouterState) -> RouterState:
        # Build minimal context
        res = pm._route_from_classification(state.get("user_text", ""), state.get("classification", {}))
        state["logs"] = res.get("logs", {})
        state["agents"] = res.get("agents", [])
        state["context"] = res.get("context", {})
        return state

    def consolidate_node(state: RouterState) -> RouterState:
        outs = (state.get("logs", {}) or {}).get("agent_outputs") or {}
        consolidated = pm._consolidate(outs if isinstance(outs, dict) else {}, state.get("context", {}), goal="single_workout")
        state["reply_text"] = consolidated or ""
        return state

    graph.add_node("classify", classify_node)
    graph.add_node("run_agents", run_agents_node)
    graph.add_node("consolidate", consolidate_node)
    graph.add_edge(START, "classify")
    graph.add_edge("classify", "run_agents")
    graph.add_edge("run_agents", "consolidate")
    graph.add_edge("consolidate", END)

    return graph.compile()


def build_daily_consensus_graph(pm: ProjectManagerRouter):
    """LangGraph for Workout of the Day consensus with 3 rounds.

    Round 1: Initial proposals by specialists
    Round 2: Cross-critique among specialists
    Round 3: Head Coach final revision
    """
    if StateGraph is None:
        return None

    graph = StateGraph(RouterState)

    def init_context_node(state: RouterState) -> RouterState:
        # pm will enrich context internally; here we just pass through user_text
        state.setdefault("logs", {})
        return state

    def proposals_node(state: RouterState) -> RouterState:
        res = pm.consensus_initial_proposals(state.get("user_text", ""))
        state["context"] = res.get("context", {})
        state["proposals"] = res.get("proposals", {})
        # Persist into logs
        logs = state.setdefault("logs", {})
        logs.setdefault("agent_outputs", {})
        logs["agent_outputs"].update({k: v for k, v in (state["proposals"] or {}).items()})
        state["agents"] = list((state["proposals"] or {}).keys())
        return state

    def critiques_node(state: RouterState) -> RouterState:
        crit = pm.consensus_cross_critique(state.get("context", {}), state.get("proposals", {}))
        state["critiques"] = crit.get("critiques", {})
        # Append to logs
        logs = state.setdefault("logs", {})
        logs.setdefault("agent_outputs", {})
        # Prefix critique keys to distinguish
        for role, content in (state["critiques"] or {}).items():
            logs["agent_outputs"][f"critique_{role}"] = content
        return state

    def finalize_node(state: RouterState) -> RouterState:
        final = pm.finalize_daily_consensus(state.get("context", {}), state.get("proposals", {}), state.get("critiques", {}))
        state["reply_text"] = final or ""
        return state

    graph.add_node("init_context", init_context_node)
    graph.add_node("proposals", proposals_node)
    graph.add_node("critiques", critiques_node)
    graph.add_node("finalize", finalize_node)

    graph.add_edge(START, "init_context")
    graph.add_edge("init_context", "proposals")
    graph.add_edge("proposals", "critiques")
    graph.add_edge("critiques", "finalize")
    graph.add_edge("finalize", END)

    return graph.compile()


