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
from agentic_app.agents.data_analyst_tool import data_analyst_tool

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
    # For sequential agent chaining
    current_agent_index: int
    accumulated_context: Dict[str, Any]


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
        
        # SESSION CONTEXT FIX: Ensure full conversation history is available
        short_history = state.get("short_history") or []
        context["short_memory"] = short_history
        
        # Add session-based conversation context for agents
        if short_history:
            conversation_context = []
            for turn in short_history[-6:]:  # Last 6 turns for context
                role = turn.get("role", "")
                text = turn.get("text", "")
                if role and text:
                    conversation_context.append(f"{role.title()}: {text}")
            
            if conversation_context:
                context["conversation_history"] = "\n".join(conversation_context)
                context["has_conversation_context"] = True
        else:
            context["conversation_history"] = ""
            context["has_conversation_context"] = False
        
        context.setdefault("latest_health", context.get("latest_health", {}))
        state["context"] = context
        return state

    def classify_node(state: BenjaminRouterState) -> BenjaminRouterState:
        user_text = state.get("contextualized_text") or state.get("user_text", "")
        forced_intent = state.get("forced_intent")
        forced_metric = state.get("forced_metric")
        is_followup = (len((user_text or '').split()) <= 12) or any(k in (user_text or '').lower() for k in ["that", "this", "it", "how so", "why", "how did", "explain", "justify"])
        if forced_intent == "workout_of_the_day":
            classification = {"intent": "workout_of_the_day", "agents": ["running_coach", "cycling_coach", "strength_coach", "nutritionist", "psychologist"]}
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
        
        # NEW: Handle sequential agent chaining
        # The agents list from classification is now treated as an ordered sequence
        agents_list = classification.get("agents", [])
        if isinstance(agents_list, list) and len(agents_list) > 0:
            state["agents"] = agents_list.copy()  # Use agents as the sequence
            state["current_agent_index"] = 0
            state["accumulated_context"] = {}
        
        return state

    def handle_metric_query(state: BenjaminRouterState) -> BenjaminRouterState:
        metric = state.get("classification", {}).get("metric", "vo2max_running")
        ctx = state.get("context", {})
        
        # Use data analyst tool to get metric data
        from agentic_app.agents.data_analyst_tool import data_analyst_tool
        metric_data = data_analyst_tool.get_metric(ctx, metric)
        
        # Format response
        if metric_data["found"]:
            value_str = f"{metric_data['value']}"
            if metric_data["unit"]:
                value_str += f" {metric_data['unit']}"
            reply_text = f"Latest {metric_data['label']}: {value_str}"
        else:
            # Try getting fitness trends if specific metric not found
            if metric.lower() in ["trend", "overall fitness trend", "fitness trend", "overall fitness"]:
                trends = data_analyst_tool.get_fitness_trends(ctx)
                parts = []
                if "vo2max_running" in trends:
                    parts.append(f"VO2max(run): {trends['vo2max_running']:.1f}")
                if "vo2max_cycling" in trends:
                    parts.append(f"VO2max(bike): {trends['vo2max_cycling']:.1f}")
                if "training_load" in trends:
                    tl = trends["training_load"]
                    parts.append(f"Load 7d/28d: {tl['acute_7d']}/{tl['chronic_28d']}")
                if "weight_change_60d" in trends:
                    parts.append(f"Weight 60d Δ: {trends['weight_change_60d']:+.1f} kg")
                if "activities_14d" in trends:
                    parts.append(f"Activities last 14d: {trends['activities_14d']}")
                reply_text = ", ".join(parts) if parts else "I could not find enough data to compute an overall fitness trend."
            else:
                reply_text = f"I could not find {metric.replace('_', ' ')} in the latest data."
        
        state["agents"] = ["data_tool"]
        state.setdefault("agent_outputs", {})["data_tool"] = reply_text
        state["reply_text"] = reply_text
        state["formatted_message"] = reply_text
        return state

    def handle_wod(state: BenjaminRouterState) -> BenjaminRouterState:
        ctx = state.get("context", {})
        
        # Step 1: Assess Readiness (Gatekeeper)
        readiness_assessment = pm.recovery.assess_readiness(ctx)
        ctx["readiness_assessment"] = readiness_assessment
        
        # Check if rest is required
        if readiness_assessment.get("status") == "rest_required":
            rec = pm.recovery.advise(ctx, reason=readiness_assessment.get("notes", "Readiness assessment indicates rest needed"))
            state["agents"] = [rec.role]
            state.setdefault("agent_outputs", {})[rec.role] = rec.content
            state["reply_text"] = rec.content
            state["formatted_message"] = rec.content
            return state
        
        # Step 2: Assess Overall Fitness
        fitness_summary = pm.overall_fitness.assess_fitness(ctx)
        ctx["fitness_summary"] = fitness_summary
        
        # Step 3: Create streamlined context for main agents
        streamlined_ctx = {
            "now": ctx.get("now"),
            "today": ctx.get("today"),
            "weekday": ctx.get("weekday"),
            "weather_vincennes": ctx.get("weather_vincennes"),
            "upcoming_events": ctx.get("upcoming_events"),
            "latest_health": ctx.get("latest_health"),
            "readiness_assessment": readiness_assessment,
            "fitness_summary": fitness_summary,
            "contextualized_query": ctx.get("contextualized_query"),
            "memory_hits": ctx.get("memory_hits", [])[:3],
        }
        sport = state.get("classification", {}).get("sport") or "running"
        proposals: Dict[str, str] = {}
        agents: List[str] = []
        if sport == "cycling":
            c = pm.cycling.execute_task(streamlined_ctx, task="GENERATE_WORKOUT_JSON")
            proposals[c.role] = c.content
            agents = [c.role]
        elif sport == "strength":
            s = pm.strength.execute_task(streamlined_ctx, task="GENERATE_WORKOUT_JSON")
            proposals[s.role] = s.content
            agents = [s.role]
        else:
            r = pm.running.execute_task(streamlined_ctx, task="GENERATE_WORKOUT_JSON")
            proposals[r.role] = r.content
            agents = [r.role]
        try:
            n = pm.nutrition.execute_task(streamlined_ctx, task="ANALYZE_DATA_AND_SUMMARIZE", query=None)
            proposals["nutritionist"] = n.content
        except Exception:
            pass
        try:
            p = pm.psychology.execute_task(streamlined_ctx, task="ANALYZE_DATA_AND_SUMMARIZE", query=None)
            proposals["psychologist"] = p.content
        except Exception:
            pass
        # Recovery advisor already reviewed in pre-processing, but can add final comments
        try:
            rec = pm.recovery.advise(streamlined_ctx, reason="final review of workout plan")
            proposals["recovery_advisor"] = rec.content
            agents.append(rec.role)
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
        state["formatted_message"] = reply_text
        state["agents"] = []
        state.setdefault("agent_outputs", {})
        return state

    def handle_explanation(state: BenjaminRouterState) -> BenjaminRouterState:
        """Handle workout_explanation with CONVERSATIONAL MODE - single agent, no consolidation."""
        ctx = state.get("context", {})
        user_text = state.get("contextualized_text") or state.get("user_text", "")
        agent_list = state.get("classification", {}).get("agents") or []
        
        # DEBUGGING: Track agent execution
        agent_sequence = []
        state.setdefault("logs", {})["agent_sequence"] = agent_sequence
        
        # CONVERSATIONAL MODE: Select single most relevant specialist
        # Determine best specialist based on query content
        lower_text = (user_text or "").lower()
        
        if any(word in lower_text for word in ["nutrition", "food", "eat", "diet", "meal", "hungry", "craving"]):
            specialist = "nutritionist"
        elif any(word in lower_text for word in ["mental", "psychology", "motivation", "stress", "anxious", "confidence"]):
            specialist = "psychologist"
        elif any(word in lower_text for word in ["recovery", "sleep", "rest", "tired", "fatigue", "sore"]):
            specialist = "recovery_advisor"
        elif any(word in lower_text for word in ["strength", "gym", "weights", "lifting", "muscle"]):
            specialist = "strength_coach"
        elif any(word in lower_text for word in ["bike", "cycling", "ride", "gravel"]):
            specialist = "cycling_coach"
        elif agent_list:
            # Use classifier suggestion if available
            specialist = agent_list[0]
        else:
            specialist = "running_coach"  # Default
        
        # Execute single specialist with conversational task
        try:
            if specialist == "nutritionist":
                agent_sequence.append("🥗 Nutritionist (Conversational)")
                agent_output = pm.nutrition.execute_task(ctx, task="ANSWER_USER_QUESTION", query=user_text)
            elif specialist == "psychologist":
                agent_sequence.append("🧠 Psychologist (Conversational)")
                agent_output = pm.psychology.execute_task(ctx, task="ANSWER_USER_QUESTION", query=user_text)
            elif specialist == "recovery_advisor":
                agent_sequence.append("😴 Recovery Advisor (Conversational)")
                agent_output = pm.recovery.execute_task(ctx, task="ANSWER_USER_QUESTION", query=user_text)
            elif specialist == "strength_coach":
                agent_sequence.append("💪 Strength Coach (Conversational)")
                agent_output = pm.strength.execute_task(ctx, task="ANSWER_USER_QUESTION", query=user_text)
            elif specialist == "cycling_coach":
                agent_sequence.append("🚴 Cycling Coach (Conversational)")
                agent_output = pm.cycling.execute_task(ctx, task="ANSWER_USER_QUESTION", query=user_text)
            else:
                agent_sequence.append("🏃 Running Coach (Conversational)")
                agent_output = pm.running.execute_task(ctx, task="ANSWER_USER_QUESTION", query=user_text)
            
            state["agents"] = [agent_output.role]
            state.setdefault("agent_outputs", {})[agent_output.role] = agent_output.content
            state["reply_text"] = agent_output.content
            state["formatted_message"] = agent_output.content
            
            # DEBUGGING: Ensure specialist output is captured
            state.setdefault("logs", {})["specialist_debug"] = {
                "role": agent_output.role,
                "content_length": len(agent_output.content),
                "content_preview": agent_output.content[:100] + "..." if len(agent_output.content) > 100 else agent_output.content,
            }
            
        except Exception:
            # Fallback
            agent_sequence.append("🏃 Running Coach (Fallback)")
            content = "I'd be happy to help explain that. Could you provide a bit more context about what specifically you'd like me to explain?"
            state["agents"] = ["running_coach"]
            state.setdefault("agent_outputs", {})["running_coach"] = content
            state["reply_text"] = content
            state["formatted_message"] = content
        
        # Mark as conversational to bypass consolidation
        state.setdefault("logs", {})["is_conversational"] = True
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
        # Include recovery advisor for all workout changes
        try:
            rec = pm.recovery.advise(ctx, reason="reviewing modified workout plan")
            state.setdefault("agent_outputs", {})["recovery_advisor"] = rec.content
            state["agents"].append(rec.role)
        except Exception:
            pass
        return state

    def handle_nutrition_psych(state: BenjaminRouterState) -> BenjaminRouterState:
        ctx = state.get("context", {})
        user_text = state.get("contextualized_text") or state.get("user_text", "")
        
        # For single-specialist requests, route to most appropriate specialist
        lower_text = (user_text or "").lower()
        needs_both = any(phrase in lower_text for phrase in [
            "emotional eating", "stress eating", "anxiety about food", 
            "mental relationship with food", "psychological eating",
            "food and mood", "eating behavior", "nutrition for mental health"
        ])
        
        if needs_both:
            # Genuine multi-specialist request
            n = pm.nutrition.execute_task(ctx, task="PROVIDE_EXPERT_OPINION", query=user_text)
            p = pm.psychology.execute_task(ctx, task="PROVIDE_EXPERT_OPINION", query=user_text)
            state["agents"] = [n.role, p.role]
            state.setdefault("agent_outputs", {})[n.role] = n.content
            state.setdefault("agent_outputs", {})[p.role] = p.content
        else:
            # Single specialist request - use smart selection
            if any(phrase in lower_text for phrase in [
                "shopping list", "meal plan", "what to eat", "food prep",
                "grocery", "recipe", "cooking", "nutrition"
            ]):
                specialist = "nutritionist"
                agent_output = pm.nutrition.execute_task(ctx, task="ANSWER_USER_QUESTION", query=user_text)
            else:
                specialist = "psychologist"
                agent_output = pm.psychology.execute_task(ctx, task="ANSWER_USER_QUESTION", query=user_text)
            
            state["agents"] = [agent_output.role]
            state.setdefault("agent_outputs", {})[agent_output.role] = agent_output.content
            state["reply_text"] = agent_output.content
            state["formatted_message"] = agent_output.content
            # Mark as conversational to bypass consolidation
            state.setdefault("logs", {})["is_conversational"] = True
        
        return state

    def handle_performance_forecast(state: BenjaminRouterState) -> BenjaminRouterState:
        ctx = state.get("context", {})
        # Use data_analyst_tool instead of removed analyst agent
        trends_data = data_analyst_tool.get_fitness_trends(ctx)
        # Create a response-like object for consistency
        from types import SimpleNamespace
        d = SimpleNamespace()
        d.role = "data_analyst"
        d.content = f"Fitness trends analysis: {trends_data}"
        
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
        # Include recovery advisor for all training plans
        try:
            rec = pm.recovery.advise(ctx, reason="reviewing training plan for recovery optimization")
            state.setdefault("agent_outputs", {})["recovery_advisor"] = rec.content
            state["agents"].append(rec.role)
        except Exception:
            pass
        return state

    def sequential_agent_execution(state: BenjaminRouterState) -> BenjaminRouterState:
        """Execute agents sequentially, passing outputs as context to next agent."""
        # Get the original agent sequence from classification
        classification = state.get("classification", {})
        original_agents = classification.get("agents", [])
        current_index = state.get("current_agent_index", 0)
        
        if not original_agents or current_index >= len(original_agents):
            return state
        
        ctx = state.get("context", {}).copy()
        user_text = state.get("contextualized_text") or state.get("user_text", "")
        intent = state.get("intent", "general")
        
        # Apply preprocessing pipeline for workout-related intents (only on first agent)
        if current_index == 0 and intent in ["workout_of_the_day", "daily_workout", "workout_change", "plan_request"]:
            # Convert agents list to track executed agents (start with empty, will add as we execute)
            state["agents"] = []
                
            # Step 1: Assess Readiness (Gatekeeper)
            readiness_assessment = pm.recovery.assess_readiness(ctx)
            ctx["readiness_assessment"] = readiness_assessment
            state.setdefault("accumulated_context", {})["readiness_assessment"] = readiness_assessment
            # Log preprocessing agent execution
            state["agents"].append("recovery_advisor_preprocessing")
            
            # Check if rest is required
            if readiness_assessment.get("status") == "rest_required":
                rec = pm.recovery.advise(ctx, reason=readiness_assessment.get("notes", "Readiness assessment indicates rest needed"))
                state["agents"].append(rec.role)
                state.setdefault("agent_outputs", {})[rec.role] = rec.content
                state["reply_text"] = rec.content
                # Skip the rest of the sequence
                state["current_agent_index"] = len(original_agents)
                return state
            
            # Step 2: Assess Overall Fitness
            fitness_summary = pm.overall_fitness.assess_fitness(ctx)
            ctx["fitness_summary"] = fitness_summary
            state["accumulated_context"]["fitness_summary"] = fitness_summary
            # Log preprocessing agent execution
            state["agents"].append("overall_fitness_preprocessing")
            
            # Step 3: Create streamlined context
            streamlined_ctx = {
                "now": ctx.get("now"),
                "today": ctx.get("today"),
                "weekday": ctx.get("weekday"),
                "weather_vincennes": ctx.get("weather_vincennes"),
                "upcoming_events": ctx.get("upcoming_events"),
                "latest_health": ctx.get("latest_health"),
                "readiness_assessment": readiness_assessment,
                "fitness_summary": fitness_summary,
                "contextualized_query": ctx.get("contextualized_query"),
                "memory_hits": ctx.get("memory_hits", [])[:3],
            }
            # Use streamlined context for workout generation
            ctx = streamlined_ctx
        
        current_agent = original_agents[current_index]
        
        # For subsequent agents in workout intents, use streamlined context if available
        if current_index > 0 and intent in ["workout_of_the_day", "daily_workout", "workout_change", "plan_request"]:
            # Check if we have the assessments from preprocessing
            accumulated = state.get("accumulated_context", {})
            if "readiness_assessment" in accumulated and "fitness_summary" in accumulated:
                # Recreate streamlined context
                ctx = {
                    "now": ctx.get("now"),
                    "today": ctx.get("today"),
                    "weekday": ctx.get("weekday"),
                    "weather_vincennes": ctx.get("weather_vincennes"),
                    "upcoming_events": ctx.get("upcoming_events"),
                    "latest_health": ctx.get("latest_health"),
                    "readiness_assessment": accumulated["readiness_assessment"],
                    "fitness_summary": accumulated["fitness_summary"],
                    "contextualized_query": ctx.get("contextualized_query"),
                    "memory_hits": ctx.get("memory_hits", [])[:3],
                }
        
        # Add accumulated context from previous agents
        accumulated = state.get("accumulated_context", {})
        if accumulated:
            ctx["previous_agent_outputs"] = accumulated
            ctx["agent_chain_history"] = list(accumulated.keys())
        
        # Execute the current agent based on its role
        agent_output = None
        if current_agent == "running_coach":
            if intent in ["workout_of_the_day", "daily_workout", "workout_change", "plan_request"]:
                agent_output = pm.running.execute_task(ctx, task="GENERATE_WORKOUT_JSON", query=user_text)
            else:
                agent_output = pm.running.execute_task(ctx, task="ANALYZE_DATA_AND_SUMMARIZE", query=user_text)
        elif current_agent == "cycling_coach":
            if intent in ["workout_of_the_day", "daily_workout", "workout_change", "plan_request"]:
                agent_output = pm.cycling.execute_task(ctx, task="GENERATE_WORKOUT_JSON", query=user_text)
            else:
                agent_output = pm.cycling.execute_task(ctx, task="ANALYZE_DATA_AND_SUMMARIZE", query=user_text)
        elif current_agent == "strength_coach":
            if intent in ["workout_of_the_day", "daily_workout", "plan_request"]:
                agent_output = pm.strength.execute_task(ctx, task="GENERATE_WORKOUT_JSON", query=user_text)
            else:
                agent_output = pm.strength.execute_task(ctx, task="ANALYZE_DATA_AND_SUMMARIZE", query=user_text)
        elif current_agent == "nutritionist":
            # Check if we have a workout from a previous agent to provide specific fueling advice
            if accumulated and any("workout" in str(v).lower() for v in accumulated.values()):
                agent_output = pm.nutrition.execute_task(ctx, task="PROVIDE_EXPERT_OPINION", 
                                                       query=f"Provide fueling plan for: {user_text}")
            else:
                agent_output = pm.nutrition.execute_task(ctx, task="PROVIDE_EXPERT_OPINION", query=user_text)
        elif current_agent == "psychologist":
            agent_output = pm.psychology.execute_task(ctx, task="PROVIDE_EXPERT_OPINION", query=user_text)
        elif current_agent == "data_analyst":
            metric = state.get("classification", {}).get("metric", "trend")
            # Use data_analyst_tool instead of removed analyst agent
            metric_data = data_analyst_tool.get_metric(ctx, metric)
            # Create a response-like object for consistency
            from types import SimpleNamespace
            agent_output = SimpleNamespace()
            agent_output.role = "data_analyst"
            if metric_data.get("found"):
                agent_output.content = f"{metric_data.get('label', metric)}: {metric_data.get('value')} {metric_data.get('unit', '')}".strip()
            else:
                agent_output.content = f"No data found for metric: {metric}"
        elif current_agent == "recovery_advisor":
            agent_output = pm.recovery.advise(ctx, reason="sequential chain request")
        
        # Store the output
        if agent_output:
            state.setdefault("agent_outputs", {})[agent_output.role] = agent_output.content
            state.setdefault("accumulated_context", {})[agent_output.role] = agent_output.content
            # Initialize agents list if it doesn't exist (for non-workout intents)
            if "agents" not in state or not isinstance(state["agents"], list):
                state["agents"] = []
            state["agents"].append(agent_output.role)
            
            # MULTI-AGENT FIX: Build consolidated response instead of just using last agent
            all_outputs = state.get("agent_outputs", {})
            if len(all_outputs) == 1:
                # Single agent - use direct output
                state["reply_text"] = agent_output.content
                state["formatted_message"] = agent_output.content
            else:
                # Multiple agents - create consolidated response optimized for Telegram
                consolidated_parts = []
                agent_order = state.get("agents", [])
                
                # TELEGRAM FIX: Better multi-agent formatting
                for i, agent_role in enumerate(agent_order):
                    if agent_role in all_outputs:
                        content = all_outputs[agent_role]
                        if content and content.strip():
                            # Use emoji icons for better Telegram display
                            agent_icons = {
                                "running_coach": "🏃",
                                "cycling_coach": "🚴",
                                "strength_coach": "💪",
                                "nutritionist": "🥗",
                                "psychologist": "🧠",
                                "recovery_advisor": "😴",
                                "data_analyst": "📊",
                                "data_tool": "📊"
                            }
                            
                            icon = agent_icons.get(agent_role, "🤖")
                            role_display = agent_role.replace('_', ' ').title()
                            
                            # For sequential agents, show progression
                            if len(agent_order) > 1:
                                consolidated_parts.append(f"{icon} **{role_display}:**\n{content}")
                            else:
                                # Single agent doesn't need header
                                consolidated_parts.append(content)
                
                if consolidated_parts:
                    if len(consolidated_parts) == 1:
                        # Single response, no need for separation
                        state["reply_text"] = consolidated_parts[0]
                        state["formatted_message"] = consolidated_parts[0]
                    else:
                        # Multi-agent response with clear separation
                        state["reply_text"] = "\n\n---\n\n".join(consolidated_parts)
                        state["formatted_message"] = "\n\n---\n\n".join(consolidated_parts)
                else:
                    # Fallback to last agent if consolidation fails
                    state["reply_text"] = agent_output.content
                    state["formatted_message"] = agent_output.content
        
        # Move to next agent
        state["current_agent_index"] = current_index + 1
        
        # TELEGRAM FIX: Ensure final response is set at each step for debugging
        state.setdefault("logs", {})["sequential_debug"] = {
            "current_agent_index": current_index,
            "total_agents": len(original_agents),
            "reply_text_set": bool(state.get("reply_text")),
            "formatted_message_set": bool(state.get("formatted_message")),
            "reply_text_preview": str(state.get("reply_text", ""))[:100],
        }
        
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
        # Data analyst is now a tool, not an agent
        if outputs:
            state["agent_outputs"] = {**(state.get("agent_outputs") or {}), **outputs}
            state["agents"] = agents
            # Use the output from the last (most important) agent as the final response
            if agents:
                final_output = outputs.get(agents[-1], "")
                state["reply_text"] = final_output
                state["formatted_message"] = final_output
            # Mark as conversational if single agent response
            if len(agents) == 1:
                state.setdefault("logs", {})["is_conversational"] = True
            return state
        r = pm.running.execute_task(ctx, task="ANALYZE_DATA_AND_SUMMARIZE", query=user_text)
        state["agents"] = [r.role]
        state.setdefault("agent_outputs", {})[r.role] = r.content
        state["reply_text"] = r.content
        state["formatted_message"] = r.content
        # Mark single agent general responses as conversational
        state.setdefault("logs", {})["is_conversational"] = True
        return state

    def handle_metric_query(state: BenjaminRouterState) -> BenjaminRouterState:
        """Handle metric query requests with robust error handling and fallback"""
        from agentic_app.agents.data_analyst_tool import data_analyst_tool
        
        metric = state.get("classification", {}).get("metric", "vo2max_running")
        ctx = state.get("context", {})
        user_text = state.get("user_text", "")
        
        # DATA ANALYST FIX: Robust error handling and fallback
        try:
            metric_data = data_analyst_tool.get_metric(ctx, metric)
            
            if metric_data.get("found") and metric_data.get("value") is not None:
                value_str = f"{metric_data['value']}"
                if metric_data.get("unit"):
                    value_str += f" {metric_data['unit']}"
                reply_text = f"Your latest {metric_data.get('label', metric)} is {value_str}."
                
                state["agents"] = ["data_tool"]
                state.setdefault("agent_outputs", {})["data_tool"] = reply_text
                state["reply_text"] = reply_text
                state["formatted_message"] = reply_text
                return state
            else:
                raise ValueError(f"Metric not found or invalid: {metric}")
                
        except Exception as e:
            # DATA ANALYST ERROR - Route to conversational specialist instead  
            # Enhanced context with error information
            enhanced_context = ctx.copy()
            enhanced_context["metric_request_failed"] = {
                "requested_metric": metric,
                "error": str(e),
                "original_request": user_text
            }
            
            # Route to running coach as fallback for metric analysis
            try:
                agent_output = pm.running.execute_task(enhanced_context, task="ANSWER_USER_QUESTION", query=user_text)
                state["agents"] = [agent_output.role]
                state.setdefault("agent_outputs", {})[agent_output.role] = agent_output.content
                state["reply_text"] = agent_output.content
                state["formatted_message"] = agent_output.content
            except Exception:
                # Ultimate fallback
                fallback_text = f"I'd be happy to help analyze your {metric.replace('_', ' ')} data. Could you provide more context about what specifically you'd like to know?"
                state["agents"] = ["running_coach"]
                state.setdefault("agent_outputs", {})["running_coach"] = fallback_text
                state["reply_text"] = fallback_text
                state["formatted_message"] = fallback_text
            
            return state


    graph.add_node("contextualize", contextualize_node)
    graph.add_node("enrich_context", enrich_context_node)
    graph.add_node("classify", classify_node)
    graph.add_node("sequential_agent_execution", sequential_agent_execution)
    graph.add_node("finalize_sequential", finalize_sequential)
    graph.add_node("handle_metric_query", handle_metric_query)
    graph.add_node("handle_wod", handle_wod)
    graph.add_node("handle_event", handle_event)
    graph.add_node("handle_explanation", handle_explanation)
    graph.add_node("handle_workout_change", handle_workout_change)
    graph.add_node("handle_nutrition_psych", handle_nutrition_psych)
    graph.add_node("handle_performance_forecast", handle_performance_forecast)
    graph.add_node("handle_plan_request", handle_plan_request)
    graph.add_node("handle_general", handle_general)

    graph.add_edge(START, "contextualize")
    graph.add_edge("contextualize", "enrich_context")
    graph.add_edge("enrich_context", "classify")

    def route_intent(state: BenjaminRouterState) -> str:
        # Check if we have a sequential agent chain
        agents = state.get("agents", [])
        if agents and len(agents) > 1:
            # Multiple agents in sequence - use sequential execution
            return "sequential_agent_execution"
        
        # Otherwise use standard routing
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
        "sequential_agent_execution": "sequential_agent_execution",
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

    # Add conditional edge for sequential execution
    def check_sequential_completion(state: BenjaminRouterState) -> str:
        agents = state.get("agents", [])
        current_index = state.get("current_agent_index", 0)
        
        # Get the original agent count from classification to check sequence completion
        classification = state.get("classification", {})
        original_agents = classification.get("agents", [])
        
        if original_agents and current_index < len(original_agents):
            # More agents to execute
            return "sequential_agent_execution"
        else:
            # TELEGRAM FIX: Done with sequence - do final consolidation
            return "finalize_sequential"
    
    def finalize_sequential(state: BenjaminRouterState) -> BenjaminRouterState:
        """Final consolidation of sequential agent outputs for Telegram delivery"""
        all_outputs = state.get("agent_outputs", {})
        agents_list = state.get("agents", [])
        
        # DEBUGGING: Log what we're processing
        import logging
        logger = logging.getLogger("agentic_app")
        logger.info(f"FINALIZE_DEBUG: Called with {len(all_outputs)} outputs from {len(agents_list)} agents")
        logger.info(f"FINALIZE_DEBUG: agents_list={agents_list}")
        logger.info(f"FINALIZE_DEBUG: output_keys={list(all_outputs.keys())}")
        for role, content in all_outputs.items():
            logger.info(f"FINALIZE_DEBUG: {role} -> {len(content)} chars: {content[:100]}...")
        
        if not all_outputs:
            # No outputs - set fallback
            state["reply_text"] = "I couldn't generate a proper response. Please try again."
            state["formatted_message"] = "I couldn't generate a proper response. Please try again."
            return state
        
        if len(all_outputs) == 1:
            # Single agent - use direct output
            agent_role = list(all_outputs.keys())[0]
            content = all_outputs[agent_role]
            state["reply_text"] = content
            state["formatted_message"] = content
        else:
            # Multiple agents - create consolidated response optimized for Telegram
            consolidated_parts = []
            
            # TELEGRAM FIX: Better multi-agent formatting for final response
            for agent_role in agents_list:
                if agent_role in all_outputs:
                    content = all_outputs[agent_role]
                    if content and content.strip():
                        # Use emoji icons for better Telegram display
                        agent_icons = {
                            "running_coach": "🏃",
                            "cycling_coach": "🚴", 
                            "strength_coach": "💪",
                            "nutritionist": "🥗",
                            "psychologist": "🧠",
                            "recovery_advisor": "😴",
                            "data_analyst": "📊",
                            "data_tool": "📊"
                        }
                        
                        icon = agent_icons.get(agent_role, "🤖")
                        role_display = agent_role.replace('_', ' ').title()
                        consolidated_parts.append(f"{icon} **{role_display}:**\n{content}")
            
            if consolidated_parts:
                # Multi-agent response with clear separation
                final_response = "\n\n---\n\n".join(consolidated_parts)
                state["reply_text"] = final_response
                state["formatted_message"] = final_response
            else:
                # Fallback if consolidation fails
                last_agent = agents_list[-1] if agents_list else "unknown"
                last_content = all_outputs.get(last_agent, "No content available")
                state["reply_text"] = last_content
                state["formatted_message"] = last_content
        
        # Add final debugging info
        final_reply = state.get("reply_text", "")
        final_formatted = state.get("formatted_message", "")
        
        logger.info(f"FINALIZE_DEBUG: Final reply_text: {len(final_reply)} chars")
        logger.info(f"FINALIZE_DEBUG: Final formatted_message: {len(final_formatted)} chars")
        logger.info(f"FINALIZE_DEBUG: Final reply preview: {final_reply[:200]}...")
        
        state.setdefault("logs", {})["final_consolidation"] = {
            "total_agents": len(agents_list),
            "total_outputs": len(all_outputs),
            "final_response_length": len(final_reply),
            "agents_processed": agents_list,
        }
        
        return state
    
    graph.add_conditional_edges("sequential_agent_execution", check_sequential_completion, {
        "sequential_agent_execution": "sequential_agent_execution",
        "finalize_sequential": "finalize_sequential",
    })
    
    # Finalize sequential goes directly to END
    graph.add_edge("finalize_sequential", END)
    
    # All handlers go directly to END - NO consolidation, NO summarization
    # This ensures specialist outputs go straight to the response without interference
    for handler in [
        "handle_metric_query", "handle_wod", "handle_event", "handle_explanation",
        "handle_workout_change", "handle_nutrition_psych", "handle_performance_forecast",
        "handle_plan_request", "handle_general",
    ]:
        graph.add_edge(handler, END)

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


# Legacy graph builder removed - use build_pm_router_graph instead


# Legacy collaboration and consensus graphs removed - use build_pm_router_graph instead


