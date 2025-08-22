from __future__ import annotations

from typing import Dict, Any, Tuple, List
import json
import time
import logging

from openai import OpenAI

from agentic_app.config.settings import settings
from agentic_app.agents.specialists import (
    RunningCoachAgent,
    CyclingCoachAgent,
    StrengthCoachAgent,
    NutritionistAgent,
    PsychologistAgent,
    AgentReply,
    RecoveryAdvisorAgent,
    OverallFitnessAgent,
)
from agentic_app.agents.data_analyst_tool import data_analyst_tool
from agentic_app.agents.tools import data_api, memory_store, short_memory
from agentic_app.agents.contextualizer import QueryContextualizer
from langsmith.run_helpers import traceable
from agentic_app.agents.graph import build_pm_router_graph
from agentic_app.agents.llm_utils import complete_text_with_guards
from agentic_app.agents.debug_helpers import add_debug_info_to_response, create_telegram_debug_message


class ProjectManagerRouter:
    """Routes user requests to the appropriate specialist agents."""

    def __init__(self) -> None:
        self.running = RunningCoachAgent()
        self.cycling = CyclingCoachAgent()
        self.strength = StrengthCoachAgent()
        self.nutrition = NutritionistAgent()
        self.psychology = PsychologistAgent()
        self.recovery = RecoveryAdvisorAgent()
        self.overall_fitness = OverallFitnessAgent()
        self.client = OpenAI(api_key=settings.openai_api_key) if settings.openai_api_key else None
        self.contextualizer = QueryContextualizer()
        self._llm_diag: Dict[str, Any] = {}
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
            "agents (array of strings from: running_coach, cycling_coach, strength_coach, nutritionist, psychologist, recovery_advisor). "
            
            "CRITICAL CLASSIFICATION GUIDANCE:"
            "- metric_query: ONLY for very explicit metric lookups using phrases like 'what is my [metric]', 'show me my [metric]', 'tell me my [metric]'. Examples: 'what is my VO2max', 'show me my weight', 'what's my HRV score', 'tell me my resting heart rate'"
            "- general: Use for ALL conversational analysis, rating, discussion requests, AND simple single-specialist requests. Examples: 'rate my run', 'how did I do', 'analyze my workout', 'thoughts on my performance', 'how was my endurance', 'rate my speed', 'what about my recovery', 'shopping list', 'meal plan', 'what should I eat', 'recipe ideas', 'help with motivation'"
            "- performance_forecast: For future predictions like 'how will I perform at my race'"
            "- nutrition_psych_support: ONLY for complex requests needing BOTH nutrition AND psychology input like 'help with emotional eating patterns' or 'nutrition plan for stress management'. For simple food requests like 'shopping list' or 'meal plan', use general."
            "- workout_change: For modifying existing workouts"
            "- plan_request: For requesting new workout plans"
            
            "IMPORTANT: When in doubt between metric_query and general, choose general. Only use metric_query for explicit 'what is my X' or 'show me my X' requests."
            
            "IMPORTANT: The agents array represents an ORDERED SEQUENCE of execution. Each agent's output will be passed as context to the next agent. "
            "For example: [\"running_coach\", \"nutritionist\", \"running_coach\"] means the running coach generates a workout, "
            "the nutritionist provides fueling advice for that workout, then the running coach makes final adjustments based on both. "
            "Optional fields: metric, sport, intensity, days, race, preference, event_title, event_date, delete_target."
        )
        user = f"User message: {user_text}"
        text, diag = complete_text_with_guards(
            self.client,
            model=(settings.openai_context_model or settings.openai_model),
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            max_completion_tokens=5120,
        )
        try:
            logging.getLogger("agentic_app").debug(f"gpt5.diag.classify: {diag}")
        except Exception:
            pass
        try:
            self._llm_diag["classify"] = diag
        except Exception:
            pass
        try:
            parsed = json.loads(text)
            return parsed
        except Exception:
            return {"intent": "general", "agents": ["running_coach"]}

    @traceable(name="pm.route", run_type="chain")
    def route(self, user_text: str, forced_intent: str | None = None, forced_metric: str | None = None, session_id: str | None = None, status_tracker = None) -> Dict[str, Any]:
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
                # Update status if tracker available
                if status_tracker and session_id:
                    status_tracker.set_status(session_id, "thinking", "Analyzing your request...", [])
                
                initial_state = {
                    "user_text": user_text,
                    "session_id": session_id,
                    "forced_intent": forced_intent,
                    "forced_metric": forced_metric,
                    "short_history": short_history,
                    "status_tracker": status_tracker,
                }
                
                if status_tracker and session_id:
                    status_tracker.set_status(session_id, "thinking", "Executing AI workflow...", [])
                    
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
        return self._legacy_route(user_text, forced_intent, forced_metric, session_id, t0, status_tracker)

    # Legacy routing preserved as a fallback path
    def _legacy_route(self, user_text: str, forced_intent: str | None, forced_metric: str | None, session_id: str | None, t0: float, status_tracker = None) -> Dict[str, Any]:
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
        
        # Include previous agent outputs (e.g., recently generated workout) for context continuity
        try:
            if session_id:
                previous_agent_outputs = short_memory.get_last_agent_outputs(session_id)
                if previous_agent_outputs:
                    context["previous_agent_outputs"] = previous_agent_outputs
        except Exception:
            pass
        is_followup = (len(user_text.split()) <= 12) or any(k in user_text.lower() for k in ["that", "this", "it", "how so", "why", "how did", "explain", "justify"]) 
        
        # Update status for classification
        if status_tracker and session_id:
            status_tracker.set_status(session_id, "thinking", "Understanding your request...", [])
            
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
        # Include any collected LLM diagnostics so they appear in API responses / traces
        try:
            if getattr(self, "_llm_diag", None):
                logs["gpt5_diag"] = dict(self._llm_diag)
        except Exception:
            pass
        context.setdefault("latest_health", context.get("latest_health", {}))
        intent = classification.get("intent")
        if is_followup and any(w in user_text.lower() for w in ["explain", "justify", "how", "why"]):
            intent = intent or "general"
            classification["intent"] = "workout_explanation"
        
        # Update status for agent routing
        if status_tracker and session_id:
            agents = classification.get("agents", [])
            if agents:
                status_tracker.set_status(session_id, "thinking", f"Consulting {', '.join(agents)}...", agents)
            else:
                status_tracker.set_status(session_id, "thinking", "Processing with AI specialists...", [])
        
        result = self._route_from_classification(rewritten_text, classification, context, logs)
        
        # Check if this came from conversational pipeline (should skip formatting)
        logs = result.get("logs", {}) or {}
        is_conversational = logs.get("is_conversational", False) or not self._is_training_of_the_day_request(intent, rewritten_text)
        
        # Always use direct agent output - NO consolidation
        result["formatted_message"] = result.get("reply_text", "")
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
        """
        SIMPLIFIED TWO-MODE ARCHITECTURE
        
        Mode 1: Plan Generation Pipeline - Heavy-duty workout generation
        Mode 2: Conversational Pipeline - Lightweight, natural responses
        """
        context = context or {}
        logs = logs or {"classification": classification}
        intent = classification.get("intent")
        
        # =====================================================
        # SINGLE TOP-LEVEL DECISION: Training of the Day?
        # =====================================================
        
        if self._is_training_of_the_day_request(intent, user_text):
            # MODE 1: PLAN GENERATION PIPELINE
            return self._execute_plan_generation_pipeline(context, logs, user_text)
        else:
            # MODE 2: CONVERSATIONAL PIPELINE (Default for everything else)
            return self._execute_conversational_pipeline(user_text, classification, context, logs)
    
    def _is_training_of_the_day_request(self, intent: str, user_text: str) -> bool:
        """Determine if this is a Training of the Day request."""
        # Explicit workout/plan generation requests
        training_intents = {
            "workout_of_the_day", 
            "daily_workout", 
            "plan_request",
            "training_for_today"
            # NOTE: workout_explanation is NOT here - it goes to conversational pipeline
        }
        
        if intent in training_intents:
            return True
            
        # Check for explicit workout request phrases in user text
        workout_phrases = [
            "workout of the day", "training of the day", "what should i train today",
            "give me a workout", "create a workout", "plan my training", "today's workout"
        ]
        
        lower_text = (user_text or "").lower()
        return any(phrase in lower_text for phrase in workout_phrases)
    
    def _execute_plan_generation_pipeline(
        self, 
        context: Dict[str, Any], 
        logs: Dict[str, Any],
        user_text: str
    ) -> Dict[str, Any]:
        """Execute the heavy-duty, multi-agent Plan Generation Pipeline."""
        
        # DEBUGGING: Track agent execution sequence
        agent_sequence = []
        logs["agent_sequence"] = agent_sequence
        
        # Step 1: Assess Readiness (Gatekeeper)
        agent_sequence.append("🛡️ Recovery Advisor (Readiness Assessment)")
        readiness_assessment = self.recovery.assess_readiness(context)
        context["readiness_assessment"] = readiness_assessment
        
        # Check if rest is required
        if readiness_assessment.get("status") == "rest_required":
            rec = self.recovery.advise(context, reason=readiness_assessment.get("notes", "Readiness assessment indicates rest needed"))
            agents = [rec.role]
            final_text = rec.content
            logs["agent_outputs"] = {rec.role: rec.content}
            logs["readiness_assessment"] = readiness_assessment
            return {
                "reply_text": final_text,
                "agents": agents,
                "context": context,
                "logs": logs,
                "duration_seconds": 0.0,
            }
        
        # Step 2: Assess Overall Fitness
        agent_sequence.append("📊 Overall Fitness Agent (Fitness Assessment)")
        fitness_summary = self.overall_fitness.assess_fitness(context)
        context["fitness_summary"] = fitness_summary
        
        # Step 3: Engage Full Team of Specialists
        proposals: Dict[str, str] = {}
        agents: List[str] = []
        
        # Generate workout proposals from sport coaches
        try:
            agent_sequence.append("🏃 Running Coach (Workout Generation)")
            r = self.running.execute_task(context, task="GENERATE_WORKOUT_JSON", query=user_text)
            proposals[r.role] = r.content
            agents.append(r.role)
        except Exception:
            pass
            
        try:
            agent_sequence.append("🚴 Cycling Coach (Workout Generation)")
            c = self.cycling.execute_task(context, task="GENERATE_WORKOUT_JSON", query=user_text)
            proposals[c.role] = c.content
            agents.append(c.role)
        except Exception:
            pass
            
        try:
            agent_sequence.append("💪 Strength Coach (Workout Generation)")
            s = self.strength.execute_task(context, task="GENERATE_WORKOUT_JSON", query=user_text)
            proposals[s.role] = s.content
            agents.append(s.role)
        except Exception:
            pass
        
        # Get support from nutrition and psychology
        try:
            agent_sequence.append("🥗 Nutritionist (Expert Opinion)")
            n = self.nutrition.execute_task(context, task="PROVIDE_EXPERT_OPINION", query=user_text)
            proposals[n.role] = n.content
            agents.append(n.role)
        except Exception:
            pass
            
        try:
            agent_sequence.append("🧠 Psychologist (Expert Opinion)")
            p = self.psychology.execute_task(context, task="PROVIDE_EXPERT_OPINION", query=user_text)
            proposals[p.role] = p.content
            agents.append(p.role)
        except Exception:
            pass
        
        # Step 4: HEAD COACH CONSOLIDATION (Critical for Plan Generation)
        if proposals:
            agent_sequence.append("👨‍💼 Head Coach (Consolidation & Synthesis)")
            consolidated_plan = self._consolidate(proposals, context, goal="single_workout")
            final_text = consolidated_plan or "\n\n".join(v for v in proposals.values() if v)
            logs["agent_outputs"] = {**proposals, "head_coach_consolidated": consolidated_plan}
        else:
            final_text = "I couldn't generate a training plan right now. Please try again."
            logs["agent_outputs"] = {}
        
        return {
            "reply_text": final_text,
            "agents": agents,
            "context": context,
            "logs": logs,
            "duration_seconds": 0.0,
        }
    
    def _execute_conversational_pipeline(
        self,
        user_text: str,
        classification: Dict[str, Any],
        context: Dict[str, Any],
        logs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute the lightweight Conversational Pipeline for all non-training requests."""
        
        # DEBUGGING: Track single agent execution
        agent_sequence = []
        logs["agent_sequence"] = agent_sequence
        
        intent = classification.get("intent")
        agents_list = classification.get("agents", [])
        
        # Handle metric queries with validation and graceful fallback
        if intent == "metric_query":
            agent_sequence.append("📊 Data Analyst (Metric Query)")
            metric = classification.get("metric", "vo2max_running")
            
            # First check if this is a supported metric
            metric_data = data_analyst_tool.get_metric(context, metric)
            
            if metric_data["found"]:
                value_str = f"{metric_data['value']}"
                if metric_data["unit"]:
                    value_str += f" {metric_data['unit']}"
                final_text = f"Your latest {metric_data['label']} is {value_str}."
                agents = ["data_analyst"]
                logs["agent_outputs"] = {"data_analyst": final_text}
            else:
                # Fallback: Route to conversational specialist for analysis
                agent_sequence.append("🔄 Fallback to Conversational Analysis")
                primary_agent = self._select_conversational_specialist([], "general", user_text)
                specialist_icons = {
                    "running_coach": "🏃 Running Coach",
                    "cycling_coach": "🚴 Cycling Coach", 
                    "strength_coach": "💪 Strength Coach",
                    "nutritionist": "🥗 Nutritionist",
                    "psychologist": "🧠 Psychologist",
                    "recovery_advisor": "😴 Recovery Advisor"
                }
                specialist_name = specialist_icons.get(primary_agent, f"🤖 {primary_agent}")
                agent_sequence.append(f"{specialist_name} (Metric Analysis)")
                
                agent_output = self._execute_single_specialist(primary_agent, context, user_text)
                
                if agent_output:
                    agents = [agent_output.role]
                    final_text = agent_output.content
                    logs["agent_outputs"] = {agent_output.role: agent_output.content}
                else:
                    # Ultimate fallback
                    agents = ["running_coach"]
                    final_text = f"I'd be happy to help you with {metric.replace('_', ' ')} analysis. Could you provide a bit more context about what specifically you'd like to know?"
                    logs["agent_outputs"] = {"running_coach": final_text}
            
        # Handle all other conversational requests with sequential agent execution
        else:
            # Multi-agent sequential execution
            if agents_list and len(agents_list) > 1:
                return self._execute_sequential_agents(agents_list, context, user_text, intent, logs)
            
            # Single agent execution
            elif intent == "nutrition_psych_support":
                # Determine if this is primarily nutrition or psychology focused
                primary_agent = self._select_nutrition_psych_specialist(user_text)
            else:
                # Determine the best specialist for this conversation
                primary_agent = self._select_conversational_specialist(agents_list, intent, user_text)
            
            # Add debugging info for specialist selection
            specialist_icons = {
                "running_coach": "🏃 Running Coach",
                "cycling_coach": "🚴 Cycling Coach", 
                "strength_coach": "💪 Strength Coach",
                "nutritionist": "🥗 Nutritionist",
                "psychologist": "🧠 Psychologist",
                "recovery_advisor": "😴 Recovery Advisor"
            }
            specialist_name = specialist_icons.get(primary_agent, f"🤖 {primary_agent}")
            agent_sequence.append(f"{specialist_name} (Conversational)")
            
            agent_output = self._execute_single_specialist(primary_agent, context, user_text)
            
            if agent_output:
                agents = [agent_output.role]
                final_text = agent_output.content
                logs["agent_outputs"] = {agent_output.role: agent_output.content}
            else:
                # Fallback to running coach
                agent_sequence.append("🏃 Running Coach (Fallback)")
                agent_output = self.running.execute_task(context, task="ANSWER_USER_QUESTION", query=user_text)
                agents = [agent_output.role]
                final_text = agent_output.content
                logs["agent_outputs"] = {agent_output.role: agent_output.content}
        
        # Return direct specialist response - NO consolidation, NO summarization
        # Mark this as conversational to bypass downstream consolidation
        logs["is_conversational"] = True
        return {
            "reply_text": final_text,
            "agents": agents,
            "context": context,
            "logs": logs,
            "duration_seconds": 0.0,
        }
    
    def _select_conversational_specialist(self, agents_list: List[str], intent: str, user_text: str) -> str:
        """Select the most appropriate specialist for conversational response."""
        # Use classifier suggestions if available
        if agents_list:
            return agents_list[0]
        
        # Fallback logic based on content
        lower_text = (user_text or "").lower()
        
        if any(word in lower_text for word in ["nutrition", "food", "eat", "diet", "meal", "hungry", "craving", "shopping", "grocery", "recipe", "cook", "breakfast", "lunch", "dinner", "snack"]):
            return "nutritionist"
        elif any(word in lower_text for word in ["mental", "psychology", "motivation", "stress", "anxious", "confidence", "mindset", "behavior", "habit", "focus", "mood"]):
            return "psychologist"
        elif any(word in lower_text for word in ["recovery", "sleep", "rest", "tired", "fatigue", "sore"]):
            return "recovery_advisor"
        elif any(word in lower_text for word in ["strength", "gym", "weights", "lifting", "muscle"]):
            return "strength_coach"
        elif any(word in lower_text for word in ["bike", "cycling", "ride", "gravel"]):
            return "cycling_coach"
        elif any(word in lower_text for word in ["run", "running", "jog", "jogging", "pace", "marathon", "5k", "10k"]):
            return "running_coach"
        else:
            return "running_coach"  # Default specialist
    
    def _select_nutrition_psych_specialist(self, user_text: str) -> str:
        """Smart selection between nutritionist and psychologist for mixed intent."""
        lower_text = (user_text or "").lower()
        
        # Clear nutrition tasks
        nutrition_keywords = [
            "meal", "food", "eat", "recipe", "cook", "shopping", "grocery", 
            "breakfast", "lunch", "dinner", "snack", "protein", "carb", 
            "nutrition", "diet", "calorie", "weight loss", "weight gain",
            "supplement", "vitamin", "hydration", "fuel", "recovery food"
        ]
        
        # Clear psychology tasks  
        psychology_keywords = [
            "stress", "anxiety", "motivation", "confidence", "mindset",
            "mental", "psychology", "pressure", "fear", "worry", "habit",
            "behavior", "emotional", "mood", "focus", "meditation",
            "sleep quality", "rest", "burnout", "overwhelm"
        ]
        
        nutrition_score = sum(1 for keyword in nutrition_keywords if keyword in lower_text)
        psychology_score = sum(1 for keyword in psychology_keywords if keyword in lower_text)
        
        # Special cases that are clearly nutrition
        if any(phrase in lower_text for phrase in [
            "shopping list", "meal plan", "what to eat", "food prep",
            "grocery", "recipe", "cooking", "breakfast ideas"
        ]):
            return "nutritionist"
        
        # Special cases that are clearly psychology
        if any(phrase in lower_text for phrase in [
            "feeling anxious", "stressed about", "mental health", 
            "motivation issues", "confidence problem", "mindset help"
        ]):
            return "psychologist"
        
        # Score-based decision
        if nutrition_score > psychology_score:
            return "nutritionist"
        elif psychology_score > nutrition_score:
            return "psychologist"
        else:
            # Default to nutritionist for ambiguous cases in nutrition_psych_support intent
            return "nutritionist"
    
    def _execute_sequential_agents(self, agents_list: List[str], context: Dict[str, Any], user_text: str, intent: str, logs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute multiple agents sequentially, with each agent improving on the previous output."""
        
        # Order agents by importance (most important last)
        ordered_agents = self._order_agents_by_importance(agents_list, intent, user_text)
        
        agent_sequence = []
        logs["agent_sequence"] = agent_sequence
        
        current_output = None
        final_agent = None
        enhanced_context = context.copy()
        
        for i, agent in enumerate(ordered_agents):
            is_last = (i == len(ordered_agents) - 1)
            
            # Add previous agent output to context for improvement
            if current_output and i > 0:
                enhanced_context["previous_agent_output"] = current_output
                enhanced_context["improvement_instruction"] = f"Build upon and enhance the previous response while maintaining its quality and completeness."
            
            # Execute current agent
            specialist_icons = {
                "running_coach": "🏃 Running Coach",
                "cycling_coach": "🚴 Cycling Coach", 
                "strength_coach": "💪 Strength Coach",
                "nutritionist": "🥗 Nutritionist",
                "psychologist": "🧠 Psychologist",
                "recovery_advisor": "😴 Recovery Advisor"
            }
            
            specialist_name = specialist_icons.get(agent, f"🤖 {agent}")
            stage = "Final Enhancement" if is_last else f"Stage {i+1}"
            agent_sequence.append(f"{specialist_name} ({stage})")
            
            # All agents use conversational task - they handle improvement internally
            task = "ANSWER_USER_QUESTION"
            
            # Execute agent with enhanced context containing improvement instructions
            agent_output = self._execute_conversational_specialist(agent, enhanced_context, user_text, task)
            
            if agent_output:
                current_output = agent_output.content
                final_agent = agent_output.role
                # Store intermediate outputs for debugging
                logs.setdefault("agent_outputs", {})[f"{agent}_stage_{i+1}"] = current_output
        
        # Return final enhanced output (NO consolidation)
        logs["is_conversational"] = True
        logs["sequential_execution"] = True
        
        return {
            "reply_text": current_output or "Unable to generate response",
            "agents": [final_agent] if final_agent else ordered_agents,
            "context": enhanced_context,
            "logs": logs,
            "duration_seconds": 0.0,
        }
    
    def _order_agents_by_importance(self, agents_list: List[str], intent: str, user_text: str) -> List[str]:
        """Order agents so the most important one is executed last."""
        
        # Agent importance hierarchy based on intent and content
        lower_text = (user_text or "").lower()
        
        # Define specialist expertise priorities
        primary_specialist = None
        
        # Determine primary specialist based on content
        if any(word in lower_text for word in ["nutrition", "food", "eat", "diet", "meal", "shopping", "recipe", "cook"]):
            primary_specialist = "nutritionist"
        elif any(word in lower_text for word in ["mental", "psychology", "motivation", "stress", "mindset", "behavior"]):
            primary_specialist = "psychologist"
        elif any(word in lower_text for word in ["recovery", "sleep", "rest", "tired", "fatigue"]):
            primary_specialist = "recovery_advisor"
        elif any(word in lower_text for word in ["strength", "gym", "weights", "lifting", "muscle"]):
            primary_specialist = "strength_coach"
        elif any(word in lower_text for word in ["bike", "cycling", "ride", "gravel"]):
            primary_specialist = "cycling_coach"
        elif any(word in lower_text for word in ["run", "running", "jog", "pace", "marathon"]):
            primary_specialist = "running_coach"
        
        # Standard importance order (least to most important)
        importance_order = [
            "recovery_advisor",    # Provides context
            "strength_coach",      # Provides support info
            "cycling_coach",       # Sport-specific input  
            "running_coach",       # Sport-specific input
            "psychologist",        # Behavioral enhancement
            "nutritionist"         # Often most detailed/practical
        ]
        
        # If we identified a primary specialist, put them last
        if primary_specialist and primary_specialist in agents_list:
            ordered = [agent for agent in importance_order if agent in agents_list and agent != primary_specialist]
            ordered.append(primary_specialist)
            return ordered
        
        # Otherwise use standard importance order
        return [agent for agent in importance_order if agent in agents_list]
    
    def _execute_conversational_specialist(self, specialist: str, context: Dict[str, Any], user_text: str, task: str = "ANSWER_USER_QUESTION"):
        """Execute the appropriate specialist with specified task."""
        try:
            if specialist == "running_coach":
                return self.running.execute_task(context, task=task, query=user_text)
            elif specialist == "cycling_coach":
                return self.cycling.execute_task(context, task=task, query=user_text)
            elif specialist == "strength_coach":
                return self.strength.execute_task(context, task=task, query=user_text)
            elif specialist == "nutritionist":
                return self.nutrition.execute_task(context, task=task, query=user_text)
            elif specialist == "psychologist":
                return self.psychology.execute_task(context, task=task, query=user_text)
            elif specialist == "recovery_advisor":
                return self.recovery.execute_task(context, task=task, query=user_text)
            else:
                return self.running.execute_task(context, task=task, query=user_text)
        except Exception:
            return None

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
        # Always include recovery advisor in consensus workflows
        rec = self.recovery.advise(context, reason="reviewing daily training consensus")
        proposals = {r.role: r.content, c.role: c.content, s.role: s.content, n.role: n.content, p.role: p.content, rec.role: rec.content}
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
            msg, diag2 = complete_text_with_guards(
                self.client,
                model=settings.openai_model,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                max_completion_tokens=12000,
            )
            try:
                logging.getLogger("agentic_app").debug(f"gpt5.diag.summarize: {diag2}")
            except Exception:
                pass
            try:
                self._llm_diag["summarize_to_telegram"] = diag2
            except Exception:
                pass
            msg = msg.strip()
            # Remove backticks / code fences and hard trim
            msg = msg.replace("```", "").replace("`", "")
            # If still empty, synthesize from hints/outputs to avoid a blank UI message
            if not msg:
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
                else:
                    bits: list[str] = []
                    for role, content in (agent_outputs or {}).items():
                        snippet = self._format_for_markdown(content).strip()
                        if snippet:
                            bits.append(snippet)
                    msg = ("\n\n".join(bits)) or "I could not generate a response right now. Please try again."
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

    def _execute_sequential_agents(self, agents_list: List[str], context: Dict[str, Any], user_text: str, intent: str, logs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute multiple agents sequentially, with each agent improving on the previous output."""
        
        # Order agents by importance (most important last)
        ordered_agents = self._order_agents_by_importance(agents_list, intent, user_text)
        
        agent_sequence = logs.get("agent_sequence", [])
        
        current_output = None
        final_agent = None
        enhanced_context = context.copy()
        
        for i, agent in enumerate(ordered_agents):
            is_last = (i == len(ordered_agents) - 1)
            
            # Add previous agent output to context for improvement
            if current_output and i > 0:
                enhanced_context["previous_agent_output"] = current_output
                enhanced_context["improvement_instruction"] = f"Build upon and enhance the previous response while maintaining its quality and completeness."
            
            # Execute current agent
            specialist_icons = {
                "running_coach": "🏃 Running Coach",
                "cycling_coach": "🚴 Cycling Coach", 
                "strength_coach": "💪 Strength Coach",
                "nutritionist": "🥗 Nutritionist",
                "psychologist": "🧠 Psychologist",
                "recovery_advisor": "😴 Recovery Advisor"
            }
            
            specialist_name = specialist_icons.get(agent, f"🤖 {agent}")
            stage = "Final Enhancement" if is_last else f"Stage {i+1}"
            agent_sequence.append(f"{specialist_name} ({stage})")
            
            # Execute agent with enhancement context
            agent_output = self._execute_single_specialist(agent, enhanced_context, user_text)
            
            if agent_output:
                current_output = agent_output.content
                final_agent = agent_output.role
                # Store intermediate outputs for debugging
                logs.setdefault("agent_outputs", {})[f"{agent}_stage_{i+1}"] = current_output
        
        # Return final enhanced output (NO consolidation)
        logs["is_conversational"] = True
        logs["sequential_execution"] = True
        
        return {
            "reply_text": current_output or "Unable to generate response",
            "agents": [final_agent] if final_agent else ordered_agents,
            "context": enhanced_context,
            "logs": logs,
            "duration_seconds": 0.0,
        }
    
    def _order_agents_by_importance(self, agents_list: List[str], intent: str, user_text: str) -> List[str]:
        """Order agents so the most important one is executed last."""
        
        # Agent importance hierarchy based on intent and content
        lower_text = (user_text or "").lower()
        
        # Define specialist expertise priorities
        primary_specialist = None
        
        # Determine primary specialist based on content
        if any(word in lower_text for word in ["nutrition", "food", "eat", "diet", "meal", "shopping", "recipe", "cook"]):
            primary_specialist = "nutritionist"
        elif any(word in lower_text for word in ["mental", "psychology", "motivation", "stress", "mindset", "behavior"]):
            primary_specialist = "psychologist"
        elif any(word in lower_text for word in ["recovery", "sleep", "rest", "tired", "fatigue"]):
            primary_specialist = "recovery_advisor"
        elif any(word in lower_text for word in ["strength", "gym", "weights", "lifting", "muscle"]):
            primary_specialist = "strength_coach"
        elif any(word in lower_text for word in ["bike", "cycling", "ride", "gravel"]):
            primary_specialist = "cycling_coach"
        elif any(word in lower_text for word in ["run", "running", "jog", "pace", "marathon"]):
            primary_specialist = "running_coach"
        
        # Standard importance order (least to most important)
        importance_order = [
            "recovery_advisor",    # Provides context
            "strength_coach",      # Provides support info
            "cycling_coach",       # Sport-specific input  
            "running_coach",       # Sport-specific input
            "psychologist",        # Behavioral enhancement
            "nutritionist"         # Often most detailed/practical
        ]
        
        # If we identified a primary specialist, put them last
        if primary_specialist and primary_specialist in agents_list:
            ordered = [agent for agent in importance_order if agent in agents_list and agent != primary_specialist]
            ordered.append(primary_specialist)
            return ordered
        
        # Otherwise use standard importance order
        return [agent for agent in importance_order if agent in agents_list]
    
    def _execute_single_specialist(self, specialist: str, context: Dict[str, Any], user_text: str):
        """Execute a single specialist with enhancement-aware context."""
        try:
            if specialist == "running_coach":
                return self.running.execute_task(context, task="ANSWER_USER_QUESTION", query=user_text)
            elif specialist == "cycling_coach":
                return self.cycling.execute_task(context, task="ANSWER_USER_QUESTION", query=user_text)
            elif specialist == "strength_coach":
                return self.strength.execute_task(context, task="ANSWER_USER_QUESTION", query=user_text)
            elif specialist == "nutritionist":
                return self.nutrition.execute_task(context, task="ANSWER_USER_QUESTION", query=user_text)
            elif specialist == "psychologist":
                return self.psychology.execute_task(context, task="ANSWER_USER_QUESTION", query=user_text)
            elif specialist == "recovery_advisor":
                return self.recovery.execute_task(context, task="ANSWER_USER_QUESTION", query=user_text)
            else:
                return self.running.execute_task(context, task="ANSWER_USER_QUESTION", query=user_text)
        except Exception:
            return None

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
                "You are the Head Coach of a world-class multidisciplinary sports science team. Your expertise lies in synthesizing complex specialist inputs into unified, actionable plans that optimize both performance and long-term athlete wellbeing. "
                
                "Your approach is holistic, intelligent, and deeply personalized. You understand that every athlete is unique, and you expertly balance competing demands to create plans that are both ambitious and sustainable. "
                
                "Core coaching philosophy: "
                "• Recovery and long-term health always take priority over short-term gains "
                "• Consistency and progressive overload create lasting adaptation "
                "• Mental and physical training must be perfectly aligned "
                "• Every recommendation must serve the athlete's bigger picture goals "
                
                "When goal=single_workout: Create ONE masterfully integrated daily training plan that harmonizes all specialist inputs. Prioritize the most impactful elements, resolve any conflicts intelligently, and ensure perfect timing and sequencing. Present as a clear, motivating daily plan. "
                
                "When goal=weekly_plan: Design a progressive, periodized plan that builds systematically toward the athlete's goals while respecting recovery principles and life demands. "
                
                "When goal=support_synthesis: Weave nutrition and psychology insights into practical, actionable guidance that supports both immediate performance and long-term behavioral change. CRITICAL: If any specialist mentions specific lists, recipes, or detailed information (like 'shopping lists' or 'meal plans'), you MUST include that complete information in your response. Never reference information that isn't accessible to the user. "
                
                "When goal=performance_forecast: Provide an evidence-based, holistic forecast that considers all physiological, psychological, and contextual factors affecting the athlete's trajectory. "
                
                "Communication style: Authoritative yet encouraging, technical yet accessible. Use Telegram Markdown formatting. Be comprehensive but concise - every word should add value. NEVER reference external information that the user cannot access."
            )
            stitched = "\n\n".join([f"[{role}]\n{content}" for role, content in (agent_outputs or {}).items()])
            events_hint = ", ".join([f"{e.get('title')} in {e.get('days_until','?')}d" for e in (context.get('upcoming_events') or [])[:3] if isinstance(e, dict)])
            user = (
                f"Goal: {goal}. Today: {context.get('today')}. Weekday: {context.get('weekday')}. Upcoming: {events_hint or 'none'}.\n\n"
                f"Specialists say:\n{stitched}"
            )
            msg, diag3 = complete_text_with_guards(
                self.client,
                model=settings.openai_model,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                max_completion_tokens=12000,
            )
            try:
                logging.getLogger("agentic_app").debug(f"gpt5.diag.consolidate: {diag3}")
            except Exception:
                pass
            try:
                self._llm_diag["consolidate"] = diag3
            except Exception:
                pass
            msg = (msg or "").strip()
            msg = msg.replace("```", "").replace("`", "")
            if not msg:
                combined = "\n\n".join(f"{k}: {v}" for k, v in (agent_outputs or {}).items())
                return self._format_for_markdown(combined)[:3000]
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
            text, diag4 = complete_text_with_guards(
                self.client,
                model=settings.openai_model,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                max_completion_tokens=3000,
            )
            try:
                logging.getLogger("agentic_app").debug(f"gpt5.diag.event_parse: {diag4}")
            except Exception:
                pass
            try:
                self._llm_diag["event_parse"] = diag4
            except Exception:
                pass
            import json as _json
            return _json.loads(text)
        except Exception:
            return {}


