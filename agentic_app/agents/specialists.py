from __future__ import annotations

from typing import Dict, Any, List
from dataclasses import dataclass

from openai import OpenAI
from agentic_app.config.settings import settings
from langsmith.run_helpers import traceable
import logging
from agentic_app.agents.llm_utils import complete_text_with_guards
from agentic_app.agents.data_analyst_tool import data_analyst_tool


def _get_client() -> OpenAI:
    if not settings.openai_api_key:
        raise RuntimeError("AGENTIC_APP_OPENAI_API_KEY is not configured")
    return OpenAI(api_key=settings.openai_api_key)


@traceable(name="specialist.chat", run_type="llm")
def _chat(messages: List[Dict[str, str]], temperature: float = 0.4, max_tokens: int = 600) -> str:
    client = _get_client()
    text, diag = complete_text_with_guards(
        client,
        model=settings.openai_model,
        messages=messages,
        max_completion_tokens=max(9000, (max_tokens or 900) * 10),
    )
    try:
        logging.getLogger("agentic_app").debug(f"gpt5.diag.specialist.chat: {diag}")
    except Exception:
        pass
    return text


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
            "You are a caring and knowledgeable running coach who understands that every athlete's journey is unique. "
            "Your role is to provide personalized, empathetic guidance that respects the athlete's current state, goals, and life circumstances. "
            "Focus on being supportive, encouraging, and practical in your advice. Use the athlete's readiness assessment and fitness summary "
            "to give context-aware recommendations that prioritize long-term health and consistency over short-term gains. "
            "Communicate like a trusted coach who genuinely cares about the athlete's wellbeing and progress."
        )
        instructions: List[str] = []
        if task == "GENERATE_WORKOUT_JSON":
            instructions.append("Generate a single running workout for today as strict JSON with keys: title, summary, details, duration_min, intensity, sport='running'.")
        elif task == "ANSWER_USER_QUESTION":
            instructions.append("Answer in a warm, conversational tone as if speaking directly to the athlete. Be encouraging, specific, and actionable. If there are previous agent outputs in context (like recently generated workouts), reference them appropriately to maintain conversation continuity. Avoid technical jargon unless necessary, and always explain things in relatable terms. For performance analysis requests, look at recent activities data and provide specific insights about pace, heart rate, power, and how it relates to their fitness level and readiness.")
            # Check for sequential improvement context
            if context.get("previous_agent_output"):
                instructions.append(f"IMPORTANT: A colleague has already provided this initial response: '{context['previous_agent_output'][:500]}...' Your job is to build upon their excellent work, enhance it with your running expertise, maintain all their good content, and make the final response even better and more complete.")
        elif task == "PROVIDE_EXPERT_OPINION":
            instructions.append("Share your expert perspective in a friendly, approachable way. Explain your reasoning clearly and help the athlete understand the 'why' behind your recommendations.")
            # Check for sequential improvement context
            if context.get("previous_agent_output"):
                instructions.append(f"IMPORTANT: A colleague has provided this initial foundation: '{context['previous_agent_output'][:500]}...' Add your running expertise to enhance their work while preserving their valuable insights.")
        elif task == "ANALYZE_DATA_AND_SUMMARIZE":
            instructions.append("Translate the data into practical insights the athlete can understand and act upon. Focus on what matters most for their running today.")
        else:
            instructions.append("Respond in a helpful, conversational manner that shows you understand and care about the athlete's situation.")
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
            "You are an enthusiastic and experienced cycling coach who loves helping athletes discover the joy of riding. "
            "Your approach is balanced, practical, and always considers the athlete's overall wellbeing alongside their cycling goals. "
            "You understand that cycling should be sustainable and enjoyable, whether it's for fitness, competition, or pure pleasure. "
            "Use your expertise to provide guidance that considers terrain, weather, equipment, and the athlete's current state. "
            "Always communicate with warmth and enthusiasm while keeping safety and long-term development as top priorities."
        )
        instructions: List[str] = []
        if task == "GENERATE_WORKOUT_JSON":
            instructions.append("Generate a single cycling session for today as strict JSON with keys: title, summary, details, duration_min, intensity, sport='cycling'.")
        elif task == "ANSWER_USER_QUESTION":
            instructions.append("Respond with genuine enthusiasm and care. Share practical cycling insights in an accessible way that motivates and informs the athlete. If there are previous agent outputs in context (like recently generated workouts), reference them appropriately to maintain conversation continuity.")
        elif task == "PROVIDE_EXPERT_OPINION":
            instructions.append("Offer your cycling expertise in a friendly, encouraging manner. Help the athlete understand how your recommendations will benefit their riding and overall fitness.")
        elif task == "ANALYZE_DATA_AND_SUMMARIZE":
            instructions.append("Interpret the data to give the athlete clear, actionable cycling guidance for today. Focus on what will help them ride better and feel great.")
        else:
            instructions.append("Be helpful, encouraging, and specific in your guidance. Show your passion for cycling while keeping the athlete's needs at the center.")
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
            "You are a knowledgeable and supportive strength coach who believes that strength training should empower, not exhaust. "
            "Your philosophy centers on building resilient, balanced athletes through smart, progressive training that complements their endurance activities. "
            "You prioritize movement quality, injury prevention, and helping athletes feel strong and confident in their bodies. "
            "Your approach is encouraging and considers the athlete's overall training load, recovery status, and life demands. "
            "Communicate with enthusiasm about the benefits of strength training while keeping recommendations practical and achievable."
        )
        instructions: List[str] = []
        if task == "GENERATE_WORKOUT_JSON":
            instructions.append("Generate a short full-body routine as strict JSON with keys: title, summary, blocks (each with exercises, sets, reps), duration_min.")
        elif task == "ANSWER_USER_QUESTION":
            instructions.append("Respond with positivity and practical wisdom. Help the athlete understand how strength training will make them feel stronger and more resilient in their sport and daily life. If there are previous agent outputs in context (like recently generated workouts), reference them appropriately to maintain conversation continuity.")
        elif task == "PROVIDE_EXPERT_OPINION":
            instructions.append("Share your strength expertise in an encouraging way that motivates the athlete to embrace strength training as part of their journey.")
        elif task == "ANALYZE_DATA_AND_SUMMARIZE":
            instructions.append("Translate the data into strength training insights that will help the athlete train smarter and feel stronger today.")
        else:
            instructions.append("Be encouraging and practical in your strength guidance. Show how strength training fits into the athlete's bigger picture.")
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
            "You are a compassionate and knowledgeable sports nutritionist who understands that food is more than fuel—it's deeply personal and often emotional. "
            "Your approach is non-judgmental, evidence-based, and focused on helping athletes develop a healthy, sustainable relationship with food. "
            "You recognize that each athlete's nutritional needs are unique and influenced by their sport, body, lifestyle, and psychological relationship with eating. "
            "Your guidance is practical, flexible, and always considers the athlete's whole life, not just their training. "
            "You speak with warmth and understanding, especially when addressing sensitive topics around eating patterns and body image."
        )
        instructions: List[str] = []
        if task == "ANSWER_USER_QUESTION":
            instructions.append("Respond with empathy and practical nutrition wisdom. Help the athlete understand how good nutrition can support their goals while feeling sustainable and enjoyable. If there are previous agent outputs in context (like recently generated workouts), reference them appropriately to maintain conversation continuity.")
            # Check for sequential improvement context
            if context.get("previous_agent_output"):
                instructions.append(f"IMPORTANT: A colleague has already provided this response: '{context['previous_agent_output'][:500]}...' Your job is to build upon their excellent work, enhance it with your nutrition expertise, maintain all their valuable content, and make the final response even better and more comprehensive.")
        elif task == "PROVIDE_EXPERT_OPINION":
            instructions.append("Share evidence-based nutritional guidance in a supportive, non-prescriptive way. Focus on helping the athlete make informed choices that work for their life.")
            # Check for sequential improvement context
            if context.get("previous_agent_output"):
                instructions.append(f"IMPORTANT: A colleague has provided this foundation: '{context['previous_agent_output'][:500]}...' Add your nutrition expertise to enhance their work while preserving their valuable insights.")
        elif task == "ANALYZE_DATA_AND_SUMMARIZE":
            instructions.append("Translate the athlete's current state into practical nutritional insights that support both their performance and wellbeing today.")
        else:
            instructions.append("Be supportive and practical in your nutritional guidance. Help the athlete see how good nutrition can enhance their training and life.")
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
            "You are a warm and insightful sports psychologist who believes that mental fitness is just as important as physical fitness. "
            "Your approach is empathetic, non-judgmental, and focused on helping athletes develop sustainable mental strategies that work in both sport and life. "
            "You understand that each athlete's psychological needs are unique and that motivation, confidence, and resilience can be developed over time. "
            "You create a safe space for athletes to explore their thoughts and feelings about training, performance, and personal goals. "
            "Your guidance is practical, hopeful, and always affirms the athlete's worth beyond their performance."
        )
        instructions: List[str] = []
        if task == "ANSWER_USER_QUESTION":
            instructions.append("Respond with genuine empathy and psychological insight. Help the athlete understand their mental patterns and provide practical strategies they can use right away. If there are previous agent outputs in context (like recently generated workouts), reference them appropriately to maintain conversation continuity.")
            # Check for sequential improvement context
            if context.get("previous_agent_output"):
                instructions.append(f"IMPORTANT: A colleague has already provided this response: '{context['previous_agent_output'][:500]}...' Your job is to build upon their excellent work, enhance it with your psychological expertise, maintain all their valuable content, and make the final response even better with mental strategies and insights.")
        elif task == "PROVIDE_EXPERT_OPINION":
            instructions.append("Share your psychological expertise with warmth and compassion. Offer concrete mental strategies that will help the athlete feel more confident and resilient.")
            # Check for sequential improvement context
            if context.get("previous_agent_output"):
                instructions.append(f"IMPORTANT: A colleague has provided this foundation: '{context['previous_agent_output'][:500]}...' Add your psychological expertise to enhance their work while preserving their valuable insights.")
        elif task == "ANALYZE_DATA_AND_SUMMARIZE":
            instructions.append("Interpret the athlete's current state through a psychological lens, offering insights that will support their mental wellbeing and motivation today.")
        else:
            instructions.append("Be empathetic and supportive in your psychological guidance. Help the athlete see their mental strength and potential for growth.")
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


# DataAnalystAgent has been refactored into data_analyst_tool
# See data_analyst_tool.py for the new implementation


class OverallFitnessAgent:
    @traceable(name="overall_fitness.assess", run_type="chain")
    def assess_fitness(self, context: Dict[str, Any]) -> str:
        """Analyze long-term fitness metrics and produce a qualitative summary."""
        # Use data analyst tool to get comprehensive fitness trends
        fitness_trends = data_analyst_tool.get_fitness_trends(context)
        all_metrics = data_analyst_tool.get_all_metrics(context)
        
        system = (
            "You are a fitness assessment specialist on a multidisciplinary AI coaching team. "
            "Your task is to analyze the athlete's overall fitness state and produce a concise, qualitative summary. "
            "Focus on long-term trends, strengths, weaknesses, and areas for development. "
            "Consider VO2max for both running and cycling, weight trends, activity volume, and training load patterns. "
            "Provide a clear, actionable summary in 3-4 sentences that other specialists can use for context. "
            "Be specific about which areas are strong, plateauing, or need development."
        )
        
        # Prepare fitness data summary
        fitness_summary = {
            "fitness_trends": fitness_trends,
            "latest_health": all_metrics.get("latest_health", {}),
            "upcoming_events": context.get("upcoming_events", [])[:3]  # Next 3 events
        }
        
        user = (
            f"Fitness data: {fitness_summary}. "
            "Provide a qualitative assessment of the athlete's overall fitness state."
        )
        
        content = _chat([{"role": "system", "content": system}, {"role": "user", "content": user}], max_tokens=300)
        return content.strip()
    
    @traceable(name="overall_fitness.execute_task", run_type="chain")
    def execute_task(self, context: Dict[str, Any], task: str, query: str | None = None) -> AgentReply:
        """Execute task with consistent interface for integration."""
        if task == "ASSESS_FITNESS":
            content = self.assess_fitness(context)
        else:
            content = self.assess_fitness(context)  # Default to fitness assessment
        
        return AgentReply(role="overall_fitness", content=content)


class RecoveryAdvisorAgent:
    @traceable(name="recovery_advisor.assess_readiness", run_type="chain")
    def assess_readiness(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Assess athlete's readiness for training and return structured assessment."""
        # Use data analyst tool to get recovery metrics
        recovery_metrics = data_analyst_tool.get_recovery_metrics(context)
        training_load = data_analyst_tool.get_training_load(context)
        
        now = context.get("now") or ""
        system = (
            "You are a readiness assessment specialist on a multidisciplinary AI coaching team. "
            "Your primary task is to evaluate the athlete's readiness for training based on recovery metrics. "
            "Analyze sleep, HRV, training readiness, stress, body battery, and training load ratios. "
            "Return a JSON assessment with keys: 'status' (one of: 'ready_for_intensity', 'ready_for_moderate', 'recovery_recommended', 'rest_required'), "
            "'readiness_score' (0-100), 'notes' (brief explanation), 'limiting_factors' (list of concerns if any)."
        )
        
        # Prepare condensed metrics for assessment
        metrics_summary = {
            "recovery_metrics": recovery_metrics,
            "training_load": training_load,
            "weather": context.get("weather_vincennes", {})
        }
        
        user = (
            f"Today: {now}. Metrics: {metrics_summary}. "
            "Assess the athlete's readiness for training today."
        )
        
        content = _chat([{"role": "system", "content": system}, {"role": "user", "content": user}], max_tokens=400)
        
        # Parse the response
        try:
            import json
            assessment = json.loads(content)
            # Ensure required fields
            if "status" not in assessment:
                assessment["status"] = "ready_for_moderate"
            if "readiness_score" not in assessment:
                assessment["readiness_score"] = 50
            if "notes" not in assessment:
                assessment["notes"] = "Assessment completed"
            if "limiting_factors" not in assessment:
                assessment["limiting_factors"] = []
        except Exception:
            # Fallback assessment if parsing fails
            assessment = {
                "status": "ready_for_moderate",
                "readiness_score": 50,
                "notes": content[:200] if content else "Unable to parse detailed assessment",
                "limiting_factors": []
            }
        
        return assessment
    
    @traceable(name="recovery_advisor.execute_task", run_type="chain")
    def execute_task(self, context: Dict[str, Any], task: str, query: str | None = None) -> AgentReply:
        """Execute conversational tasks for the recovery advisor."""
        now = context.get("now") or ""
        system = (
            "You are a compassionate recovery and wellness advisor who prioritizes the athlete's long-term health and sustainable training. "
            "Your expertise lies in understanding the delicate balance between training stress and recovery, helping athletes listen to their bodies, and making smart decisions about rest. "
            "You believe that recovery is not time lost, but time invested in better performance and overall wellbeing. "
            "Your guidance is practical, encouraging, and always considers the athlete's whole life, not just their training schedule. "
            "You help athletes see recovery as an active, valuable part of their fitness journey."
        )
        instructions: List[str] = []
        if task == "ANSWER_USER_QUESTION":
            instructions.append("Respond with care and wisdom about recovery, sleep, stress management, and sustainable training. Help the athlete understand how proper recovery will make them stronger. If there are previous agent outputs in context (like recently generated workouts), reference them appropriately to maintain conversation continuity.")
        elif task == "PROVIDE_EXPERT_OPINION":
            instructions.append("Share your recovery expertise with empathy and practical guidance that helps the athlete make smart decisions about rest and training load.")
        elif task == "ANALYZE_DATA_AND_SUMMARIZE":
            instructions.append("Interpret the athlete's current recovery status and provide insights that will help them train more effectively today.")
        else:
            instructions.append("Be supportive and wise in your recovery guidance. Help the athlete see the value in taking care of their body.")
        
        if query:
            instructions.append(f"User query: {query}.")
        
        user = (
            f"Today: {now}. Context: {context}. Assigned task: {task}. "
            + " ".join(instructions)
        )
        content = _chat([{"role": "system", "content": system}, {"role": "user", "content": user}], max_tokens=350)
        return AgentReply(role="recovery_advisor", content=content)

    @traceable(name="recovery_advisor.advise", run_type="chain")
    def advise(self, context: Dict[str, Any], reason: str | None = None) -> AgentReply:
        now = context.get("now") or ""
        system = (
            "You are a recovery advisor on a multidisciplinary AI coaching team. "
            "When a rest day is warranted, provide a concise day plan focused on recovery with optional gentle movement, mobility, sleep targets, and readiness checks. "
            "Keep it athlete-friendly. "
            "You have access to a data analyst tool that can retrieve specific metrics when needed."
        )
        user = (
            f"Today: {now}. Context (condensed): { {k: context.get(k) for k in ['latest_health','training_load','readiness_recent','weather_vincennes']} }. "
            f"Reason for rest: {reason or 'automatic threshold'}."
        )
        content = _chat([{"role": "system", "content": system}, {"role": "user", "content": user}], max_tokens=320)
        return AgentReply(role="recovery_advisor", content=content)


