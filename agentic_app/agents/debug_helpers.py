"""
Debugging helpers for the Benjamin AI System.

Provides utilities to surface agent execution information and add transparency
to the multi-agent coaching system.
"""

from typing import Dict, Any, List

def format_agent_sequence_for_display(agent_sequence: List[str]) -> str:
    """Format the agent execution sequence for user display."""
    if not agent_sequence:
        return ""
    
    # Create a visual flow of agent execution
    formatted = "🔍 **Agent Execution Flow:**\n"
    for i, agent in enumerate(agent_sequence):
        if i == 0:
            formatted += f"   {agent}\n"
        else:
            formatted += f"   ↓\n   {agent}\n"
    
    return formatted

def add_debug_info_to_response(final_text: str, logs: Dict[str, Any]) -> str:
    """Add debugging information to the final response."""
    agent_sequence = logs.get("agent_sequence", [])
    
    if not agent_sequence:
        return final_text
    
    # Add agent flow to response
    debug_info = format_agent_sequence_for_display(agent_sequence)
    
    # Add separator and append debug info
    if final_text:
        return f"{final_text}\n\n---\n{debug_info}"
    else:
        return debug_info

def create_telegram_debug_message(logs: Dict[str, Any], intent: str) -> str:
    """Create a short debug message for Telegram showing system operation."""
    agent_sequence = logs.get("agent_sequence", [])
    
    if not agent_sequence:
        return ""
    
    # Create compact debug message
    mode = "🎯 Plan Generation" if intent in ["workout_of_the_day", "daily_workout", "plan_request"] else "💬 Conversational"
    agents_count = len(agent_sequence)
    
    # Show first and last agent if multiple, or just the single agent
    if agents_count == 1:
        agent_summary = agent_sequence[0]
    elif agents_count > 1:
        first = agent_sequence[0].split(" (")[0]  # Get just the emoji and name
        last = agent_sequence[-1].split(" (")[0]
        agent_summary = f"{first} → ... → {last}"
    else:
        agent_summary = "No agents"
    
    return f"ℹ️ {mode} | {agents_count} step{'s' if agents_count != 1 else ''} | {agent_summary}"
