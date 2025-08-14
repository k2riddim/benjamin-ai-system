from __future__ import annotations

from datetime import datetime, date
from typing import Any, Dict, List, Optional
import sys
from pathlib import Path

# Ensure access to shared package
sys.path.append(str(Path(__file__).parent.parent.parent))

import logging
from shared.database.connection import get_db_session
from shared.database.models import AgentDiscussions


def log_agent_discussion(
    discussion_type: str,
    topic: str,
    summary: str,
    message_content: str,
    agents_involved: List[str],
    agent_inputs: Optional[Dict[str, Any]] = None,
    agent_analyses: Optional[Dict[str, Any]] = None,
    agent_recommendations: Optional[Dict[str, Any]] = None,
    final_recommendations: Optional[Dict[str, Any]] = None,
    benjamin_context: Optional[Dict[str, Any]] = None,
    data_sources_used: Optional[Dict[str, Any]] = None,
    duration_seconds: Optional[float] = None,
) -> str | None:
    """Persist an AgentDiscussions row. Returns the created discussion id or None on failure."""
    try:
        with get_db_session() as db:
            row = AgentDiscussions(
                discussion_date=date.today(),
                discussion_type=discussion_type,
                agents_involved=agents_involved,
                discussion_duration_seconds=duration_seconds,
                topic=topic,
                summary=summary[:5000] if summary else None,
                consensus_reached=None,
                agent_inputs=agent_inputs,
                agent_analyses=agent_analyses,
                agent_recommendations=agent_recommendations,
                final_recommendations=final_recommendations,
                workout_plan=None,
                message_content=message_content[:12000] if message_content else None,
                benjamin_context=benjamin_context,
                data_sources_used=data_sources_used,
                started_at=datetime.utcnow(),
                completed_at=datetime.utcnow(),
            )
            db.add(row)
            db.commit()
            return str(row.id)
    except Exception:
        # Do not raise; logging should be non-intrusive, but record error to file
        try:
            logging.getLogger("agentic_app").exception("Failed to log AgentDiscussion")
        except Exception:
            pass
        return None


