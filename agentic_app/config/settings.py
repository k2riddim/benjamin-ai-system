from __future__ import annotations

import os
from pathlib import Path
from pydantic import BaseModel


class AgenticSettings(BaseModel):
    app_name: str = "Benjamin Agentic App"
    version: str = "0.1.0"

    # Logging
    log_level: str = os.environ.get("AGENTIC_APP_LOG_LEVEL", "INFO")
    logs_dir: Path = Path(os.environ.get("AGENTIC_APP_LOGS_DIR", "/home/chorizo/projects/benjamin_ai_system/agentic_app/logs"))
    log_file: str = os.environ.get("AGENTIC_APP_LOG_FILE", "agentic_app.log")

    # Services
    data_api_base_url: str = os.environ.get("AGENTIC_APP_DATA_API_BASE_URL", "http://127.0.0.1:8010")
    telegram_integration_enabled: bool = os.environ.get("AGENTIC_APP_TELEGRAM_INTEGRATION_ENABLED", "false").lower() == "true"
    telegram_api_base_url: str = os.environ.get("AGENTIC_APP_TELEGRAM_API_BASE_URL", "http://127.0.0.1:8011")

    # OpenAI / LangChain
    openai_api_key: str | None = os.environ.get("AGENTIC_APP_OPENAI_API_KEY")
    openai_model: str = os.environ.get("AGENTIC_APP_OPENAI_MODEL", "gpt-4o-mini")
    embedding_model: str = os.environ.get("AGENTIC_APP_EMBEDDING_MODEL", "text-embedding-3-small")
    langsmith_project: str | None = os.environ.get("LANGCHAIN_PROJECT")
    qdrant_url: str | None = os.environ.get("AGENTIC_APP_QDRANT_URL")
    qdrant_api_key: str | None = os.environ.get("AGENTIC_APP_QDRANT_API_KEY")

    # Timeouts
    http_timeout_seconds: int = int(os.environ.get("AGENTIC_APP_HTTP_TIMEOUT", "30"))

    # Request logging toggle
    request_logging_enabled: bool = os.environ.get("AGENTIC_APP_REQUEST_LOGGING", "false").lower() == "true"

    # Context debug toggle (log/return context on /route when enabled or requested)
    context_debug_enabled: bool = os.environ.get("AGENTIC_APP_CONTEXT_DEBUG", "false").lower() == "true"


settings = AgenticSettings()
LOGS_DIR = settings.logs_dir
LOGS_DIR.mkdir(parents=True, exist_ok=True)


