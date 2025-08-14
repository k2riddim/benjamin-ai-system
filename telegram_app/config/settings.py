"""
Configuration settings for TELEGRAM APP
"""

import os
from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings


class TelegramAppSettings(BaseSettings):
    """Configuration for the TELEGRAM APP"""
    
    # Application
    app_name: str = "Benjamin AI - Telegram App"
    version: str = "1.0.0"
    debug: bool = False
    
    # Telegram Bot
    telegram_bot_token: str = ""  # Set via environment variable
    telegram_chat_id: str = ""    # Benjamin's chat ID
    telegram_webhook_url: Optional[str] = None
    telegram_webhook_port: int = 8443
    
    # Data API Integration
    data_api_base_url: str = "http://127.0.0.1:8010"
    data_api_timeout: int = 30

    # Agentic API Integration
    agentic_api_base_url: str = "http://127.0.0.1:8012"

    # OpenAI (for AI replies in Telegram)
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    
    # Message Settings
    daily_message_time: str = "07:00"  # Time to send daily messages
    max_message_length: int = 4000
    enable_rich_formatting: bool = True
    
    # Interactive Features
    enable_quick_replies: bool = True
    enable_workout_feedback: bool = True
    enable_goal_tracking: bool = True
    
    # Logging
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_file: str = "telegram_app.log"
    log_max_bytes: int = 10 * 1024 * 1024  # 10MB
    log_backup_count: int = 5
    
    # Schedule Settings
    check_interval_minutes: int = 5  # How often to check for scheduled messages
    
    # User Preferences (will be stored in database later)
    default_timezone: str = "Europe/Paris"
    preferred_units: str = "metric"  # metric or imperial
    
    class Config:
        env_file = ".env"
        env_prefix = "TELEGRAM_APP_"


# Global settings instance
settings = TelegramAppSettings()

# Paths
BASE_DIR = Path(__file__).parent.parent
LOGS_DIR = BASE_DIR / "logs"
CONFIG_DIR = BASE_DIR / "config"

# Ensure directories exist
LOGS_DIR.mkdir(exist_ok=True)
CONFIG_DIR.mkdir(exist_ok=True)
LOGS_DIR = BASE_DIR / "logs"
CONFIG_DIR = BASE_DIR / "config"

# Ensure directories exist
LOGS_DIR.mkdir(exist_ok=True)
CONFIG_DIR.mkdir(exist_ok=True)

LOGS_DIR = BASE_DIR / "logs"
CONFIG_DIR = BASE_DIR / "config"

# Ensure directories exist
LOGS_DIR.mkdir(exist_ok=True)
CONFIG_DIR.mkdir(exist_ok=True)

LOGS_DIR = BASE_DIR / "logs"
CONFIG_DIR = BASE_DIR / "config"

# Ensure directories exist
LOGS_DIR.mkdir(exist_ok=True)
CONFIG_DIR.mkdir(exist_ok=True)

LOGS_DIR = BASE_DIR / "logs"
CONFIG_DIR = BASE_DIR / "config"

# Ensure directories exist
LOGS_DIR.mkdir(exist_ok=True)
CONFIG_DIR.mkdir(exist_ok=True)

