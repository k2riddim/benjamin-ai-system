"""
Configuration settings for DATA APP
"""

import os
from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings


class DataAppSettings(BaseSettings):
    """Configuration for the DATA APP"""
    
    # Application
    app_name: str = "Benjamin AI - Data App"
    version: str = "1.0.0"
    debug: bool = False
    
    # Database
    database_url: str = ""
    
    # Garmin Connect
    garmin_token_dir: str = "/home/chorizo/.garmin_tokens"
    garmin_max_retry_attempts: int = 3
    garmin_retry_delay_seconds: int = 5
    garmin_api_key: Optional[str] = None
    
    # Strava API
    strava_client_id: str = ""
    strava_client_secret: str = ""
    strava_access_token: str = ""
    strava_refresh_token: str = ""
    strava_token_expires_at: int = 0
    
    # Sync settings
    sync_interval_hours: int = 2
    sync_lookback_days: int = 7  # How many days back to sync
    sync_timeout_seconds: int = 300
    
    # Monitoring
    health_check_interval_hours: int = 24
    data_freshness_alert_hours: int = 6
    
    # Logging
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_file: str = "data_app.log"
    log_max_bytes: int = 10 * 1024 * 1024  # 10MB
    log_backup_count: int = 5
    
    # API endpoints
    garmin_api_base_url: str = "http://localhost:8008"
    strava_api_base_url: str = "https://www.strava.com/api/v3"
    
    # Rate limiting
    strava_requests_per_minute: int = 100
    garmin_requests_per_minute: int = 60
    
    class Config:
        # Ensure we load the .env co-located with this settings.py (data_app/config/.env)
        env_file = str(Path(__file__).resolve().parent / ".env")
        env_prefix = "DATA_APP_"


# Global settings instance
settings = DataAppSettings()


# Paths
BASE_DIR = Path(__file__).parent.parent
LOGS_DIR = BASE_DIR / "logs"
CONFIG_DIR = BASE_DIR / "config"

# Ensure directories exist
LOGS_DIR.mkdir(exist_ok=True)
CONFIG_DIR.mkdir(exist_ok=True)
