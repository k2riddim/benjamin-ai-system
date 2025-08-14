"""
Database models for Benjamin AI System (agentic-ready, deduplicated)
"""

from datetime import datetime
from sqlalchemy import Column, String, Integer, Float, Text, DateTime, JSON, Date, Boolean, BigInteger
from sqlalchemy.dialects.postgresql import UUID
import uuid

from .connection import Base


class HealthMetrics(Base):
    __tablename__ = "garmin_health"

    date = Column(Date, primary_key=True)
    steps = Column(Integer)
    resting_heart_rate = Column(Integer)
    sleep_score = Column(Float)
    hrv_score = Column(Float)
    hrv_weekly_avg = Column(Float)
    hrv_status = Column(String)
    stress_score = Column(Float)
    max_stress = Column(Float)
    body_battery = Column(Float)
    training_readiness = Column(Float)
    sleep_hours = Column(Float)
    deep_sleep_minutes = Column(Integer)
    rem_sleep_minutes = Column(Integer)
    body_weight_kg = Column(Float)
    vo2max_running = Column(Float)
    vo2max_cycling = Column(Float)
    fitness_age = Column(Float)

    raw_stats = Column(JSON)
    raw_hr = Column(JSON)
    raw_sleep = Column(JSON)
    raw_hrv = Column(JSON)
    raw_training_readiness = Column(JSON)
    raw_training_status = Column(JSON)
    raw_body_battery = Column(JSON)
    raw_stress = Column(JSON)
    raw_respiration = Column(JSON)
    raw_spo2 = Column(JSON)
    raw_body_composition = Column(JSON)

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class Activities(Base):
    __tablename__ = "strava_activities"

    activity_id = Column("activity_id", BigInteger, primary_key=True)
    id = Column(String)
    name = Column(String)
    activity_type = Column(String)
    sport_type = Column(String)
    start_date = Column(DateTime)
    start_date_local = Column(DateTime)
    timezone = Column(String)
    duration = Column(Float)
    moving_time = Column(Float)
    distance = Column(Float)
    average_pace = Column(Float)
    average_speed = Column(Float)
    max_speed = Column(Float)
    elevation_gain = Column(Float)
    elev_high = Column(Float)
    elev_low = Column(Float)
    average_heart_rate = Column(Float)
    max_heart_rate = Column(Float)
    has_heartrate = Column(Boolean)
    average_watts = Column(Float)
    max_watts = Column(Float)
    weighted_average_watts = Column(Float)
    kilojoules = Column(Float)
    calories = Column(Float)
    suffer_score = Column(Float)
    training_load = Column(Float)
    source = Column(String, default='strava')
    device_name = Column(String)
    trainer = Column(Boolean)
    commute = Column(Boolean)
    manual = Column(Boolean)
    raw_data = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)


class DataSyncLog(Base):
    __tablename__ = "data_sync_logs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    sync_type = Column(String, nullable=False)
    sync_date = Column(Date, nullable=False)
    status = Column(String, nullable=False)
    records_processed = Column(Integer, default=0)
    records_updated = Column(Integer, default=0)
    records_created = Column(Integer, default=0)
    records_failed = Column(Integer, default=0)
    started_at = Column(DateTime, nullable=False)
    completed_at = Column(DateTime)
    duration_seconds = Column(Float)
    message = Column(Text)
    error_details = Column(JSON)
    data_range = Column(JSON)
    version = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)


class AgentDiscussions(Base):
    __tablename__ = "agent_discussions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    discussion_date = Column(Date, nullable=False)
    discussion_type = Column(String, nullable=False)
    agents_involved = Column(JSON)
    discussion_duration_seconds = Column(Float)
    topic = Column(String)
    summary = Column(Text)
    consensus_reached = Column(Boolean)
    agent_inputs = Column(JSON)
    agent_analyses = Column(JSON)
    agent_recommendations = Column(JSON)
    final_recommendations = Column(JSON)
    workout_plan = Column(JSON)
    message_content = Column(Text)
    benjamin_context = Column(JSON)
    data_sources_used = Column(JSON)
    started_at = Column(DateTime, nullable=False)
    completed_at = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)


class TelegramInteractions(Base):
    __tablename__ = "telegram_interactions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(String, nullable=False)
    chat_id = Column(String, nullable=False)
    username = Column(String)
    message_type = Column(String)
    message_content = Column(Text)
    command = Column(String)
    bot_response_type = Column(String)
    bot_response_content = Column(Text)
    response_success = Column(Boolean)
    conversation_context = Column(JSON)
    user_preferences = Column(JSON)
    agents_consulted = Column(JSON)
    agent_response_time_seconds = Column(Float)
    timestamp = Column(DateTime, default=datetime.utcnow)
    processed_at = Column(DateTime)


class SystemHealth(Base):
    __tablename__ = "system_health"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    check_date = Column(Date, nullable=False)
    check_time = Column(DateTime, nullable=False)
    data_app_status = Column(String)
    telegram_app_status = Column(String)
    agentic_app_status = Column(String)
    database_status = Column(String)
    last_garmin_sync = Column(DateTime)
    last_strava_sync = Column(DateTime)
    garmin_data_freshness_hours = Column(Float)
    strava_data_freshness_hours = Column(Float)
    garmin_api_responsive = Column(Boolean)
    strava_api_responsive = Column(Boolean)
    telegram_api_responsive = Column(Boolean)
    database_query_avg_ms = Column(Float)
    agent_response_avg_seconds = Column(Float)
    telegram_response_avg_seconds = Column(Float)
    health_details = Column(JSON)
    errors = Column(JSON)
    warnings = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)


class ReadinessDaily(Base):
    __tablename__ = "readiness_daily"

    date = Column(Date, primary_key=True)
    hrv_status = Column(String)
    hrv_overnight_avg = Column(Float)
    hrv_7d_avg = Column(Float)
    hrv_baseline_low = Column(Float)
    hrv_baseline_high = Column(Float)
    hrv_deviation = Column(Float)
    rhr = Column(Float)
    rhr_7d_avg = Column(Float)
    rhr_deviation = Column(Float)
    sleep_hours = Column(Float)
    sleep_score = Column(Float)
    deep_sleep_minutes = Column(Integer)
    rem_sleep_minutes = Column(Integer)
    training_status = Column(String)
    acute_load_7d = Column(Float)
    chronic_load_28d = Column(Float)
    weight_kg = Column(Float)
    weight_trend_28d = Column(Float)
    energy_level = Column(Integer)
    motivation = Column(Integer)
    soreness = Column(Integer)
    external_stressors = Column(Text)
    readiness_flag = Column(String)
    readiness_notes = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class TrainingLoadDaily(Base):
    __tablename__ = "training_load_daily"

    date = Column(Date, primary_key=True)
    daily_load = Column(Float)
    acute_load_7d = Column(Float)
    chronic_load_28d = Column(Float)
    training_status = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class BodyWeightLog(Base):
    __tablename__ = "body_weight_log"

    date = Column(Date, primary_key=True)
    weight_kg = Column(Float)
    trend_28d = Column(Float)
    source = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class SubjectiveFeedbackDaily(Base):
    __tablename__ = "subjective_feedback_daily"

    date = Column(Date, primary_key=True)
    energy_level = Column(Integer)
    motivation = Column(Integer)
    soreness = Column(Integer)
    stress_level = Column(Integer)
    notes = Column(Text)
    warnings = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
