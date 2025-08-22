"""
DataSynchronizer orchestrates Garmin and Strava collection and database upserts.

Features:
- Sync last N days or explicit date ranges
- Upsert into base tables `garmin_health` and `strava_activities`
- Forward-fill missing daily `vo2max` and `body_weight_kg`
- Write sync attempt summaries into `data_sync_logs`
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .garmin_collector import GarminCollector, GarminHealthData
from .strava_collector import StravaCollector, StravaActivity
from ..config.settings import settings

# Import shared database components
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from shared.database import (  # type: ignore
    get_db_session,
    HealthMetrics,
    Activities,
    DataSyncLog,
    ReadinessDaily,
    TrainingLoadDaily,
    BodyWeightLog,
)


@dataclass
class SyncResult:
    success: bool
    records: int = 0
    created: int = 0
    updated: int = 0
    errors: List[str] = None  # type: ignore

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "records": self.records,
            "created": self.created,
            "updated": self.updated,
            "errors": self.errors or [],
        }


class DataSynchronizer:
    """Coordinates collectors and database writes."""

    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)
        self.garmin = GarminCollector()
        self.strava = StravaCollector()

    # ---------- Connection tests ----------
    def test_all_connections(self) -> Dict[str, bool]:
        results: Dict[str, bool] = {
            "garmin": False,
            "strava": False,
            "database": False,
        }
        try:
            results["garmin"] = bool(self.garmin.test_connection())
        except Exception:
            results["garmin"] = False
        try:
            results["strava"] = bool(self.strava.test_connection())
        except Exception:
            results["strava"] = False
        try:
            # Light DB ping
            from sqlalchemy import text
            with get_db_session() as db:
                db.execute(text("SELECT 1"))
            results["database"] = True
        except Exception:
            results["database"] = False
        return results

    # ---------- Public sync orchestration ----------
    def sync_all_data(self, days_back: int = None) -> Dict[str, Dict[str, Any]]:
        if days_back is None:
            days_back = settings.sync_lookback_days
        end_date = date.today()
        start_date = end_date - timedelta(days=days_back)
        return self._sync_both(start_date, end_date)

    def sync_all_data_range(self, start_date: date, end_date: date) -> Dict[str, Dict[str, Any]]:
        return self._sync_both(start_date, end_date)

    def _sync_both(self, start_date: date, end_date: date) -> Dict[str, Dict[str, Any]]:
        self.logger.info(f"Starting combined sync {start_date} → {end_date}")
        garmin_res = self.sync_garmin_health_data(start_date, end_date)
        strava_res = self.sync_strava_recent((end_date - start_date).days)
        return {
            "garmin_health": garmin_res,
            "strava_activities": strava_res,
        }

    # ---------- Garmin health ----------
    def sync_garmin_health_data(self, start_date: date, end_date: date) -> Dict[str, Any]:
        sync_started = datetime.utcnow()
        result = SyncResult(success=True, records=0, created=0, updated=0, errors=[])
        data_range = {"start": start_date.isoformat(), "end": end_date.isoformat()}

        try:
            # Collect
            collected: List[GarminHealthData] = self.garmin.collect_date_range(start_date, end_date)
            # Index by date for convenience
            date_to_data: Dict[date, GarminHealthData] = {d.date: d for d in collected}

            with get_db_session() as db:
                # Preload existing rows across window to allow forward-fill based on historical values
                existing_rows: List[HealthMetrics] = (
                    db.query(HealthMetrics)
                    .filter(HealthMetrics.date >= start_date, HealthMetrics.date <= end_date)
                    .order_by(HealthMetrics.date.asc())
                    .all()
                )
                existing_by_date: Dict[date, HealthMetrics] = {r.date: r for r in existing_rows}

                # Determine last known vo2 and weight from historical DB before window for robust forward-fill
                prior_row: Optional[HealthMetrics] = (
                    db.query(HealthMetrics)
                    .filter(HealthMetrics.date < start_date)
                    .order_by(HealthMetrics.date.desc())
                    .first()
                )
                last_fitness_age: Optional[float] = (
                    float(prior_row.fitness_age) if prior_row and isinstance(prior_row.fitness_age, (int, float)) else None
                )
                last_weight: Optional[float] = float(prior_row.body_weight_kg) if prior_row and isinstance(prior_row.body_weight_kg, (int, float)) else None

                # Prefetch Strava activities to infer modality for VO2 assignment
                activities = (
                    db.query(Activities)
                    .filter(Activities.start_date >= datetime.combine(start_date, datetime.min.time()))
                    .filter(Activities.start_date <= datetime.combine(end_date, datetime.max.time()))
                    .all()
                )
                run_dates = set()
                ride_dates = set()
                for act in activities:
                    if not act or not act.start_date:
                        continue
                    d_only = act.start_date.date()
                    atype = (act.activity_type or "").lower()
                    sport = (getattr(act, "sport_type", None) or atype).lower()
                    # Running modalities
                    if (
                        atype in ("run", "trailrun")
                        or any(k in sport for k in ["run", "trail", "ultrarun", "race run"])  # broad match
                    ):
                        run_dates.add(d_only)
                    # Cycling modalities: ride, gravel, mtb, cyclocross, virtual
                    if (
                        atype in ("ride", "ebikeride", "virtualride", "gravelride", "mtb", "cyclocross")
                        or any(k in sport for k in ["ride", "cycle", "bike", "gravel", "mtb", "cyclocross", "virtual"])
                    ):
                        ride_dates.add(d_only)

                # Iterate day by day in ascending order for forward-fill
                current = start_date
                while current <= end_date:
                    d = current
                    source: Optional[GarminHealthData] = date_to_data.get(d)

                    # Extract values
                    steps_val: Optional[int] = None
                    rhr_val: Optional[int] = None
                    sleep_score_val: Optional[float] = None
                    vo2_val: Optional[float] = None
                    fitness_age_val: Optional[float] = None
                    hrv_score_val: Optional[float] = None
                    hrv_weekly_avg_val: Optional[float] = None
                    hrv_status_val: Optional[str] = None
                    stress_score_val: Optional[float] = None
                    max_stress_val: Optional[float] = None
                    body_battery_val: Optional[float] = None
                    training_readiness_val: Optional[float] = None
                    sleep_hours_val: Optional[float] = None
                    deep_sleep_minutes_val: Optional[int] = None
                    rem_sleep_minutes_val: Optional[int] = None
                    body_weight_val: Optional[float] = None
                    raw_stats: Optional[Dict[str, Any]] = None
                    raw_hr: Optional[Dict[str, Any]] = None
                    raw_sleep: Optional[Dict[str, Any]] = None
                    raw_hrv: Optional[Dict[str, Any]] = None
                    raw_training_readiness: Optional[Dict[str, Any]] = None
                    raw_training_status: Optional[Dict[str, Any]] = None
                    raw_body_battery: Optional[Dict[str, Any]] = None
                    raw_stress: Optional[Dict[str, Any]] = None
                    raw_respiration: Optional[Dict[str, Any]] = None
                    raw_spo2: Optional[Dict[str, Any]] = None
                    raw_body_composition: Optional[Dict[str, Any]] = None

                    if source:
                        steps_val = source.steps
                        rhr_val = source.resting_heart_rate
                        sleep_score_val = source.sleep_score
                        vo2_val = source.vo2max
                        fitness_age_val = source.fitness_age
                        hrv_score_val = source.hrv_score
                        hrv_weekly_avg_val = source.hrv_weekly_avg
                        hrv_status_val = source.hrv_status
                        stress_score_val = source.stress_score
                        max_stress_val = source.max_stress
                        body_battery_val = source.body_battery
                        training_readiness_val = source.training_readiness
                        sleep_hours_val = source.sleep_hours
                        deep_sleep_minutes_val = source.deep_sleep_minutes
                        rem_sleep_minutes_val = source.rem_sleep_minutes
                        body_weight_val = source.body_weight_kg
                        raw_stats = source.raw_stats
                        raw_hr = source.raw_hr
                        raw_sleep = source.raw_sleep
                        raw_hrv = source.raw_hrv
                        raw_training_readiness = source.raw_training_readiness
                        raw_training_status = getattr(source, "raw_training_status", None)
                        raw_body_battery = source.raw_body_battery
                        raw_stress = source.raw_stress
                        raw_respiration = source.raw_respiration
                        raw_spo2 = source.raw_spo2
                        raw_body_composition = source.raw_body_composition

                    # Determine modality-specific VO2 placement
                    vo2_run_val: Optional[float] = None
                    vo2_cycle_val: Optional[float] = None
                    if isinstance(vo2_val, (int, float)):
                        if d in run_dates:
                            vo2_run_val = float(vo2_val)
                        if d in ride_dates:
                            vo2_cycle_val = float(vo2_val)

                    # Maintain last-known per-modality values
                    if not isinstance(vo2_run_val, (int, float)):
                        vo2_run_val = None
                    if not isinstance(vo2_cycle_val, (int, float)):
                        vo2_cycle_val = None

                    # Forward-fill fitness_age and weight if missing
                    if not isinstance(fitness_age_val, (int, float)) or fitness_age_val is None:
                        fitness_age_val = last_fitness_age
                    else:
                        last_fitness_age = float(fitness_age_val)

                    if not isinstance(body_weight_val, (int, float)) or body_weight_val is None:
                        body_weight_val = last_weight
                    else:
                        last_weight = float(body_weight_val)

                    # Upsert
                    row: Optional[HealthMetrics] = existing_by_date.get(d)
                    if row is None:
                        row = HealthMetrics(date=d)
                        db.add(row)
                        result.created += 1
                    else:
                        result.updated += 1

                    row.steps = int(steps_val) if isinstance(steps_val, (int, float)) else row.steps
                    row.resting_heart_rate = int(rhr_val) if isinstance(rhr_val, (int, float)) else row.resting_heart_rate
                    row.sleep_score = float(sleep_score_val) if isinstance(sleep_score_val, (int, float)) else row.sleep_score
                    # Per-modality columns with forward-fill
                    # Initialize last-known modality values from existing row if present
                    # (only for the first time we see this date)
                    # Track across days using outer variables
                    try:
                        last_vo2_run
                    except NameError:
                        last_vo2_run = None
                    try:
                        last_vo2_cycle
                    except NameError:
                        last_vo2_cycle = None

                    # On first day of loop, seed from prior_row
                    if current == start_date:
                        last_vo2_run = (
                            float(getattr(prior_row, "vo2max_running", None))
                            if prior_row and isinstance(getattr(prior_row, "vo2max_running", None), (int, float))
                            else None
                        )
                        last_vo2_cycle = (
                            float(getattr(prior_row, "vo2max_cycling", None))
                            if prior_row and isinstance(getattr(prior_row, "vo2max_cycling", None), (int, float))
                            else None
                        )

                    # Determine today's values or forward-fill
                    today_vo2_run = vo2_run_val if isinstance(vo2_run_val, (int, float)) else last_vo2_run
                    today_vo2_cycle = vo2_cycle_val if isinstance(vo2_cycle_val, (int, float)) else last_vo2_cycle

                    if isinstance(today_vo2_run, (int, float)):
                        row.vo2max_running = float(today_vo2_run)
                        last_vo2_run = float(today_vo2_run)
                    if isinstance(today_vo2_cycle, (int, float)):
                        row.vo2max_cycling = float(today_vo2_cycle)
                        last_vo2_cycle = float(today_vo2_cycle)
                    row.fitness_age = float(fitness_age_val) if isinstance(fitness_age_val, (int, float)) else row.fitness_age
                    row.hrv_score = float(hrv_score_val) if isinstance(hrv_score_val, (int, float)) else row.hrv_score
                    row.hrv_weekly_avg = float(hrv_weekly_avg_val) if isinstance(hrv_weekly_avg_val, (int, float)) else row.hrv_weekly_avg
                    row.hrv_status = hrv_status_val if isinstance(hrv_status_val, str) else row.hrv_status
                    row.stress_score = float(stress_score_val) if isinstance(stress_score_val, (int, float)) else row.stress_score
                    row.max_stress = float(max_stress_val) if isinstance(max_stress_val, (int, float)) else row.max_stress
                    row.body_battery = float(body_battery_val) if isinstance(body_battery_val, (int, float)) else row.body_battery
                    row.training_readiness = float(training_readiness_val) if isinstance(training_readiness_val, (int, float)) else row.training_readiness
                    row.sleep_hours = float(sleep_hours_val) if isinstance(sleep_hours_val, (int, float)) else row.sleep_hours
                    row.deep_sleep_minutes = int(deep_sleep_minutes_val) if isinstance(deep_sleep_minutes_val, (int, float)) else row.deep_sleep_minutes
                    row.rem_sleep_minutes = int(rem_sleep_minutes_val) if isinstance(rem_sleep_minutes_val, (int, float)) else row.rem_sleep_minutes
                    row.body_weight_kg = float(body_weight_val) if isinstance(body_weight_val, (int, float)) else row.body_weight_kg
                    row.raw_stats = raw_stats or row.raw_stats
                    row.raw_hr = raw_hr or row.raw_hr
                    row.raw_sleep = raw_sleep or row.raw_sleep
                    row.raw_hrv = raw_hrv or row.raw_hrv
                    row.raw_training_readiness = raw_training_readiness or row.raw_training_readiness
                    row.raw_body_battery = raw_body_battery or row.raw_body_battery
                    row.raw_stress = raw_stress or row.raw_stress
                    row.raw_respiration = raw_respiration or row.raw_respiration
                    row.raw_spo2 = raw_spo2 or row.raw_spo2
                    row.raw_body_composition = raw_body_composition or row.raw_body_composition
                    row.updated_at = datetime.utcnow()

                    existing_by_date[d] = row

                    # Upsert minimal readiness_daily row with training status and sleep metrics
                    rd = db.query(ReadinessDaily).filter(ReadinessDaily.date == d).first()
                    if rd is None:
                        rd = ReadinessDaily(date=d)
                        db.add(rd)
                    if isinstance(hrv_status_val, str):
                        rd.hrv_status = hrv_status_val
                    if isinstance(hrv_score_val, (int, float)):
                        rd.hrv_overnight_avg = float(hrv_score_val)
                    if isinstance(rhr_val, (int, float)):
                        rd.rhr = float(rhr_val)
                    if isinstance(sleep_hours_val, (int, float)):
                        rd.sleep_hours = float(sleep_hours_val)
                    if isinstance(sleep_score_val, (int, float)):
                        rd.sleep_score = float(sleep_score_val)
                    # Parse and set training status string from source
                    ts_str = getattr(source, "training_status", None) if source else None
                    if isinstance(ts_str, str) and ts_str:
                        rd.training_status = ts_str
                    # Set default readiness flag from training readiness if present
                    if isinstance(training_readiness_val, (int, float)):
                        if training_readiness_val >= 70:
                            rd.readiness_flag = "high"
                        elif training_readiness_val >= 50:
                            rd.readiness_flag = "moderate"
                        else:
                            rd.readiness_flag = "low"
                    result.records += 1

                    # Periodic commit to avoid long transactions
                    if result.records % 200 == 0:
                        db.commit()

                    current += timedelta(days=1)

                db.commit()

                # Log sync
                self._log_sync(db, sync_type="garmin_health", start=sync_started, status="success",
                               processed=result.records, updated=result.updated, created=result.created, data_range=data_range)
                db.commit()

        except Exception as e:
            self.logger.exception("Garmin health sync error")
            result.success = False
            result.errors.append(str(e))
            try:
                with get_db_session() as db:
                    self._log_sync(db, sync_type="garmin_health", start=sync_started, status="error",
                                   processed=result.records, updated=result.updated, created=result.created,
                                   data_range=data_range, message=str(e))
                    db.commit()
            except Exception:
                pass

        return result.to_dict()

    # ---------- Strava ----------
    def sync_strava_recent(self, days_back: int = 7) -> Dict[str, Any]:
        sync_started = datetime.utcnow()
        result = SyncResult(success=True, records=0, created=0, updated=0, errors=[])
        data_range = {"days_back": days_back}

        try:
            activities: List[StravaActivity] = self.strava.collect_recent_activities(days_back=days_back)
            with get_db_session() as db:
                existing_ids = {
                    a[0]
                    for a in db.query(Activities.activity_id).all()  # type: ignore
                    if a[0] is not None
                }

                for act in activities:
                    if act is None or act.id is None:
                        continue
                    try:
                        activity_pk = int(act.id)
                    except Exception:
                        # Fallback: textual id column
                        activity_pk = None  # type: ignore

                    row = None
                    if activity_pk is not None and activity_pk in existing_ids:
                        row = db.query(Activities).filter(Activities.activity_id == activity_pk).first()
                        result.updated += 1
                    else:
                        row = Activities()
                        result.created += 1

                    # Populate fields
                    if activity_pk is not None:
                        row.activity_id = activity_pk
                    row.id = act.id
                    row.name = act.name
                    row.activity_type = act.activity_type
                    row.sport_type = act.sport_type
                    row.start_date = act.start_date
                    row.start_date_local = act.start_date_local
                    row.timezone = getattr(act, "timezone", None)
                    row.duration = act.duration
                    row.moving_time = act.moving_time
                    row.distance = act.distance
                    row.average_pace = act.average_pace
                    row.average_speed = act.average_speed
                    row.max_speed = act.max_speed
                    row.elevation_gain = act.elevation_gain
                    row.elev_high = act.elev_high
                    row.elev_low = act.elev_low
                    row.average_heart_rate = act.average_heart_rate
                    row.max_heart_rate = act.max_heart_rate
                    row.has_heartrate = act.has_heartrate
                    row.average_watts = act.average_watts
                    row.max_watts = act.max_watts
                    row.weighted_average_watts = act.weighted_average_watts
                    row.kilojoules = act.kilojoules
                    row.calories = act.calories
                    row.suffer_score = act.suffer_score
                    row.training_load = act.suffer_score  # Map suffer_score to training_load
                    row.source = "strava"
                    row.device_name = act.device_name
                    row.trainer = act.trainer
                    row.commute = act.commute
                    row.manual = act.manual
                    row.raw_data = act.raw_data

                    db.add(row)
                    result.records += 1

                    if result.records % 200 == 0:
                        db.commit()

                db.commit()
                
                # Update training load daily aggregations for affected dates
                if result.records > 0:
                    unique_dates = set()
                    for act in activities:
                        if act and act.start_date:
                            unique_dates.add(act.start_date.date())
                    
                    for activity_date in unique_dates:
                        try:
                            self.update_training_load_daily(activity_date)
                        except Exception as e:
                            self.logger.warning(f"Failed to update training load for {activity_date}: {e}")
                
                self._log_sync(db, sync_type="strava_activities", start=sync_started, status="success",
                               processed=result.records, updated=result.updated, created=result.created, data_range=data_range)
                db.commit()

        except Exception as e:
            self.logger.exception("Strava sync error")
            result.success = False
            result.errors.append(str(e))
            try:
                with get_db_session() as db:
                    self._log_sync(db, sync_type="strava_activities", start=sync_started, status="error",
                                   processed=result.records, updated=result.updated, created=result.created,
                                   data_range=data_range, message=str(e))
                    db.commit()
            except Exception:
                pass

        return result.to_dict()

    def sync_strava_backfill_chunked(self, start_date: date, end_date: date, chunk_days: int = 30) -> Dict[str, Any]:
        sync_started = datetime.utcnow()
        result = SyncResult(success=True, records=0, created=0, updated=0, errors=[])
        data_range = {"start": start_date.isoformat(), "end": end_date.isoformat(), "chunk_days": chunk_days}

        try:
            with get_db_session() as db:
                current_start = start_date
                while current_start <= end_date:
                    current_end = min(current_start + timedelta(days=chunk_days - 1), end_date)
                    acts = self.strava.collect_activities(current_start, current_end)
                    for act in acts:
                        if act is None or act.id is None:
                            continue
                        try:
                            activity_pk = int(act.id)
                        except Exception:
                            activity_pk = None  # type: ignore

                        row = None
                        if activity_pk is not None:
                            row = db.query(Activities).filter(Activities.activity_id == activity_pk).first()
                        if row is None and act.id is not None:
                            row = db.query(Activities).filter(Activities.id == act.id).first()

                        if row is None:
                            row = Activities()
                            result.created += 1
                        else:
                            result.updated += 1

                        if activity_pk is not None:
                            row.activity_id = activity_pk
                        row.id = act.id
                        row.name = act.name
                        row.activity_type = act.activity_type
                        row.sport_type = act.sport_type
                        row.start_date = act.start_date
                        row.start_date_local = act.start_date_local
                        row.timezone = getattr(act, "timezone", None)
                        row.duration = act.duration
                        row.moving_time = act.moving_time
                        row.distance = act.distance
                        row.average_pace = act.average_pace
                        row.average_speed = act.average_speed
                        row.max_speed = act.max_speed
                        row.elevation_gain = act.elevation_gain
                        row.elev_high = act.elev_high
                        row.elev_low = act.elev_low
                        row.average_heart_rate = act.average_heart_rate
                        row.max_heart_rate = act.max_heart_rate
                        row.has_heartrate = act.has_heartrate
                        row.average_watts = act.average_watts
                        row.max_watts = act.max_watts
                        row.weighted_average_watts = act.weighted_average_watts
                        row.kilojoules = act.kilojoules
                        row.calories = act.calories
                        row.suffer_score = act.suffer_score
                        row.source = "strava"
                        row.device_name = act.device_name
                        row.trainer = act.trainer
                        row.commute = act.commute
                        row.manual = act.manual
                        row.raw_data = act.raw_data

                        db.add(row)
                        result.records += 1

                        if result.records % 200 == 0:
                            db.commit()

                    current_start = current_end + timedelta(days=1)

                db.commit()
                self._log_sync(db, sync_type="strava_activities", start=sync_started, status="success",
                               processed=result.records, updated=result.updated, created=result.created, data_range=data_range)
                db.commit()

        except Exception as e:
            self.logger.exception("Strava backfill error")
            result.success = False
            result.errors.append(str(e))
            try:
                with get_db_session() as db:
                    self._log_sync(db, sync_type="strava_activities", start=sync_started, status="error",
                                   processed=result.records, updated=result.updated, created=result.created,
                                   data_range=data_range, message=str(e))
                    db.commit()
            except Exception:
                pass

        return result.to_dict()

    # ---------- Derivatives (minimal implementation) ----------
    def compute_agentic_derivatives(self, start_date: date, end_date: date) -> Dict[str, Any]:
        """Compute minimal derived tables from base data.

        For now:
        - BodyWeightLog: daily weight from garmin_health
        - TrainingLoadDaily/ReadinessDaily left as placeholders with minimal/no-op logic
        """
        started = datetime.utcnow()
        result = SyncResult(success=True, records=0, created=0, updated=0, errors=[])
        try:
            with get_db_session() as db:
                rows: List[HealthMetrics] = (
                    db.query(HealthMetrics)
                    .filter(HealthMetrics.date >= start_date, HealthMetrics.date <= end_date)
                    .order_by(HealthMetrics.date.asc())
                    .all()
                )
                for h in rows:
                    if not isinstance(h.body_weight_kg, (int, float)):
                        continue
                    existing = db.query(BodyWeightLog).filter(BodyWeightLog.date == h.date).first()
                    if existing is None:
                        bw = BodyWeightLog(date=h.date, weight_kg=float(h.body_weight_kg), source="garmin")
                        db.add(bw)
                        result.created += 1
                    else:
                        existing.weight_kg = float(h.body_weight_kg)
                        result.updated += 1
                    result.records += 1
                    if result.records % 200 == 0:
                        db.commit()
                db.commit()
                # Log as a generic derivative sync
                self._log_sync(db, sync_type="derivatives", start=started, status="success",
                               processed=result.records, updated=result.updated, created=result.created,
                               data_range={"start": start_date.isoformat(), "end": end_date.isoformat()})
                db.commit()
        except Exception as e:
            self.logger.exception("Derivative compute error")
            result.success = False
            result.errors.append(str(e))
            try:
                with get_db_session() as db:
                    self._log_sync(db, sync_type="derivatives", start=started, status="error",
                                   processed=result.records, updated=result.updated, created=result.created,
                                   data_range={"start": start_date.isoformat(), "end": end_date.isoformat()},
                                   message=str(e))
                    db.commit()
            except Exception:
                pass
        return result.to_dict()

    # ---------- Status ----------
    def get_sync_status(self) -> Dict[str, Any]:
        try:
            with get_db_session() as db:
                latest_garmin = (
                    db.query(DataSyncLog)
                    .filter(DataSyncLog.sync_type == "garmin_health")
                    .order_by(DataSyncLog.created_at.desc())
                    .first()
                )
                latest_strava = (
                    db.query(DataSyncLog)
                    .filter(DataSyncLog.sync_type == "strava_activities")
                    .order_by(DataSyncLog.created_at.desc())
                    .first()
                )
                latest_health_date = (
                    db.query(HealthMetrics)
                    .order_by(HealthMetrics.date.desc())
                    .first()
                )
                latest_activity = (
                    db.query(Activities)
                    .order_by(Activities.start_date.desc())
                    .first()
                )

                return {
                    "last_sync": {
                        "garmin": latest_garmin.completed_at.isoformat() if latest_garmin else None,
                        "strava": latest_strava.completed_at.isoformat() if latest_strava else None,
                    },
                    "data_freshness": {
                        "latest_health_date": latest_health_date.date.isoformat() if latest_health_date else None,
                        "latest_activity_date": latest_activity.start_date.isoformat() if latest_activity else None,
                    },
                    "last_status": {
                        "garmin": (latest_garmin.status if latest_garmin else "unknown"),
                        "strava": (latest_strava.status if latest_strava else "unknown"),
                    },
                }
        except Exception as e:
            return {"error": str(e)}

    # ---------- Helpers ----------
    def _log_sync(
        self,
        db,
        *,
        sync_type: str,
        start: datetime,
        status: str,
        processed: int,
        updated: int,
        created: int,
        data_range: Dict[str, Any] = None,  # type: ignore
        message: Optional[str] = None,
    ) -> None:
        now = datetime.utcnow()
        log = DataSyncLog(
            sync_type=sync_type,
            sync_date=date.today(),
            status=status,
            records_processed=processed,
            records_updated=updated,
            records_created=created,
            started_at=start,
            completed_at=now,
            duration_seconds=(now - start).total_seconds(),
            message=message,
            data_range=data_range or {},
            version=settings.version,
        )
        db.add(log)

    def update_training_load_daily(self, target_date: date) -> None:
        """Update TrainingLoadDaily record for a specific date"""
        from datetime import datetime
        
        with get_db_session() as db:
            # Calculate daily load for target date
            daily_load = (
                db.query(Activities.training_load)
                .filter(Activities.start_date >= datetime.combine(target_date, datetime.min.time()))
                .filter(Activities.start_date <= datetime.combine(target_date, datetime.max.time()))
                .filter(Activities.training_load.isnot(None))
            )
            daily_total = sum(float(load[0]) for load in daily_load if load[0] is not None)
            
            # Get historical data for rolling averages (last 28 days)
            start_calc = target_date - timedelta(days=27)
            historical_activities = (
                db.query(Activities)
                .filter(Activities.start_date >= datetime.combine(start_calc, datetime.min.time()))
                .filter(Activities.start_date <= datetime.combine(target_date, datetime.max.time()))
                .filter(Activities.training_load.isnot(None))
                .all()
            )
            
            # Group by date
            daily_loads = {}
            for activity in historical_activities:
                activity_date = activity.start_date.date()
                if activity_date not in daily_loads:
                    daily_loads[activity_date] = 0.0
                daily_loads[activity_date] += float(activity.training_load or 0)
            
            # Calculate rolling averages
            all_dates = sorted([d for d in daily_loads.keys() if d <= target_date])
            
            if len(all_dates) >= 7:
                # 7-day acute load
                acute_dates = [d for d in all_dates if (target_date - d).days < 7]
                acute_load = sum(daily_loads.get(d, 0) for d in acute_dates) / len(acute_dates)
                
                # 28-day chronic load
                chronic_dates = [d for d in all_dates if (target_date - d).days < 28]
                chronic_load = sum(daily_loads.get(d, 0) for d in chronic_dates) / len(chronic_dates)
                
                # Determine training status
                if chronic_load > 0:
                    ratio = acute_load / chronic_load
                    if ratio > 1.5:
                        training_status = "Overreaching"
                    elif ratio > 1.3:
                        training_status = "Functional Overreaching"
                    elif ratio > 0.8:
                        training_status = "Maintaining"
                    else:
                        training_status = "Detraining"
                else:
                    training_status = "Insufficient Data"
                
                # Update or create record
                existing = db.query(TrainingLoadDaily).filter(TrainingLoadDaily.date == target_date).first()
                
                if existing:
                    existing.daily_load = daily_total
                    existing.acute_load_7d = acute_load
                    existing.chronic_load_28d = chronic_load
                    existing.training_status = training_status
                    existing.updated_at = datetime.utcnow()
                else:
                    new_record = TrainingLoadDaily(
                        date=target_date,
                        daily_load=daily_total,
                        acute_load_7d=acute_load,
                        chronic_load_28d=chronic_load,
                        training_status=training_status
                    )
                    db.add(new_record)
                
                db.commit()
                self.logger.info(f"Updated training load for {target_date}: daily={daily_total:.1f}, acute={acute_load:.1f}, chronic={chronic_load:.1f}, status={training_status}")


