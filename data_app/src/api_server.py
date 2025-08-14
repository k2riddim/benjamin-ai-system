"""
FastAPI server for Benjamin AI Data App
Provides API endpoints for accessing health and activity data
"""

import logging
from datetime import date, datetime, timedelta
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, HTTPException, Depends, Query, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

# Import shared database components
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from shared.database import get_db, HealthMetrics, Activities, DataSyncLog, SystemHealth, ReadinessDaily, TrainingLoadDaily, BodyWeightLog, SubjectiveFeedbackDaily
from data_app.config.settings import settings
from .data_synchronizer import DataSynchronizer


# Pydantic models for API responses
class HealthDataResponse(BaseModel):
    date: date
    steps: Optional[int] = None
    resting_heart_rate: Optional[int] = None
    sleep_score: Optional[float] = None
    hrv_score: Optional[float] = None
    hrv_weekly_avg: Optional[float] = None
    hrv_status: Optional[str] = None
    stress_score: Optional[float] = None
    max_stress: Optional[float] = None
    body_battery: Optional[float] = None
    training_readiness: Optional[float] = None
    sleep_hours: Optional[float] = None
    deep_sleep_minutes: Optional[int] = None
    rem_sleep_minutes: Optional[int] = None
    body_weight_kg: Optional[float] = None
    vo2max_running: Optional[float] = None
    vo2max_cycling: Optional[float] = None
    fitness_age: Optional[float] = None
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class ActivityResponse(BaseModel):
    id: str
    name: str
    activity_type: str
    sport_type: Optional[str] = None
    start_date: datetime
    start_date_local: Optional[datetime] = None
    duration: Optional[float] = None  # minutes
    moving_time: Optional[float] = None  # minutes
    distance: Optional[float] = None  # km
    average_pace: Optional[float] = None  # min/km
    average_speed: Optional[float] = None  # km/h
    max_speed: Optional[float] = None
    elevation_gain: Optional[float] = None
    average_heart_rate: Optional[float] = None
    max_heart_rate: Optional[float] = None
    has_heartrate: Optional[bool] = None
    average_watts: Optional[float] = None
    calories: Optional[float] = None
    suffer_score: Optional[float] = None
    source: Optional[str] = None
    created_at: datetime

    class Config:
        from_attributes = True


class SyncStatusResponse(BaseModel):
    last_sync: Dict[str, Optional[str]]
    data_freshness: Dict[str, Optional[str]]
    last_status: Dict[str, str]


class HealthReportResponse(BaseModel):
    timestamp: str
    overall_status: str
    components: Dict[str, Dict[str, Any]]
    data_freshness: Dict[str, Any]
    data_metrics: Dict[str, Any]


# Agent mega-dashboard
class AgentContextResponse(BaseModel):
    latest_health: Dict[str, Any]
    readiness_recent: List[Dict[str, Any]]
    training_load: List[Dict[str, Any]]
    weight_recent: List[Dict[str, Any]]
    activities_recent: List[Dict[str, Any]]
    stats: Dict[str, Any]
    trends: Dict[str, Any] | None = None
    baselines: Dict[str, Any] | None = None


# FastAPI app
app = FastAPI(
    title="Benjamin AI Data API",
    description="API for accessing Benjamin's health and activity data",
    version=settings.version,
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global components
synchronizer = DataSynchronizer()


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information"""
    return {
        "service": "Benjamin AI Data API",
        "version": settings.version,
        "status": "running",
        "description": "API for accessing health and activity data"
    }


@app.get("/health", response_model=Dict[str, str])
async def health_check():
    """Health check endpoint"""
    try:
        # Quick database connectivity test
        from sqlalchemy import text
        with next(get_db()) as db:
            db.execute(text("SELECT 1"))
        
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "database": "connected"
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Health check failed: {str(e)}"
        )


@app.get("/health/detailed")
async def detailed_health():
    return {"status": "deprecated"}


@app.get("/sync/status", response_model=SyncStatusResponse)
async def sync_status():
    """Get current synchronization status"""
    try:
        status_data = synchronizer.get_sync_status()
        if 'error' in status_data:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=status_data['error']
            )
        return SyncStatusResponse(**status_data)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get sync status: {str(e)}"
        )


@app.post("/sync/trigger")
async def trigger_sync(
    days_back: Optional[int] = Query(None, description="Days back to sync"),
    force: bool = Query(False, description="Force sync even if recent data exists")
):
    """Trigger manual data synchronization"""
    try:
        if days_back is None:
            days_back = settings.sync_lookback_days
        
        logger.info(f"Manual sync triggered for {days_back} days back")
        
        results = synchronizer.sync_all_data(days_back)
        
        return {
            "message": "Sync completed",
            "results": results,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Sync failed: {str(e)}"
        )


@app.post("/sync/backfill")
async def sync_backfill(
    start_date: date = Query(..., description="Start date YYYY-MM-DD"),
    end_date: date = Query(..., description="End date YYYY-MM-DD"),
    chunk_days: int = Query(30, description="Chunk size for Strava backfill"),
):
    """Backfill all data for an explicit date range (large historical sync)."""
    try:
        # Use extended synchronizer backfill
        # Note: chunk_days is used inside synchronizer; exposed for flexibility
        results = synchronizer.sync_all_data_range(start_date, end_date)
        return {
            "message": "Backfill completed",
            "results": results,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Backfill failed: {str(e)}"
        )


@app.get("/health-data", response_model=List[HealthDataResponse])
async def get_health_data(
    start_date: Optional[date] = Query(None, description="Start date (YYYY-MM-DD)"),
    end_date: Optional[date] = Query(None, description="End date (YYYY-MM-DD)"),
    limit: int = Query(30, description="Maximum number of records"),
    db: Session = Depends(get_db)
):
    """Get health data for date range"""
    try:
        query = db.query(HealthMetrics)
        
        if start_date:
            query = query.filter(HealthMetrics.date >= start_date)
        if end_date:
            query = query.filter(HealthMetrics.date <= end_date)
        
        health_data = query.order_by(HealthMetrics.date.desc()).limit(limit).all()
        
        return [HealthDataResponse.from_orm(data) for data in health_data]
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get health data: {str(e)}"
        )


@app.get("/health-data/latest", response_model=Optional[HealthDataResponse])
async def get_latest_health_data(db: Session = Depends(get_db)):
    """Get most recent health data"""
    try:
        latest = db.query(HealthMetrics).order_by(HealthMetrics.date.desc()).first()
        
        if not latest:
            return None
        
        return HealthDataResponse.from_orm(latest)
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get latest health data: {str(e)}"
        )


@app.get("/health-data/range", response_model=List[HealthDataResponse])
async def get_health_data_range(
    start_date: date = Query(..., description="Start date (YYYY-MM-DD)"),
    end_date: date = Query(..., description="End date (YYYY-MM-DD)"),
    limit: int = Query(60, description="Maximum number of records"),
    db: Session = Depends(get_db)
):
    """Get health data within a specific date range (inclusive)."""
    try:
        query = db.query(HealthMetrics).filter(HealthMetrics.date >= start_date, HealthMetrics.date <= end_date)
        health_data = query.order_by(HealthMetrics.date.desc()).limit(limit).all()
        return [HealthDataResponse.from_orm(data) for data in health_data]
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get health data for range {start_date} to {end_date}: {str(e)}"
        )


@app.get("/health-data/{target_date}", response_model=Optional[HealthDataResponse])
async def get_health_data_by_date(
    target_date: date,
    db: Session = Depends(get_db)
):
    """Get health data for specific date"""
    try:
        health_data = db.query(HealthMetrics).filter(
            HealthMetrics.date == target_date
        ).first()
        
        if not health_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No health data found for {target_date}"
            )
        
        return HealthDataResponse.from_orm(health_data)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get health data for {target_date}: {str(e)}"
        )


@app.get("/activities", response_model=List[ActivityResponse])
async def get_activities(
    start_date: Optional[datetime] = Query(None, description="Start datetime"),
    end_date: Optional[datetime] = Query(None, description="End datetime"),
    activity_type: Optional[str] = Query(None, description="Activity type filter"),
    limit: int = Query(50, description="Maximum number of activities"),
    db: Session = Depends(get_db)
):
    """Get activities for date range"""
    try:
        query = db.query(Activities).filter(Activities.activity_id.isnot(None))
        
        if start_date:
            query = query.filter(Activities.start_date >= start_date)
        if end_date:
            query = query.filter(Activities.start_date <= end_date)
        if activity_type:
            query = query.filter(Activities.activity_type == activity_type)
        
        activities = query.order_by(Activities.start_date.desc()).limit(limit).all()
        # Filter out any None rows defensively
        safe_rows = [a for a in activities if a is not None]
        return [ActivityResponse.from_orm(activity) for activity in safe_rows]
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get activities: {str(e)}"
        )


@app.get("/activities/recent", response_model=List[ActivityResponse])
async def get_recent_activities(
    days: int = Query(7, description="Number of days back"),
    limit: int = Query(20, description="Maximum number of activities"),
    db: Session = Depends(get_db)
):
    """Get recent activities"""
    try:
        cutoff_date = datetime.now() - timedelta(days=days)
        
        activities = db.query(Activities).filter(
            Activities.activity_id.isnot(None),
            Activities.start_date >= cutoff_date
        ).order_by(Activities.start_date.desc()).limit(limit).all()
        
        logger.info(f"Found {len(activities)} activities in last {days} days")
        
        safe_rows = [a for a in activities if a is not None]
        return [ActivityResponse.from_orm(activity) for activity in safe_rows]
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get recent activities: {str(e)}"
        )


@app.get("/activities/{activity_id}", response_model=ActivityResponse)
async def get_activity(
    activity_id: str,
    db: Session = Depends(get_db)
):
    """Get specific activity by ID"""
    try:
        # Accept either the bigint primary key or the textual id
        activity = None
        try:
            activity_pk = int(activity_id)
            activity = db.query(Activities).filter(Activities.activity_id == activity_pk).first()
        except Exception:
            activity = db.query(Activities).filter(Activities.id == activity_id).first()
        
        if not activity:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Activity {activity_id} not found"
            )
        
        return ActivityResponse.from_orm(activity)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get activity {activity_id}: {str(e)}"
        )


@app.get("/stats/summary")
async def get_stats_summary(db: Session = Depends(get_db)):
    """Get summary statistics"""
    try:
        # Total counts
        total_health_records = db.query(HealthMetrics).count()
        total_activities = db.query(Activities).count()
        
        # Recent data
        week_ago = date.today() - timedelta(days=7)
        recent_health = db.query(HealthMetrics).filter(
            HealthMetrics.date >= week_ago
        ).count()
        
        week_ago_datetime = datetime.now() - timedelta(days=7)
        recent_activities = db.query(Activities).filter(
            Activities.start_date >= week_ago_datetime
        ).count()
        
        # Latest dates
        latest_health = db.query(HealthMetrics).order_by(
            HealthMetrics.date.desc()
        ).first()
        
        latest_activity = db.query(Activities).order_by(
            Activities.start_date.desc()
        ).first()
        
        return {
            "total_health_records": total_health_records,
            "total_activities": total_activities,
            "recent_health_records": recent_health,
            "recent_activities": recent_activities,
            "latest_health_date": latest_health.date.isoformat() if latest_health else None,
            "latest_activity_date": latest_activity.start_date.isoformat() if latest_activity else None,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get stats summary: {str(e)}"
        )


# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={"detail": "Endpoint not found"}
    )


# New endpoint: one-shot payload with maximum useful agent context
@app.get("/dashboard/agent-context", response_model=AgentContextResponse)
async def get_agent_context(db: Session = Depends(get_db)):
    try:
        latest = db.query(HealthMetrics).order_by(HealthMetrics.date.desc()).first()
        latest_payload = HealthDataResponse.from_orm(latest).model_dump() if latest else {}

        # Readiness recent (7d)
        cutoff = date.today() - timedelta(days=7)
        r_rows = db.query(ReadinessDaily).filter(ReadinessDaily.date >= cutoff).order_by(ReadinessDaily.date.desc()).all()
        readiness_payload = [
            {
                "date": r.date.isoformat(),
                "readiness_flag": r.readiness_flag,
                "hrv_status": r.hrv_status,
                "hrv_overnight_avg": r.hrv_overnight_avg,
                "rhr": r.rhr,
                "sleep_hours": r.sleep_hours,
                "sleep_score": r.sleep_score,
                "acute_load_7d": r.acute_load_7d,
                "chronic_load_28d": r.chronic_load_28d,
                "training_status": r.training_status,
            }
            for r in r_rows
        ]

        # Training load (28d)
        cutoff28 = date.today() - timedelta(days=28)
        tl_rows = db.query(TrainingLoadDaily).filter(TrainingLoadDaily.date >= cutoff28).order_by(TrainingLoadDaily.date).all()
        tl_payload = [
            {
                "date": r.date.isoformat(),
                "daily_load": r.daily_load,
                "acute_load_7d": r.acute_load_7d,
                "chronic_load_28d": r.chronic_load_28d,
                "training_status": r.training_status,
            }
            for r in tl_rows
        ]

        # Weight (60d)
        cutoff60 = date.today() - timedelta(days=60)
        w_rows = db.query(BodyWeightLog).filter(BodyWeightLog.date >= cutoff60).order_by(BodyWeightLog.date).all()
        w_payload = [
            {
                "date": r.date.isoformat(),
                "weight_kg": r.weight_kg,
                "trend_28d": r.trend_28d,
            }
            for r in w_rows
        ]

        # Activities (14d, last 50)
        cutoff_act = datetime.now() - timedelta(days=14)
        a_rows = db.query(Activities).filter(Activities.start_date >= cutoff_act).order_by(Activities.start_date.desc()).limit(50).all()
        a_payload = [
            {
                "id": a.id,
                "activity_type": a.activity_type,
                "sport_type": a.sport_type,
                "start_date": a.start_date.isoformat() if a.start_date else None,
                "duration": a.duration,
                "distance": a.distance,
                "average_heart_rate": a.average_heart_rate,
                "average_watts": a.average_watts,
            }
            for a in a_rows
            if a is not None
        ]

        # Stats
        stats = {
            "total_health_records": db.query(HealthMetrics).count(),
            "total_activities": db.query(Activities).count(),
            "latest_health_date": latest.date.isoformat() if latest else None,
        }

        # ---- Trend packs across multiple windows ----
        def summarize_numeric_series(rows, attr: str, date_attr: str = "date"):
            values = []
            dates = []
            for r in rows:
                v = getattr(r, attr, None)
                if isinstance(v, (int, float)):
                    values.append(float(v))
                    d = getattr(r, date_attr, None)
                    dates.append(d)
            if not values or not dates:
                return None
            first = values[0]
            last = values[-1]
            days = max(1, len(values) - 1)
            return {
                "count": len(values),
                "first": first,
                "last": last,
                "delta": (last - first),
                "min": min(values),
                "max": max(values),
                "avg": sum(values) / len(values),
                "slope_per_day": (last - first) / days,
                "start_date": dates[0].isoformat() if dates[0] else None,
                "end_date": dates[-1].isoformat() if dates[-1] else None,
            }

        def build_window_trends(days_back: int) -> Dict[str, Any]:
            start = date.today() - timedelta(days=days_back)
            h_rows = (
                db.query(HealthMetrics)
                .filter(HealthMetrics.date >= start)
                .order_by(HealthMetrics.date.asc())
                .all()
            )
            tl_rows = (
                db.query(TrainingLoadDaily)
                .filter(TrainingLoadDaily.date >= start)
                .order_by(TrainingLoadDaily.date.asc())
                .all()
            )
            a_rows = (
                db.query(Activities)
                .filter(Activities.start_date >= datetime.now() - timedelta(days=days_back))
                .order_by(Activities.start_date.asc())
                .all()
            )

            trend = {
                "weight_kg": summarize_numeric_series(h_rows, "body_weight_kg"),
                "vo2max_running": summarize_numeric_series(h_rows, "vo2max_running"),
                "vo2max_cycling": summarize_numeric_series(h_rows, "vo2max_cycling"),
                "resting_heart_rate": summarize_numeric_series(h_rows, "resting_heart_rate"),
                "hrv_score": summarize_numeric_series(h_rows, "hrv_score"),
                "sleep_hours": summarize_numeric_series(h_rows, "sleep_hours"),
            }

            # Training load aggregates
            if tl_rows:
                try:
                    acute_vals = [float(r.acute_load_7d) for r in tl_rows if isinstance(r.acute_load_7d, (int, float))]
                    chronic_vals = [float(r.chronic_load_28d) for r in tl_rows if isinstance(r.chronic_load_28d, (int, float))]
                    trend["training_load"] = {
                        "last_acute_7d": acute_vals[-1] if acute_vals else None,
                        "last_chronic_28d": chronic_vals[-1] if chronic_vals else None,
                        "avg_acute_7d": (sum(acute_vals) / len(acute_vals)) if acute_vals else None,
                        "avg_chronic_28d": (sum(chronic_vals) / len(chronic_vals)) if chronic_vals else None,
                    }
                except Exception:
                    trend["training_load"] = {}
            else:
                trend["training_load"] = {}

            # Activities aggregates
            if a_rows:
                try:
                    total_dist = sum(float(a.distance) for a in a_rows if isinstance(a.distance, (int, float)))
                    total_dur = sum(float(a.duration) for a in a_rows if isinstance(a.duration, (int, float)))
                    hr_vals = [float(a.average_heart_rate) for a in a_rows if isinstance(a.average_heart_rate, (int, float))]
                    trend["activities"] = {
                        "count": len(a_rows),
                        "total_distance_km": total_dist,
                        "total_duration_min": total_dur,
                        "avg_hr": (sum(hr_vals) / len(hr_vals)) if hr_vals else None,
                    }
                except Exception:
                    trend["activities"] = {"count": len(a_rows)}
            else:
                trend["activities"] = {"count": 0}

            return trend

        windows = {"30d": 30, "3m": 90, "6m": 180, "1y": 365, "2y": 730}
        trends: Dict[str, Any] = {}
        for key, dd in windows.items():
            try:
                trends[key] = build_window_trends(dd)
            except Exception:
                trends[key] = {}

        # ---- Baseline packs (long-horizon percentiles + deviation vs latest) ----
        # Compute baselines over a long window (default 180 days, fallback 365 if sparse)
        from statistics import mean, median, pstdev

        def _percentile(sorted_vals: List[float], p: float) -> float | None:
            if not sorted_vals:
                return None
            if p <= 0:
                return sorted_vals[0]
            if p >= 1:
                return sorted_vals[-1]
            k = (len(sorted_vals) - 1) * p
            f = int(k)
            c = min(f + 1, len(sorted_vals) - 1)
            if f == c:
                return sorted_vals[f]
            d0 = sorted_vals[f] * (c - k)
            d1 = sorted_vals[c] * (k - f)
            return d0 + d1

        def _safe_stats(values: List[float]) -> Dict[str, Any]:
            if not values:
                return {}
            vals = sorted(values)
            avg = mean(vals)
            med = median(vals)
            try:
                sd = pstdev(vals)
            except Exception:
                sd = 0.0
            return {
                "count": len(vals),
                "mean": avg,
                "median": med,
                "std": sd,
                "p10": _percentile(vals, 0.10),
                "p25": _percentile(vals, 0.25),
                "p75": _percentile(vals, 0.75),
                "p90": _percentile(vals, 0.90),
                "min": vals[0],
                "max": vals[-1],
            }

        def _deviation(current: float | None, stats: Dict[str, Any]) -> Dict[str, Any]:
            if current is None or not stats:
                return {"current": current}
            mu = stats.get("mean")
            sd = stats.get("std") or 0.0
            if isinstance(mu, (int, float)):
                delta = float(current) - float(mu)
                z = (delta / sd) if (isinstance(sd, (int, float)) and sd > 1e-6) else None
                band_low = stats.get("p25")
                band_high = stats.get("p75")
                flag = None
                try:
                    if band_low is not None and float(current) < float(band_low):
                        flag = "below_baseline"
                    elif band_high is not None and float(current) > float(band_high):
                        flag = "above_baseline"
                    else:
                        flag = "within_baseline"
                except Exception:
                    flag = None
                return {"current": current, "delta": delta, "z_score": z, "flag": flag}
            return {"current": current}

        # Historical rows for baseline window
        baseline_days = 180
        h_start = date.today() - timedelta(days=baseline_days)
        hist_h = (
            db.query(HealthMetrics)
            .filter(HealthMetrics.date >= h_start)
            .order_by(HealthMetrics.date.asc())
            .all()
        )
        # If too few, try 365d
        if len(hist_h) < 30:
            h_start = date.today() - timedelta(days=365)
            hist_h = (
                db.query(HealthMetrics)
                .filter(HealthMetrics.date >= h_start)
                .order_by(HealthMetrics.date.asc())
                .all()
            )

        def _extract(vals, attr: str) -> List[float]:
            out: List[float] = []
            for r in vals:
                v = getattr(r, attr, None)
                if isinstance(v, (int, float)):
                    out.append(float(v))
            return out

        baselines: Dict[str, Any] = {}

        # Core health metrics
        for key in [
            "resting_heart_rate",
            "hrv_score",
            "sleep_hours",
            "stress_score",
            "body_battery",
            "training_readiness",
            "vo2max_running",
            "vo2max_cycling",
            "body_weight_kg",
            "steps",
        ]:
            series = _extract(hist_h, key)
            stats_pack = _safe_stats(series)
            current_val = latest_payload.get(key) if isinstance(latest_payload, dict) else None
            baselines[key] = {
                "window_days": (date.today() - h_start).days,
                "stats": stats_pack,
                "deviation": _deviation(current_val, stats_pack),
            }

        # Training load baseline (from TrainingLoadDaily)
        tl_start = date.today() - timedelta(days=180)
        hist_tl = (
            db.query(TrainingLoadDaily)
            .filter(TrainingLoadDaily.date >= tl_start)
            .order_by(TrainingLoadDaily.date.asc())
            .all()
        )
        if hist_tl:
            acute_vals = [float(r.acute_load_7d) for r in hist_tl if isinstance(r.acute_load_7d, (int, float))]
            chronic_vals = [float(r.chronic_load_28d) for r in hist_tl if isinstance(r.chronic_load_28d, (int, float))]
            acute_stats = _safe_stats(acute_vals) if acute_vals else {}
            chronic_stats = _safe_stats(chronic_vals) if chronic_vals else {}
            last_tl = hist_tl[-1]
            baselines["training_load"] = {
                "acute_7d": {
                    "stats": acute_stats,
                    "deviation": _deviation(getattr(last_tl, "acute_load_7d", None), acute_stats),
                },
                "chronic_28d": {
                    "stats": chronic_stats,
                    "deviation": _deviation(getattr(last_tl, "chronic_load_28d", None), chronic_stats),
                },
            }

        # Activities baseline: weekly sessions and duration over last 12 weeks
        act_start = datetime.now() - timedelta(days=90)
        hist_act = (
            db.query(Activities)
            .filter(Activities.start_date >= act_start)
            .order_by(Activities.start_date.asc())
            .all()
        )
        if hist_act:
            weekly: Dict[str, Dict[str, float]] = {}
            for a in hist_act:
                if not a or not a.start_date:
                    continue
                year_week = a.start_date.strftime("%G-W%V")
                w = weekly.setdefault(year_week, {"count": 0.0, "duration_min": 0.0, "distance_km": 0.0})
                w["count"] += 1.0
                if isinstance(a.duration, (int, float)):
                    w["duration_min"] += float(a.duration)
                if isinstance(a.distance, (int, float)):
                    w["distance_km"] += float(a.distance)
            counts = [v["count"] for v in weekly.values()] or []
            durs = [v["duration_min"] for v in weekly.values()] or []
            dists = [v["distance_km"] for v in weekly.values()] or []
            counts_stats = _safe_stats(counts) if counts else {}
            durs_stats = _safe_stats(durs) if durs else {}
            dists_stats = _safe_stats(dists) if dists else {}
            # Deviation vs most recent week
            last_week_key = sorted(weekly.keys())[-1] if weekly else None
            last_week = weekly.get(last_week_key, {}) if last_week_key else {}
            baselines["activities_weekly"] = {
                "window_weeks": len(weekly),
                "count": {"stats": counts_stats, "deviation": _deviation(last_week.get("count"), counts_stats)},
                "duration_min": {"stats": durs_stats, "deviation": _deviation(last_week.get("duration_min"), durs_stats)},
                "distance_km": {"stats": dists_stats, "deviation": _deviation(last_week.get("distance_km"), dists_stats)},
            }

        return AgentContextResponse(
            latest_health=latest_payload,
            readiness_recent=readiness_payload,
            training_load=tl_payload,
            weight_recent=w_payload,
            activities_recent=a_payload,
            stats=stats,
            trends=trends,
            baselines=baselines,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to build agent context: {e}")
@app.exception_handler(500)
async def internal_error_handler(request, exc):
    logger.error(f"Internal server error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )


# Agentic-ready endpoints

class ReadinessResponse(BaseModel):
    date: date
    readiness_flag: str
    readiness_notes: Optional[str] = None
    hrv_status: Optional[str] = None
    hrv_overnight_avg: Optional[float] = None
    hrv_7d_avg: Optional[float] = None
    rhr: Optional[float] = None
    rhr_7d_avg: Optional[float] = None
    sleep_hours: Optional[float] = None
    sleep_score: Optional[float] = None
    acute_load_7d: Optional[float] = None
    chronic_load_28d: Optional[float] = None
    weight_kg: Optional[float] = None
    weight_trend_28d: Optional[float] = None
    training_status: Optional[str] = None
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True


@app.get("/readiness/recent", response_model=List[ReadinessResponse])
async def get_recent_readiness(days: int = Query(7), db: Session = Depends(get_db)):
    cutoff = date.today() - timedelta(days=days)
    rows = db.query(ReadinessDaily).filter(ReadinessDaily.date >= cutoff).order_by(ReadinessDaily.date.desc()).all()
    return [ReadinessResponse.from_orm(r) for r in rows]


@app.get("/training-load/recent")
async def get_training_load(days: int = Query(28), db: Session = Depends(get_db)):
    cutoff = date.today() - timedelta(days=days)
    rows = db.query(TrainingLoadDaily).filter(TrainingLoadDaily.date >= cutoff).order_by(TrainingLoadDaily.date).all()
    return [{
        "date": r.date.isoformat(),
        "daily_load": r.daily_load,
        "acute_load_7d": r.acute_load_7d,
        "chronic_load_28d": r.chronic_load_28d,
        "training_status": r.training_status
    } for r in rows]


@app.get("/weight/recent")
async def get_weight(days: int = Query(60), db: Session = Depends(get_db)):
    cutoff = date.today() - timedelta(days=days)
    rows = db.query(BodyWeightLog).filter(BodyWeightLog.date >= cutoff).order_by(BodyWeightLog.date).all()
    return [{
        "date": r.date.isoformat(),
        "weight_kg": r.weight_kg,
        "trend_28d": r.trend_28d,
        "source": r.source
    } for r in rows]


# Lightweight endpoint to fetch the latest weight directly from garmin_health, mirroring the probe logic
@app.get("/weight/latest")
async def get_latest_weight(db: Session = Depends(get_db)):
    try:
        # Prefer today's row if present
        today_row = db.query(HealthMetrics).filter(HealthMetrics.date == date.today()).first()
        if today_row and isinstance(getattr(today_row, "body_weight_kg", None), (int, float)):
            return {"date": today_row.date.isoformat(), "weight_kg": float(today_row.body_weight_kg)}

        # Else latest non-null weight across history
        latest_row = (
            db.query(HealthMetrics)
            .filter(HealthMetrics.body_weight_kg.isnot(None))
            .order_by(HealthMetrics.date.desc())
            .first()
        )
        if latest_row and isinstance(getattr(latest_row, "body_weight_kg", None), (int, float)):
            return {"date": latest_row.date.isoformat(), "weight_kg": float(latest_row.body_weight_kg)}

        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No weight found")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to get latest weight: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api_server:app",
        host="127.0.0.1",
        port=8010,
        reload=settings.debug,
        log_level=settings.log_level.lower()
    )


# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={"detail": "Endpoint not found"}
    )


@app.exception_handler(500)
async def internal_error_handler(request, exc):
    logger.error(f"Internal server error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )


# Agentic-ready endpoints

class ReadinessResponse(BaseModel):
    date: date
    readiness_flag: str
    readiness_notes: Optional[str] = None
    hrv_status: Optional[str] = None
    hrv_overnight_avg: Optional[float] = None
    hrv_7d_avg: Optional[float] = None
    rhr: Optional[float] = None
    rhr_7d_avg: Optional[float] = None
    sleep_hours: Optional[float] = None
    sleep_score: Optional[float] = None
    acute_load_7d: Optional[float] = None
    chronic_load_28d: Optional[float] = None
    weight_kg: Optional[float] = None
    weight_trend_28d: Optional[float] = None
    training_status: Optional[str] = None
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True


@app.get("/readiness/recent", response_model=List[ReadinessResponse])
async def get_recent_readiness(days: int = Query(7), db: Session = Depends(get_db)):
    cutoff = date.today() - timedelta(days=days)
    rows = db.query(ReadinessDaily).filter(ReadinessDaily.date >= cutoff).order_by(ReadinessDaily.date.desc()).all()
    return [ReadinessResponse.from_orm(r) for r in rows]


@app.get("/training-load/recent")
async def get_training_load(days: int = Query(28), db: Session = Depends(get_db)):
    cutoff = date.today() - timedelta(days=days)
    rows = db.query(TrainingLoadDaily).filter(TrainingLoadDaily.date >= cutoff).order_by(TrainingLoadDaily.date).all()
    return [{
        "date": r.date.isoformat(),
        "daily_load": r.daily_load,
        "acute_load_7d": r.acute_load_7d,
        "chronic_load_28d": r.chronic_load_28d,
        "training_status": r.training_status
    } for r in rows]


@app.get("/weight/recent")
async def get_weight(days: int = Query(60), db: Session = Depends(get_db)):
    cutoff = date.today() - timedelta(days=days)
    rows = db.query(BodyWeightLog).filter(BodyWeightLog.date >= cutoff).order_by(BodyWeightLog.date).all()
    return [{
        "date": r.date.isoformat(),
        "weight_kg": r.weight_kg,
        "trend_28d": r.trend_28d,
        "source": r.source
    } for r in rows]


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api_server:app",
        host="127.0.0.1",
        port=8010,
        reload=settings.debug,
        log_level=settings.log_level.lower()
    )


# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={"detail": "Endpoint not found"}
    )


@app.exception_handler(500)
async def internal_error_handler(request, exc):
    logger.error(f"Internal server error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )


# Agentic-ready endpoints

class ReadinessResponse(BaseModel):
    date: date
    readiness_flag: str
    readiness_notes: Optional[str] = None
    hrv_status: Optional[str] = None
    hrv_overnight_avg: Optional[float] = None
    hrv_7d_avg: Optional[float] = None
    rhr: Optional[float] = None
    rhr_7d_avg: Optional[float] = None
    sleep_hours: Optional[float] = None
    sleep_score: Optional[float] = None
    acute_load_7d: Optional[float] = None
    chronic_load_28d: Optional[float] = None
    weight_kg: Optional[float] = None
    weight_trend_28d: Optional[float] = None
    training_status: Optional[str] = None
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True


@app.get("/readiness/recent", response_model=List[ReadinessResponse])
async def get_recent_readiness(days: int = Query(7), db: Session = Depends(get_db)):
    cutoff = date.today() - timedelta(days=days)
    rows = db.query(ReadinessDaily).filter(ReadinessDaily.date >= cutoff).order_by(ReadinessDaily.date.desc()).all()
    return [ReadinessResponse.from_orm(r) for r in rows]


@app.get("/training-load/recent")
async def get_training_load(days: int = Query(28), db: Session = Depends(get_db)):
    cutoff = date.today() - timedelta(days=days)
    rows = db.query(TrainingLoadDaily).filter(TrainingLoadDaily.date >= cutoff).order_by(TrainingLoadDaily.date).all()
    return [{
        "date": r.date.isoformat(),
        "daily_load": r.daily_load,
        "acute_load_7d": r.acute_load_7d,
        "chronic_load_28d": r.chronic_load_28d,
        "training_status": r.training_status
    } for r in rows]


@app.get("/weight/recent")
async def get_weight(days: int = Query(60), db: Session = Depends(get_db)):
    cutoff = date.today() - timedelta(days=days)
    rows = db.query(BodyWeightLog).filter(BodyWeightLog.date >= cutoff).order_by(BodyWeightLog.date).all()
    return [{
        "date": r.date.isoformat(),
        "weight_kg": r.weight_kg,
        "trend_28d": r.trend_28d,
        "source": r.source
    } for r in rows]


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api_server:app",
        host="127.0.0.1",
        port=8010,
        reload=settings.debug,
        log_level=settings.log_level.lower()
    )