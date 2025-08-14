#!/usr/bin/env python3
from __future__ import annotations

from datetime import datetime
import os, sys
import logging

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(PROJECT_ROOT)

from shared.database import get_db_session, HealthMetrics  # noqa: E402
from sqlalchemy import or_  # type: ignore
from data_app.src.garmin_collector import GarminCollector  # noqa: E402


def main() -> int:
    # Configure real-time logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger("garmin_backfill_official")

    collector = GarminCollector()
    if not collector.test_connection():
        logger.error("Garmin auth failed. Ensure GARTH_HOME tokens are present.")
        return 1
    updated = 0
    updated_steps = 0
    updated_vo2 = 0
    updated_fitness_age = 0
    filled_weights = 0
    last_weight = None
    with get_db_session() as db:
        # Only process rows that are missing steps, vo2max, or body_weight_kg
        rows = (
            db.query(HealthMetrics)
            .filter(
                or_(
                    HealthMetrics.steps.is_(None),
                    HealthMetrics.vo2max.is_(None),
                    HealthMetrics.fitness_age.is_(None),
                    HealthMetrics.body_weight_kg.is_(None),
                )
            )
            .order_by(HealthMetrics.date.asc())
            .all()
        )
        total = len(rows)
        logger.info(f"Starting official backfill over {total} rows (missing steps/vo2/weight)")
        for idx, h in enumerate(rows, start=1):
            d = h.date
            if idx % 25 == 0 or idx == 1:
                logger.info(f"Processing {idx}/{total} date={d}")
            try:
                data = collector.collect_health_data(d)
            except Exception as e:
                logger.warning(f"Error collecting data for {d}: {e}")
                continue
            if not data:
                continue
            changed = False
            # Steps
            if isinstance(data.steps, (int, float)):
                new_steps = int(data.steps)
                if h.steps != new_steps:
                    h.steps = new_steps
                    changed = True
                    updated_steps += 1
            # VO2max
            if isinstance(getattr(data, 'vo2max', None), (int, float)):
                new_vo2 = float(data.vo2max)
                if h.vo2max != new_vo2:
                    h.vo2max = new_vo2
                    changed = True
                    updated_vo2 += 1
            # Fitness Age
            if isinstance(getattr(data, 'fitness_age', None), (int, float)):
                new_fa = float(data.fitness_age)
                if getattr(h, 'fitness_age', None) != new_fa:
                    h.fitness_age = new_fa
                    changed = True
                    updated_fitness_age += 1
            # Weight: set from data if present, else forward-fill from last known
            if isinstance(getattr(data, 'body_weight_kg', None), (int, float)):
                if h.body_weight_kg != float(data.body_weight_kg):
                    h.body_weight_kg = float(data.body_weight_kg)
                    changed = True
                last_weight = float(data.body_weight_kg)
            else:
                # If DB already has a weight, carry it forward in memory
                if isinstance(h.body_weight_kg, (int, float)):
                    last_weight = float(h.body_weight_kg)
                # If DB is missing weight and we have a last known value, forward-fill
                elif last_weight is not None:
                    h.body_weight_kg = float(last_weight)
                    changed = True
                    filled_weights += 1
            if changed:
                h.updated_at = datetime.utcnow()
                updated += 1
            # Commit on a time/row cadence to ensure progress even if few updates
            if (idx % 25 == 0) or (updated % 100 == 0 and updated):
                db.commit()
                logger.info(
                    f"Committed batch. Progress {idx}/{total} updated={updated} steps={updated_steps} vo2={updated_vo2} fitness_age={updated_fitness_age} weight_ff={filled_weights}"
                )
        if updated:
            db.commit()
    logger.info(
        f"Official backfill complete. Updated={updated} steps={updated_steps} vo2max={updated_vo2} fitness_age={updated_fitness_age} weight_filled={filled_weights}"
    )
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
