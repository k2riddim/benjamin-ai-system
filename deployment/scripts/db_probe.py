#!/usr/bin/env python3
from __future__ import annotations

import os
from datetime import date, datetime, timedelta
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

import sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(PROJECT_ROOT)

from shared.database import get_db_session, HealthMetrics, Activities  # type: ignore


app = typer.Typer(help="DB probe utilities (weights, VO2, health).")
console = Console()


@app.command()
def weights(days: int = typer.Option(7, help="Days back to include")) -> None:
    """Show weight stats for today and recent window.

    Prints: today's weight, latest weight, non-null stats in window, and 10 most recent rows.
    """
    today = date.today()
    start = today - timedelta(days=days)
    with get_db_session() as db:
        # Today's weight
        today_row = db.query(HealthMetrics).filter(HealthMetrics.date == today).first()
        today_weight = getattr(today_row, "body_weight_kg", None) if today_row else None

        # Latest weight overall
        latest_row = (
            db.query(HealthMetrics)
            .filter(HealthMetrics.body_weight_kg.isnot(None))
            .order_by(HealthMetrics.date.desc())
            .first()
        )
        latest_weight_date = latest_row.date if latest_row else None
        latest_weight_val = getattr(latest_row, "body_weight_kg", None) if latest_row else None

        # Window stats
        rows = (
            db.query(HealthMetrics)
            .filter(HealthMetrics.date >= start)
            .order_by(HealthMetrics.date.desc())
            .all()
        )
        non_null = [float(r.body_weight_kg) for r in rows if isinstance(r.body_weight_kg, (int, float))]

        console.print(f"Today: {today} weight={today_weight}")
        console.print(f"Latest overall: {latest_weight_date} weight={latest_weight_val}")
        console.print(f"Window [{start} → {today}] count(non-null)={len(non_null)} avg={round(sum(non_null)/len(non_null),2) if non_null else None}")

        t = Table(title="Recent weights", show_lines=False)
        t.add_column("date")
        t.add_column("weight_kg")
        for r in rows[:10]:
            t.add_row(r.date.isoformat(), str(r.body_weight_kg))
        console.print(t)


@app.command()
def vo2(days: int = typer.Option(30, help="Days back to include")) -> None:
    """Show VO2max per-modality stats for a window.
    """
    today = date.today()
    start = today - timedelta(days=days)
    with get_db_session() as db:
        rows = (
            db.query(HealthMetrics)
            .filter(HealthMetrics.date >= start)
            .order_by(HealthMetrics.date.desc())
            .all()
        )
        run_vals = [float(r.vo2max_running) for r in rows if isinstance(r.vo2max_running, (int, float))]
        cyc_vals = [float(r.vo2max_cycling) for r in rows if isinstance(r.vo2max_cycling, (int, float))]
        console.print(f"Window [{start} → {today}] run_n={len(run_vals)} cyc_n={len(cyc_vals)}")
        if run_vals:
            console.print(f"Running VO2: last={run_vals[0]} mean={round(sum(run_vals)/len(run_vals),2)}")
        if cyc_vals:
            console.print(f"Cycling VO2: last={cyc_vals[0]} mean={round(sum(cyc_vals)/len(cyc_vals),2)}")


@app.command()
def health_latest() -> None:
    """Print the latest garmin_health row (selected columns)."""
    with get_db_session() as db:
        r = db.query(HealthMetrics).order_by(HealthMetrics.date.desc()).first()
        if not r:
            console.print("No health rows found")
            raise typer.Exit(1)
        fields = {
            "date": r.date.isoformat(),
            "steps": r.steps,
            "rhr": r.resting_heart_rate,
            "sleep_score": r.sleep_score,
            "body_weight_kg": r.body_weight_kg,
            "vo2max_running": getattr(r, "vo2max_running", None),
            "vo2max_cycling": getattr(r, "vo2max_cycling", None),
            "fitness_age": getattr(r, "fitness_age", None),
        }
        for k, v in fields.items():
            console.print(f"{k}: {v}")


if __name__ == "__main__":
    app()


