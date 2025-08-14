"""
Main entry point for DATA APP
Handles scheduled data synchronization and monitoring
"""

import logging
import logging.handlers
import sys
import time
from datetime import datetime, date
from pathlib import Path
from typing import Dict, Any

import schedule
import typer
from rich.console import Console
from rich.table import Table
from rich.logging import RichHandler

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from data_app.config.settings import settings, LOGS_DIR
from data_app.src.data_synchronizer import DataSynchronizer
from shared.database import create_all_tables, test_connection
from sqlalchemy import text


# Setup logging
def setup_logging():
    """Configure logging for DATA APP"""
    LOGS_DIR.mkdir(exist_ok=True)
    
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper()),
        format=settings.log_format,
        handlers=[
            logging.handlers.RotatingFileHandler(
                LOGS_DIR / settings.log_file,
                maxBytes=settings.log_max_bytes,
                backupCount=settings.log_backup_count
            ),
            RichHandler(rich_tracebacks=True)
        ]
    )
    
    # Get logger for this module
    logger = logging.getLogger(__name__)
    logger.info(f"🚀 {settings.app_name} v{settings.version} starting up")
    return logger


# CLI application
app = typer.Typer(
    name="data-app",
    help="Benjamin AI Data Application - Sync health and activity data",
    rich_markup_mode="rich"
)

console = Console()
@app.command()
def garmin_login():
    """Interactive Garmin login using GARTH_HOME tokens (will prompt for MFA)."""
    logger = setup_logging()
    console.print("🔐 [bold blue]Starting Garmin login[/bold blue]")
    try:
        # Ensure GARTH_HOME is set from settings
        import os
        os.environ["GARTH_HOME"] = settings.garmin_token_dir
        from garminconnect import Garmin
        client = Garmin()
        client.login()
        # Attempt to print name if available
        try:
            name = client.get_full_name()
        except Exception:
            name = "(unknown)"
        console.print(f"✅ [green]Garmin login successful as[/green] {name}")
        console.print(f"📝 Tokens saved under: {os.environ['GARTH_HOME']}")
    except Exception as e:
        console.print(f"❌ [red]Garmin login failed[/red]: {e}")
        raise typer.Exit(1)


@app.command()
def garmin_sync(days_back: int = typer.Option(3, "--days", "-d", help="Days back to sync")):
    """Run Garmin-only sync for the last N days."""
    logger = setup_logging()
    console.print("🏥 [bold blue]Garmin-only synchronization[/bold blue]")
    try:
        synchronizer = DataSynchronizer()
        from datetime import date, timedelta
        end_date = date.today()
        start_date = end_date - timedelta(days=days_back)
        result = synchronizer.sync_garmin_health_data(start_date, end_date)
        status = "Success" if result.get("success") else "Failed"
        console.print(f"✅ Result: {status} | records={result.get('records',0)} created={result.get('created',0)} updated={result.get('updated',0)} errors={len(result.get('errors',[]))}")
    except Exception as e:
        console.print(f"❌ [red]Garmin sync failed[/red]: {e}")
        raise typer.Exit(1)



@app.command()
def sync(
    days_back: int = typer.Option(None, "--days", "-d", help="Days back to sync"),
    force: bool = typer.Option(False, "--force", "-f", help="Force sync even if recent data exists")
):
    """
    Perform data synchronization for Garmin and Strava
    """
    logger = setup_logging()
    
    console.print("🔄 [bold blue]Starting Data Synchronization[/bold blue]")
    
    # Initialize synchronizer
    synchronizer = DataSynchronizer()
    
    # Test connections first
    console.print("🔍 Testing data source connections...")
    connections = synchronizer.test_all_connections()
    
    connection_table = Table(title="Connection Status")
    connection_table.add_column("Service", style="cyan")
    connection_table.add_column("Status", style="green")
    
    for service, status in connections.items():
        status_text = "✅ Connected" if status else "❌ Failed"
        connection_table.add_row(service.title(), status_text)
    
    console.print(connection_table)
    
    # Check if we should proceed
    if not any(connections.values()):
        console.print("❌ [red]No connections available. Exiting.[/red]")
        raise typer.Exit(1)
    
    # Perform sync
    if days_back is None:
        days_back = settings.sync_lookback_days
    
    console.print(f"📅 Syncing data for last [bold]{days_back}[/bold] days")
    
    results = synchronizer.sync_all_data(days_back)
    
    # Display results
    results_table = Table(title="Sync Results")
    results_table.add_column("Service", style="cyan")
    results_table.add_column("Status", style="green")
    results_table.add_column("Records", style="yellow")
    results_table.add_column("Created", style="green")
    results_table.add_column("Updated", style="blue")
    results_table.add_column("Errors", style="red")
    
    for service in ['garmin_health', 'strava_activities']:
        if service in results:
            data = results[service]
            status = "✅ Success" if data['success'] else "❌ Failed"
            results_table.add_row(
                service.replace('_', ' ').title(),
                status,
                str(data.get('records', 0)),
                str(data.get('created', 0)),
                str(data.get('updated', 0)),
                str(len(data.get('errors', [])))
            )
    
    console.print(results_table)
    
    # Show errors if any
    for service in ['garmin_health', 'strava_activities']:
        if service in results and results[service].get('errors'):
            console.print(f"\n❌ [red]{service.replace('_', ' ').title()} Errors:[/red]")
            for error in results[service]['errors']:
                console.print(f"  • {error}")


@app.command()
def status():
    """
    Show current data synchronization status
    """
    logger = setup_logging()
    
    console.print("📊 [bold blue]Data Synchronization Status[/bold blue]")
    
    synchronizer = DataSynchronizer()
    status_data = synchronizer.get_sync_status()
    
    if 'error' in status_data:
        console.print(f"❌ [red]Error getting status: {status_data['error']}[/red]")
        return
    
    # Last sync times
    sync_table = Table(title="Last Synchronization")
    sync_table.add_column("Service", style="cyan")
    sync_table.add_column("Last Sync", style="yellow")
    sync_table.add_column("Status", style="green")
    
    for service in ['garmin', 'strava']:
        last_sync = status_data['last_sync'].get(service)
        last_status = status_data['last_status'].get(service, 'unknown')
        
        if last_sync:
            sync_time = datetime.fromisoformat(last_sync).strftime("%Y-%m-%d %H:%M")
        else:
            sync_time = "Never"
        
        status_emoji = {
            'success': '✅',
            'error': '❌',
            'partial': '⚠️',
            'unknown': '❓'
        }.get(last_status, '❓')
        
        sync_table.add_row(
            service.title(),
            sync_time,
            f"{status_emoji} {last_status.title()}"
        )
    
    console.print(sync_table)
    
    # Data freshness
    freshness_table = Table(title="Data Freshness")
    freshness_table.add_column("Data Type", style="cyan")
    freshness_table.add_column("Latest Date", style="yellow")
    freshness_table.add_column("Age", style="green")
    
    health_date = status_data['data_freshness'].get('latest_health_date')
    activity_date = status_data['data_freshness'].get('latest_activity_date')
    
    if health_date:
        health_age = (date.today() - date.fromisoformat(health_date)).days
        freshness_table.add_row("Health Data", health_date, f"{health_age} days")
    else:
        freshness_table.add_row("Health Data", "No data", "N/A")
    
    if activity_date:
        activity_age = (datetime.now() - datetime.fromisoformat(activity_date)).days
        freshness_table.add_row("Activities", activity_date[:10], f"{activity_age} days")
    else:
        freshness_table.add_row("Activities", "No data", "N/A")
    
    console.print(freshness_table)


@app.command()
def daemon():
    """
    Run as daemon with scheduled synchronization
    """
    logger = setup_logging()
    
    console.print("🔄 [bold blue]Starting DATA APP Daemon[/bold blue]")
    console.print(f"📅 Sync interval: every {settings.sync_interval_hours} hours")
    console.print(f"🔍 Health check: every {settings.health_check_interval_hours} hours")
    
    # Initialize components
    synchronizer = DataSynchronizer()
    
    # Schedule data synchronization
    schedule.every(settings.sync_interval_hours).hours.do(
        lambda: synchronizer.sync_all_data()
    )
    
    
    # Run initial sync and health check
    logger.info("🚀 Running initial data sync")
    synchronizer.sync_all_data()
    
    # Main daemon loop
    console.print("✅ [green]Daemon started. Press Ctrl+C to stop.[/green]")
    
    try:
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
            
    except KeyboardInterrupt:
        logger.info("🛑 Daemon stopped by user")
        console.print("🛑 [yellow]Daemon stopped.[/yellow]")


@app.command()
def reset_and_backfill(
    start: str = typer.Option("2014-01-01", help="Backfill start date YYYY-MM-DD"),
    end: str = typer.Option(None, help="Backfill end date YYYY-MM-DD (default today)"),
    drop: bool = typer.Option(False, help="Drop derived tables before recompute"),
):
    """Recompute agentic-ready derived tables and backfill from base sources.

    Does NOT drop base tables (garmin_health, strava_activities). Optionally clears derived tables.
    """
    logger = setup_logging()
    from shared.database import get_db_session, ReadinessDaily, TrainingLoadDaily, BodyWeightLog
    from datetime import datetime as dt

    start_date = dt.fromisoformat(start).date()
    end_date = dt.fromisoformat(end).date() if end else date.today()

    if drop:
        console.print("🧹 [yellow]Clearing derived tables[/yellow]")
        try:
            with get_db_session() as db:
                db.query(ReadinessDaily).delete()
                db.query(TrainingLoadDaily).delete()
                db.query(BodyWeightLog).delete()
                db.commit()
        except Exception as e:
            console.print(f"❌ [red]Failed to clear derived tables[/red]: {e}")
            raise typer.Exit(1)

    console.print(f"🔄 [bold blue]Computing derivatives for {start_date} → {end_date}[/bold blue]")
    s = DataSynchronizer()
    res = s.compute_agentic_derivatives(start_date, end_date)
    status = "Success" if res.get("success") else "Failed"
    console.print(f"✅ Result: {status} | records={res.get('records',0)} created={res.get('created',0)} updated={res.get('updated',0)} errors={len(res.get('errors',[]))}")


@app.command()
def wipe_and_backfill(
    strava_start: str = typer.Option("2014-01-01", help="Strava backfill start date YYYY-MM-DD"),
    strava_end: str = typer.Option(None, help="Strava backfill end date YYYY-MM-DD (default today)"),
    garmin_days: int = typer.Option(60, help="Days to backfill for Garmin health"),
    yes_i_am_sure: bool = typer.Option(False, help="Confirm destructive wipe of existing data"),
):
    """Delete existing data (base + derived), then backfill and recompute derivatives."""
    if not yes_i_am_sure:
        console.print("❌ [red]Refusing to proceed without --yes-i-am-sure[/red]")
        raise typer.Exit(1)

    logger = setup_logging()
    from shared.database import (
        Base, engine, get_db_session,
        HealthMetrics, Activities, DataSyncLog,
        ReadinessDaily, TrainingLoadDaily, BodyWeightLog, SubjectiveFeedbackDaily,
    )
    from datetime import datetime as dt, timedelta

    # Resolve dates
    start_dt = dt.fromisoformat(strava_start).date()
    end_dt = dt.fromisoformat(strava_end).date() if strava_end else date.today()

    console.print("🧹 [yellow]Wiping existing data (base + derived tables)[/yellow]")
    with get_db_session() as db:
        for model in [
            ReadinessDaily, TrainingLoadDaily, BodyWeightLog, SubjectiveFeedbackDaily,
            DataSyncLog, HealthMetrics, Activities,
        ]:
            db.query(model).delete()
        db.commit()

    console.print("🧱 Ensuring tables exist")
    Base.metadata.create_all(bind=engine)

    s = DataSynchronizer()

    # Garmin backfill first (recent window)
    console.print(f"🏥 [blue]Backfilling Garmin health for last {garmin_days} days[/blue]")
    g_end = date.today()
    g_start = g_end - timedelta(days=garmin_days)
    try:
        garmin_res = s.sync_garmin_health_data(g_start, g_end)
        console.print(f"Garmin: created={garmin_res.get('created',0)} updated={garmin_res.get('updated',0)}")
    except Exception as e:
        console.print(f"⚠️ Garmin backfill error: {e}")

    # Strava backfill full range in chunks
    console.print(f"🏃 [blue]Backfilling Strava activities {start_dt} → {end_dt}[/blue]")
    try:
        strava_res = s.sync_strava_backfill_chunked(start_dt, end_dt, chunk_days=30)
        console.print(f"Strava: created={strava_res.get('created',0)} updated={strava_res.get('updated',0)}")
    except Exception as e:
        console.print(f"⚠️ Strava backfill error: {e}")

    # Compute derivatives across entire window (union of both ranges)
    d_start = min(g_start, start_dt)
    d_end = end_dt
    console.print(f"🧠 [blue]Computing agentic derivatives {d_start} → {d_end}[/blue]")
    deriv = s.compute_agentic_derivatives(d_start, d_end)
    status = "Success" if deriv.get("success") else "Failed"
    console.print(f"✅ Derivatives: {status} | records={deriv.get('records',0)} created={deriv.get('created',0)} updated={deriv.get('updated',0)} errors={len(deriv.get('errors',[]))}")


@app.command(name="reset-all")
def reset_all(
    start_strava: str = typer.Option("2014-01-01", help="Strava backfill start (YYYY-MM-DD)"),
    start_garmin: str = typer.Option(None, help="Garmin backfill start (YYYY-MM-DD), default 120 days ago"),
    end: str = typer.Option(None, help="Backfill end date (YYYY-MM-DD), default today"),
    confirm: bool = typer.Option(False, help="Proceed without prompt and delete ALL data in base and derived tables"),
):
    """DANGER: Wipes base and derived tables and rebuilds from sources, then computes derivatives."""
    logger = setup_logging()
    if not confirm:
        console.print("❌ [red]Refusing to proceed without --confirm[/red]")
        raise typer.Exit(1)

    from shared.database import get_db_session
    from datetime import datetime as dt, timedelta

    end_date = dt.fromisoformat(end).date() if end else date.today()
    try:
        garmin_start = dt.fromisoformat(start_garmin).date() if start_garmin else (end_date - timedelta(days=120))
    except Exception:
        garmin_start = end_date - timedelta(days=120)
    strava_start = dt.fromisoformat(start_strava).date()

    console.print("🗑️ [yellow]Truncating tables[/yellow]")
    with get_db_session() as db:
        db.execute(text("TRUNCATE TABLE readiness_daily, training_load_daily, body_weight_log, subjective_feedback_daily RESTART IDENTITY CASCADE"))
        db.execute(text("TRUNCATE TABLE strava_activities RESTART IDENTITY CASCADE"))
        db.execute(text("TRUNCATE TABLE garmin_health RESTART IDENTITY CASCADE"))
        db.commit()

    console.print(f"🏥 [bold blue]Backfilling Garmin health {garmin_start} → {end_date}[/bold blue]")
    s = DataSynchronizer()
    garmin_res = s.sync_garmin_health_data(garmin_start, end_date)
    console.print(f"Garmin: created={garmin_res.get('created',0)} updated={garmin_res.get('updated',0)}")

    console.print(f"🏃 [bold blue]Backfilling Strava activities {strava_start} → {end_date}[/bold blue]")
    strava_res = s.sync_strava_backfill_chunked(strava_start, end_date, chunk_days=30)
    console.print(f"Strava: created={strava_res.get('created',0)} updated={strava_res.get('updated',0)}")

    console.print(f"🧮 [bold blue]Computing derivatives {strava_start} → {end_date}[/bold blue]")
    deriv = s.compute_agentic_derivatives(strava_start, end_date)
    console.print(f"Derivatives: created={deriv.get('created',0)} updated={deriv.get('updated',0)}")

    console.print("✅ [green]Reset-all completed[/green]")

@app.command()
def test():
    """
    Test data connections and basic functionality
    """
    logger = setup_logging()
    
    console.print("🔍 [bold blue]Testing Benjamin AI Data App[/bold blue]")
    
    # Test database connection
    if not test_connection():
        console.print("❌ [red]Database connection failed[/red]")
        raise typer.Exit(1)
    
    console.print("✅ Database connection successful")
    
    # Test data sources
    synchronizer = DataSynchronizer()
    connections = synchronizer.test_all_connections()
    
    connection_table = Table(title="Connection Test Results")
    connection_table.add_column("Service", style="cyan")
    connection_table.add_column("Status", style="green")
    
    for service, status in connections.items():
        status_text = "✅ Connected" if status else "❌ Failed"
        connection_table.add_row(service.title(), status_text)
    
    console.print(connection_table)
    
    if all(connections.values()):
        console.print("✅ [green]All services are ready for data sync[/green]")
    else:
        console.print("⚠️ [yellow]Some services have connectivity issues[/yellow]")

if __name__ == "__main__":
    app()
    console.print("✅ [green]Daemon started. Press Ctrl+C to stop.[/green]")
    
    try:
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
            
    except KeyboardInterrupt:
        logger.info("🛑 Daemon stopped by user")
        console.print("🛑 [yellow]Daemon stopped.[/yellow]")


@app.command()
def reset_and_backfill(
    start: str = typer.Option("2014-01-01", help="Backfill start date YYYY-MM-DD"),
    end: str = typer.Option(None, help="Backfill end date YYYY-MM-DD (default today)"),
    drop: bool = typer.Option(False, help="Drop derived tables before recompute"),
):
    """Recompute agentic-ready derived tables and backfill from base sources.

    Does NOT drop base tables (garmin_health, strava_activities). Optionally clears derived tables.
    """
    logger = setup_logging()
    from shared.database import get_db_session, ReadinessDaily, TrainingLoadDaily, BodyWeightLog
    from datetime import datetime as dt

    start_date = dt.fromisoformat(start).date()
    end_date = dt.fromisoformat(end).date() if end else date.today()

    if drop:
        console.print("🧹 [yellow]Clearing derived tables[/yellow]")
        try:
            with get_db_session() as db:
                db.query(ReadinessDaily).delete()
                db.query(TrainingLoadDaily).delete()
                db.query(BodyWeightLog).delete()
                db.commit()
        except Exception as e:
            console.print(f"❌ [red]Failed to clear derived tables[/red]: {e}")
            raise typer.Exit(1)

    console.print(f"🔄 [bold blue]Computing derivatives for {start_date} → {end_date}[/bold blue]")
    s = DataSynchronizer()
    res = s.compute_agentic_derivatives(start_date, end_date)
    status = "Success" if res.get("success") else "Failed"
    console.print(f"✅ Result: {status} | records={res.get('records',0)} created={res.get('created',0)} updated={res.get('updated',0)} errors={len(res.get('errors',[]))}")

@app.command()
def test():
    """
    Test data connections and basic functionality
    """
    logger = setup_logging()
    
    console.print("🔍 [bold blue]Testing Benjamin AI Data App[/bold blue]")
    
    # Test database connection
    if not test_connection():
        console.print("❌ [red]Database connection failed[/red]")
        raise typer.Exit(1)
    
    console.print("✅ Database connection successful")
    
    # Test data sources
    synchronizer = DataSynchronizer()
    connections = synchronizer.test_all_connections()
    
    connection_table = Table(title="Connection Test Results")
    connection_table.add_column("Service", style="cyan")
    connection_table.add_column("Status", style="green")
    
    for service, status in connections.items():
        status_text = "✅ Connected" if status else "❌ Failed"
        connection_table.add_row(service.title(), status_text)
    
    console.print(connection_table)
    
    if all(connections.values()):
        console.print("✅ [green]All services are ready for data sync[/green]")
    else:
        console.print("⚠️ [yellow]Some services have connectivity issues[/yellow]")

if __name__ == "__main__":
    app()
    console.print("✅ [green]Daemon started. Press Ctrl+C to stop.[/green]")
    
    try:
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
            
    except KeyboardInterrupt:
        logger.info("🛑 Daemon stopped by user")
        console.print("🛑 [yellow]Daemon stopped.[/yellow]")


@app.command()
def reset_and_backfill(
    start: str = typer.Option("2014-01-01", help="Backfill start date YYYY-MM-DD"),
    end: str = typer.Option(None, help="Backfill end date YYYY-MM-DD (default today)"),
    drop: bool = typer.Option(False, help="Drop derived tables before recompute"),
):
    """Recompute agentic-ready derived tables and backfill from base sources.

    Does NOT drop base tables (garmin_health, strava_activities). Optionally clears derived tables.
    """
    logger = setup_logging()
    from shared.database import get_db_session, ReadinessDaily, TrainingLoadDaily, BodyWeightLog
    from datetime import datetime as dt

    start_date = dt.fromisoformat(start).date()
    end_date = dt.fromisoformat(end).date() if end else date.today()

    if drop:
        console.print("🧹 [yellow]Clearing derived tables[/yellow]")
        try:
            with get_db_session() as db:
                db.query(ReadinessDaily).delete()
                db.query(TrainingLoadDaily).delete()
                db.query(BodyWeightLog).delete()
                db.commit()
        except Exception as e:
            console.print(f"❌ [red]Failed to clear derived tables[/red]: {e}")
            raise typer.Exit(1)

    console.print(f"🔄 [bold blue]Computing derivatives for {start_date} → {end_date}[/bold blue]")
    s = DataSynchronizer()
    res = s.compute_agentic_derivatives(start_date, end_date)
    status = "Success" if res.get("success") else "Failed"
    console.print(f"✅ Result: {status} | records={res.get('records',0)} created={res.get('created',0)} updated={res.get('updated',0)} errors={len(res.get('errors',[]))}")

@app.command()
def test():
    """
    Test data connections and basic functionality
    """
    logger = setup_logging()
    
    console.print("🔍 [bold blue]Testing Benjamin AI Data App[/bold blue]")
    
    # Test database connection
    if not test_connection():
        console.print("❌ [red]Database connection failed[/red]")
        raise typer.Exit(1)
    
    console.print("✅ Database connection successful")
    
    # Test data sources
    synchronizer = DataSynchronizer()
    connections = synchronizer.test_all_connections()
    
    connection_table = Table(title="Connection Test Results")
    connection_table.add_column("Service", style="cyan")
    connection_table.add_column("Status", style="green")
    
    for service, status in connections.items():
        status_text = "✅ Connected" if status else "❌ Failed"
        connection_table.add_row(service.title(), status_text)
    
    console.print(connection_table)
    
    if all(connections.values()):
        console.print("✅ [green]All services are ready for data sync[/green]")
    else:
        console.print("⚠️ [yellow]Some services have connectivity issues[/yellow]")

if __name__ == "__main__":
    app()
    console.print("✅ [green]Daemon started. Press Ctrl+C to stop.[/green]")
    
    try:
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
            
    except KeyboardInterrupt:
        logger.info("🛑 Daemon stopped by user")
        console.print("🛑 [yellow]Daemon stopped.[/yellow]")


@app.command()
def reset_and_backfill(
    start: str = typer.Option("2014-01-01", help="Backfill start date YYYY-MM-DD"),
    end: str = typer.Option(None, help="Backfill end date YYYY-MM-DD (default today)"),
    drop: bool = typer.Option(False, help="Drop derived tables before recompute"),
):
    """Recompute agentic-ready derived tables and backfill from base sources.

    Does NOT drop base tables (garmin_health, strava_activities). Optionally clears derived tables.
    """
    logger = setup_logging()
    from shared.database import get_db_session, ReadinessDaily, TrainingLoadDaily, BodyWeightLog
    from datetime import datetime as dt

    start_date = dt.fromisoformat(start).date()
    end_date = dt.fromisoformat(end).date() if end else date.today()

    if drop:
        console.print("🧹 [yellow]Clearing derived tables[/yellow]")
        try:
            with get_db_session() as db:
                db.query(ReadinessDaily).delete()
                db.query(TrainingLoadDaily).delete()
                db.query(BodyWeightLog).delete()
                db.commit()
        except Exception as e:
            console.print(f"❌ [red]Failed to clear derived tables[/red]: {e}")
            raise typer.Exit(1)

    console.print(f"🔄 [bold blue]Computing derivatives for {start_date} → {end_date}[/bold blue]")
    s = DataSynchronizer()
    res = s.compute_agentic_derivatives(start_date, end_date)
    status = "Success" if res.get("success") else "Failed"
    console.print(f"✅ Result: {status} | records={res.get('records',0)} created={res.get('created',0)} updated={res.get('updated',0)} errors={len(res.get('errors',[]))}")

@app.command()
def test():
    """
    Test data connections and basic functionality
    """
    logger = setup_logging()
    
    console.print("🔍 [bold blue]Testing Benjamin AI Data App[/bold blue]")
    
    # Test database connection
    if not test_connection():
        console.print("❌ [red]Database connection failed[/red]")
        raise typer.Exit(1)
    
    console.print("✅ Database connection successful")
    
    # Test data sources
    synchronizer = DataSynchronizer()
    connections = synchronizer.test_all_connections()
    
    connection_table = Table(title="Connection Test Results")
    connection_table.add_column("Service", style="cyan")
    connection_table.add_column("Status", style="green")
    
    for service, status in connections.items():
        status_text = "✅ Connected" if status else "❌ Failed"
        connection_table.add_row(service.title(), status_text)
    
    console.print(connection_table)
    
    if all(connections.values()):
        console.print("✅ [green]All services are ready for data sync[/green]")
    else:
        console.print("⚠️ [yellow]Some services have connectivity issues[/yellow]")

if __name__ == "__main__":
    app()
    console.print("✅ [green]Daemon started. Press Ctrl+C to stop.[/green]")
    
    try:
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
            
    except KeyboardInterrupt:
        logger.info("🛑 Daemon stopped by user")
        console.print("🛑 [yellow]Daemon stopped.[/yellow]")


@app.command()
def reset_and_backfill(
    start: str = typer.Option("2014-01-01", help="Backfill start date YYYY-MM-DD"),
    end: str = typer.Option(None, help="Backfill end date YYYY-MM-DD (default today)"),
    drop: bool = typer.Option(False, help="Drop derived tables before recompute"),
):
    """Recompute agentic-ready derived tables and backfill from base sources.

    Does NOT drop base tables (garmin_health, strava_activities). Optionally clears derived tables.
    """
    logger = setup_logging()
    from shared.database import get_db_session, ReadinessDaily, TrainingLoadDaily, BodyWeightLog
    from datetime import datetime as dt

    start_date = dt.fromisoformat(start).date()
    end_date = dt.fromisoformat(end).date() if end else date.today()

    if drop:
        console.print("🧹 [yellow]Clearing derived tables[/yellow]")
        try:
            with get_db_session() as db:
                db.query(ReadinessDaily).delete()
                db.query(TrainingLoadDaily).delete()
                db.query(BodyWeightLog).delete()
                db.commit()
        except Exception as e:
            console.print(f"❌ [red]Failed to clear derived tables[/red]: {e}")
            raise typer.Exit(1)

    console.print(f"🔄 [bold blue]Computing derivatives for {start_date} → {end_date}[/bold blue]")
    s = DataSynchronizer()
    res = s.compute_agentic_derivatives(start_date, end_date)
    status = "Success" if res.get("success") else "Failed"
    console.print(f"✅ Result: {status} | records={res.get('records',0)} created={res.get('created',0)} updated={res.get('updated',0)} errors={len(res.get('errors',[]))}")

@app.command()
def test():
    """
    Test data connections and basic functionality
    """
    logger = setup_logging()
    
    console.print("🔍 [bold blue]Testing Benjamin AI Data App[/bold blue]")
    
    # Test database connection
    if not test_connection():
        console.print("❌ [red]Database connection failed[/red]")
        raise typer.Exit(1)
    
    console.print("✅ Database connection successful")
    
    # Test data sources
    synchronizer = DataSynchronizer()
    connections = synchronizer.test_all_connections()
    
    connection_table = Table(title="Connection Test Results")
    connection_table.add_column("Service", style="cyan")
    connection_table.add_column("Status", style="green")
    
    for service, status in connections.items():
        status_text = "✅ Connected" if status else "❌ Failed"
        connection_table.add_row(service.title(), status_text)
    
    console.print(connection_table)
    
    if all(connections.values()):
        console.print("✅ [green]All services are ready for data sync[/green]")
    else:
        console.print("⚠️ [yellow]Some services have connectivity issues[/yellow]")

if __name__ == "__main__":
    app()