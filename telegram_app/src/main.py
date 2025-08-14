"""
Main entry point for TELEGRAM APP
Handles bot startup, scheduling, and daily message delivery
"""

import asyncio
import logging
import logging.handlers
import sys
import time
from datetime import datetime, time as dt_time
from pathlib import Path
from typing import Dict, Any

import schedule
import typer
from rich.console import Console
from rich.table import Table
from rich.logging import RichHandler

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from telegram_app.config.settings import settings, LOGS_DIR
from telegram_app.src.telegram_bot import BenjaminTelegramBot


# Setup logging
def setup_logging():
    """Configure logging for TELEGRAM APP"""
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
    # Reduce noisy HTTP logs that can leak sensitive tokens in URLs
    try:
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("urllib3").setLevel(logging.WARNING)
    except Exception:
        pass
    
    # Get logger for this module
    logger = logging.getLogger(__name__)
    logger.info(f"🚀 {settings.app_name} v{settings.version} starting up")
    return logger


# CLI application
app = typer.Typer(
    name="telegram-app",
    help="Benjamin AI Telegram Application - Interactive coaching interface",
    rich_markup_mode="rich"
)

console = Console()


@app.command()
def test():
    """
    Test Telegram bot configuration and connectivity
    """
    logger = setup_logging()
    
    console.print("🔍 [bold blue]Testing Benjamin AI Telegram Bot[/bold blue]")
    
    # Check configuration
    if not settings.telegram_bot_token:
        console.print("❌ [red]TELEGRAM_BOT_TOKEN not configured[/red]")
        console.print("Please set TELEGRAM_BOT_TOKEN environment variable")
        raise typer.Exit(1)
    
    console.print("✅ Bot token configured")
    
    # Test bot creation
    try:
        bot = BenjaminTelegramBot()
        application = bot.setup_application()
        console.print("✅ Bot application created successfully")
    except Exception as e:
        console.print(f"❌ [red]Failed to create bot: {e}[/red]")
        raise typer.Exit(1)
    
    # Test Data API connectivity
    import requests
    try:
        response = requests.get(f"{settings.data_api_base_url}/health", timeout=10)
        if response.status_code == 200:
            console.print("✅ Data API connection successful")
        else:
            console.print(f"⚠️ [yellow]Data API returned status {response.status_code}[/yellow]")
    except Exception as e:
        console.print(f"❌ [red]Data API connection failed: {e}[/red]")
    
    console.print("✅ [green]Telegram bot test completed[/green]")


@app.command()
def send_test_message(
    message: str = typer.Option("🤖 Test message from Benjamin AI!", help="Message to send")
):
    """
    Send a test message to verify bot functionality
    """
    logger = setup_logging()
    
    if not settings.telegram_chat_id:
        console.print("❌ [red]TELEGRAM_CHAT_ID not configured[/red]")
        console.print("Please set TELEGRAM_CHAT_ID environment variable")
        raise typer.Exit(1)
    
    console.print(f"📤 Sending test message: [italic]{message}[/italic]")
    
    async def send_message():
        try:
            bot = BenjaminTelegramBot()
            application = bot.setup_application()
            
            await application.bot.send_message(
                chat_id=settings.telegram_chat_id,
                text=message
            )
            
            console.print("✅ [green]Test message sent successfully![/green]")
            
        except Exception as e:
            console.print(f"❌ [red]Failed to send message: {e}[/red]")
            raise typer.Exit(1)
    
    asyncio.run(send_message())


@app.command()
def daily_message():
    """
    Send daily coaching message (simplified for now)
    """
    logger = setup_logging()
    
    console.print("📅 [bold blue]Generating Daily Coaching Message[/bold blue]")
    
    async def send_daily():
        try:
            bot = BenjaminTelegramBot()
            application = bot.setup_application()
            # Fetch AI-generated workout of the day from Agentic API
            import requests
            try:
                resp = requests.post(f"{settings.agentic_api_base_url}/daily-discussion", json={}, timeout=1800)
                if resp.status_code == 200:
                    payload = resp.json()
                    message = payload.get("telegram_message") or "Good morning!"
                else:
                    message = "🌅 Good morning! I couldn't reach the AI team. Use /discussion to try again."
            except Exception:
                message = "🌅 Good morning! I couldn't reach the AI team. Use /discussion to try again."

            await application.bot.send_message(
                chat_id=settings.telegram_chat_id,
                text=message
            )

            console.print("✅ [green]Daily message sent successfully![/green]")
        except Exception as e:
            console.print(f"❌ [red]Failed to send daily message: {e}[/red]")
            logger.error(f"Daily message failed: {e}")
            raise typer.Exit(1)
    
    asyncio.run(send_daily())


@app.command()
def polling():
    """
    Run bot in polling mode (for development/testing)
    """
    logger = setup_logging()
    
    console.print("🔄 [bold blue]Starting Telegram Bot in Polling Mode[/bold blue]")
    
    try:
        bot = BenjaminTelegramBot()
        application = bot.setup_application()
        
        console.print("🚀 Bot is running in polling mode")
        console.print("✅ [green]Press Ctrl+C to stop[/green]")
        
        # Run polling - this blocks until interrupted
        application.run_polling(
            allowed_updates=["message", "callback_query"],
            drop_pending_updates=True
        )
        
    except KeyboardInterrupt:
        logger.info("🛑 Polling stopped by user")
        console.print("🛑 [yellow]Bot stopped[/yellow]")
    except Exception as e:
        console.print(f"❌ [red]Polling failed: {e}[/red]")
        logger.error(f"Polling error: {e}")
        raise typer.Exit(1)


@app.command()
def daemon():
    """
    Run as daemon with scheduled daily messages
    """
    logger = setup_logging()
    
    console.print("🔄 [bold blue]Starting TELEGRAM APP Daemon[/bold blue]")
    console.print(f"📅 Daily messages: {settings.daily_message_time}")
    console.print(f"🔍 Check interval: every {settings.check_interval_minutes} minutes")
    
    # Schedule daily messages
    schedule.every().day.at(settings.daily_message_time).do(
        lambda: asyncio.run(send_daily_message_scheduled())
    )
    
    console.print("✅ [green]Daemon started. Press Ctrl+C to stop.[/green]")
    
    try:
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
            
    except KeyboardInterrupt:
        logger.info("🛑 Daemon stopped by user")
        console.print("🛑 [yellow]Daemon stopped[/yellow]")


async def send_daily_message_scheduled():
    """Send scheduled daily message"""
    try:
        bot = BenjaminTelegramBot()
        application = bot.setup_application()
        # Ask Agentic API for the daily discussion recommendation
        import requests
        try:
            resp = requests.post(f"{settings.agentic_api_base_url}/daily-discussion", json={}, timeout=1800)
            if resp.status_code == 200:
                payload = resp.json()
                message = payload.get("telegram_message") or "Good morning!"
            else:
                message = "🌅 Good morning! I couldn't reach the AI team. Use /discussion to try again."
        except Exception:
            message = "🌅 Good morning! I couldn't reach the AI team. Use /discussion to try again."

        await application.bot.send_message(
            chat_id=settings.telegram_chat_id,
            text=message
        )
        
        print("✅ Scheduled daily message sent")
    except Exception as e:
        print(f"❌ Failed to send scheduled message: {e}")


@app.command()
def status():
    """
    Show Telegram app status and configuration
    """
    logger = setup_logging()
    
    console.print("📊 [bold blue]Telegram App Status[/bold blue]")
    
    # Configuration status
    config_table = Table(title="Configuration")
    config_table.add_column("Setting", style="cyan")
    config_table.add_column("Status", style="green")
    config_table.add_column("Value", style="yellow")
    
    config_checks = [
        ("Bot Token", "✅ Set" if settings.telegram_bot_token else "❌ Missing", 
         "***" + settings.telegram_bot_token[-4:] if settings.telegram_bot_token else "Not set"),
        ("Chat ID", "✅ Set" if settings.telegram_chat_id else "❌ Missing", 
         settings.telegram_chat_id or "Not set"),
        ("Data API", "📡 Configured", settings.data_api_base_url),
        ("Daily Time", "⏰ Configured", settings.daily_message_time),
    ]
    
    for setting, status, value in config_checks:
        config_table.add_row(setting, status, str(value))
    
    console.print(config_table)
    
    # Test connections
    console.print("\n🔍 Testing Connections...")
    
    # Test Data API
    try:
        import requests
        response = requests.get(f"{settings.data_api_base_url}/health", timeout=5)
        if response.status_code == 200:
            console.print("✅ Data API: Connected")
        else:
            console.print(f"⚠️ Data API: Status {response.status_code}")
    except Exception as e:
        console.print(f"❌ Data API: {str(e)[:50]}...")
    
    # Test Bot Token (if available)
    if settings.telegram_bot_token:
        try:
            bot = BenjaminTelegramBot()
            application = bot.setup_application()
            console.print("✅ Bot: Token valid and application created")
        except Exception as e:
            console.print(f"❌ Bot: {str(e)[:50]}...")
    
    console.print(f"\n🚀 Telegram App v{settings.version} ready!")


if __name__ == "__main__":
    app()
