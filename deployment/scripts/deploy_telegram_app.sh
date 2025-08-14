#!/bin/bash

# Deploy Benjamin AI Telegram App
# This script deploys the Telegram bot service

set -e  # Exit on any error

# Configuration
PROJECT_DIR="/home/chorizo/projects/benjamin_ai_system"
VENV_PATH="/home/chorizo/bbox_env"
SYSTEMD_DIR="/etc/systemd/system"
SERVICE_USER="chorizo"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running as root for service installation
check_permissions() {
    if [[ $EUID -ne 0 ]]; then
        log_error "This script must be run as root for service installation"
        exit 1
    fi
}

# Create log directories
setup_directories() {
    log_info "Setting up directories..."
    
    mkdir -p $PROJECT_DIR/telegram_app/logs
    chown -R $SERVICE_USER:$SERVICE_USER $PROJECT_DIR/telegram_app/logs
    
    log_success "Directories set up"
}

# Install Python dependencies
install_dependencies() {
    log_info "Installing Python dependencies..."
    
    cd $PROJECT_DIR
    $VENV_PATH/bin/pip install python-telegram-bot schedule
    
    log_success "Dependencies installed"
}

# Test Telegram bot
test_telegram_bot() {
    log_info "Testing Telegram bot..."
    
    cd $PROJECT_DIR
    
    # Load environment from service env file if present
    if [[ -f "$PROJECT_DIR/telegram_app/config/.env" ]]; then
        set -a
        source "$PROJECT_DIR/telegram_app/config/.env"
        set +a
    fi

    # Test bot creation (requires TELEGRAM_APP_TELEGRAM_BOT_TOKEN and TELEGRAM_APP_TELEGRAM_CHAT_ID set)
    $VENV_PATH/bin/python -c "
import sys
from pathlib import Path
sys.path.append(str(Path('.').resolve()))

from telegram_app.config.settings import settings
from telegram_app.src.telegram_bot import BenjaminTelegramBot

bot = BenjaminTelegramBot()
app = bot.setup_application()
print('✅ Telegram bot test passed')
" || {
        log_error "Telegram bot test failed"
        exit 1
    }
    
    log_success "Telegram bot tests passed"
}

# Install systemd services
install_services() {
    log_info "Installing systemd services..."
    
    # Copy service files
    cp $PROJECT_DIR/deployment/systemd/benjamin-telegram-app.service $SYSTEMD_DIR/
    
    # Set permissions
    chmod 644 $SYSTEMD_DIR/benjamin-telegram-app.service
    
    # Reload systemd
    systemctl daemon-reload
    
    log_success "Services installed"
}

# Start and enable services
start_services() {
    log_info "Starting and enabling services..."
    
    # Enable services
    systemctl enable benjamin-telegram-app.service
    
    # Start services
    systemctl start benjamin-telegram-app.service
    
    log_success "Services started and enabled"
}

# Check service status
check_service_status() {
    log_info "Checking service status..."
    
    echo "=== Benjamin Telegram App Service ==="
    systemctl status benjamin-telegram-app.service --no-pager -l
    
    
    # Quick status check
    if systemctl is-active --quiet benjamin-telegram-app.service; then
        log_success "Telegram App service is running"
    else
        log_warning "Telegram App service is not running"
    fi
}

# Send deployment notification
send_notification() {
    log_info "Sending deployment notification..."
    
    cd $PROJECT_DIR
    
    # Load environment from service env file if present
    if [[ -f "$PROJECT_DIR/telegram_app/config/.env" ]]; then
        set -a
        source "$PROJECT_DIR/telegram_app/config/.env"
        set +a
    fi

    $VENV_PATH/bin/python -c "
import asyncio
import sys
from pathlib import Path
sys.path.append(str(Path('.').resolve()))

async def notify():
    from telegram_app.config.settings import settings
    from telegram_app.src.telegram_bot import BenjaminTelegramBot
    
    bot = BenjaminTelegramBot()
    app = bot.setup_application()
    
    message = '''🚀 **Benjamin AI Telegram App Deployed!**

✅ Telegram Bot Service: Running
⏱️ Daily message now triggered by systemd timer (agentic discussion)  
🤖 Interactive coaching interface ready
📅 Daily messages at 07:00
💪 Your AI coach is now online!

Try: /help to see all commands'''
    
    await app.bot.send_message(
        chat_id=settings.telegram_chat_id,
        text=message
    )
    print('✅ Notification sent')

asyncio.run(notify())
" || log_warning "Failed to send notification"
    
    log_success "Deployment notification sent"
}

# Main deployment process
main() {
    log_info "🚀 Starting Benjamin AI Telegram App Deployment"
    
    check_permissions
    setup_directories
    install_dependencies
    test_telegram_bot
    install_services
    start_services
    check_service_status
    send_notification
    
    log_success "🎉 Benjamin AI Telegram App deployment completed successfully!"
    echo
    echo "📱 Telegram bot is now running and ready for Benjamin!"
    echo "📅 Daily messages will be sent at 07:00"
    echo "🔧 Use 'sudo systemctl status benjamin-telegram-app' to check status"
    echo "📋 Use 'journalctl -u benjamin-telegram-app -f' to view logs"
}

# Run main function
main "$@"
