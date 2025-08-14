#!/bin/bash

# Benjamin AI Data App Deployment Script

set -e

echo "🚀 Deploying Benjamin AI Data App..."

# Configuration
PROJECT_DIR="/home/chorizo/projects/benjamin_ai_system"
VENV_PATH="/home/chorizo/bbox_env"
SYSTEMD_DIR="/etc/systemd/system"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

log_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

log_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

log_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Check if running as root for systemd operations
check_sudo() {
    if [ "$EUID" -ne 0 ]; then
        log_error "This script needs sudo privileges for systemd operations"
        echo "Please run: sudo $0"
        exit 1
    fi
}

# Install Python dependencies
install_dependencies() {
    log_info "Installing Python dependencies..."
    
    if [ ! -d "$VENV_PATH" ]; then
        log_error "Virtual environment not found at $VENV_PATH"
        exit 1
    fi
    
    $VENV_PATH/bin/pip install -r $PROJECT_DIR/requirements.txt
    log_success "Dependencies installed"
}

# Test system readiness  
test_system() {
    log_info "Testing system readiness..."
    
    cd $PROJECT_DIR
    $VENV_PATH/bin/python -m data_app.src.main test
    
    log_success "System tests completed"
}

# Install systemd services
install_services() {
    log_info "Installing systemd services..."
    
    # Copy service files
    cp $PROJECT_DIR/deployment/systemd/benjamin-data-app.service $SYSTEMD_DIR/
    cp $PROJECT_DIR/deployment/systemd/benjamin-data-api.service $SYSTEMD_DIR/
    
    # Set correct permissions
    chmod 644 $SYSTEMD_DIR/benjamin-data-app.service
    chmod 644 $SYSTEMD_DIR/benjamin-data-api.service
    
    # Reload systemd
    systemctl daemon-reload
    
    log_success "Systemd services installed"
}

# Start services
start_services() {
    log_info "Starting services..."
    
    # Enable services
    systemctl enable benjamin-data-app.service
    systemctl enable benjamin-data-api.service
    
    # Start services
    systemctl start benjamin-data-app.service
    systemctl start benjamin-data-api.service
    
    log_success "Services started"
}

# Check service status
check_status() {
    log_info "Checking service status..."
    
    echo -e "\n${BLUE}Data App Service:${NC}"
    systemctl status benjamin-data-app.service --no-pager -l
    
    echo -e "\n${BLUE}Data API Service:${NC}"
    systemctl status benjamin-data-api.service --no-pager -l
}

# Test API connectivity
test_api() {
    log_info "Testing API connectivity..."
    
    sleep 5  # Give services time to start
    
    # Test API health endpoint
    if curl -f -s http://127.0.0.1:8010/health > /dev/null; then
        log_success "Data API is responding"
    else
        log_warning "Data API may not be responding yet"
    fi
}

# Create log directories
setup_logs() {
    log_info "Setting up log directories..."
    
    mkdir -p $PROJECT_DIR/data_app/logs
    chown chorizo:chorizo $PROJECT_DIR/data_app/logs
    chmod 755 $PROJECT_DIR/data_app/logs
    
    log_success "Log directories created"
}

# Main deployment flow
main() {
    echo -e "${BLUE}"
    echo "=================================================="
    echo "    Benjamin AI Data App Deployment"
    echo "=================================================="
    echo -e "${NC}"
    
    check_sudo
    setup_logs
    install_dependencies
    test_system
    install_services
    start_services
    check_status
    test_api
    
    echo -e "\n${GREEN}🎉 Deployment completed successfully!${NC}"
    echo -e "\n${BLUE}Services:${NC}"
    echo "  • Data App: systemctl status benjamin-data-app.service"
    echo "  • Data API: systemctl status benjamin-data-api.service"
    echo -e "\n${BLUE}API Endpoints:${NC}"
    echo "  • Health: http://127.0.0.1:8010/health"
    echo "  • Docs: http://127.0.0.1:8010/docs"
    echo "  • Recent Activities: http://127.0.0.1:8010/activities/recent"
    echo -e "\n${BLUE}Logs:${NC}"
    echo "  • journalctl -u benjamin-data-app.service -f"
    echo "  • journalctl -u benjamin-data-api.service -f"
}

# Run main function
main "$@"