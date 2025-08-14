# Benjamin AI System

A comprehensive AI-powered personal training optimization system with three main components:

## 🏗️ Architecture Overview

```
Benjamin AI System
├── 📊 DATA APP - Data Collection & Storage
│   ├── Strava API integration (every 2h)
│   ├── Garmin Connect sync (every 2h) 
│   ├── PostgreSQL storage
│   └── Data monitoring & logging
│
├── 📱 TELEGRAM APP - User Interface
│   ├── Interactive bot interface
│   ├── Daily messages & notifications
│   └── User feedback collection
│
└── 🤖 AGENTIC APP - AI Coaching System
    ├── 5 Specialized AI Agents
    ├── Team discussions & consensus
    ├── Workout generation
    └── Comprehensive logging
```

## 📁 Directory Structure

```
benjamin_ai_system/
├── data_app/           # DATA APP - Data collection & storage
│   ├── src/           # Core data sync modules
│   ├── config/        # Configuration files
│   ├── logs/          # Data sync logs
│   └── tests/         # Data app tests
│
├── telegram_app/      # TELEGRAM APP - User interface
│   ├── src/           # Telegram bot modules
│   ├── config/        # Bot configuration
│   ├── logs/          # Bot interaction logs
│   └── tests/         # Telegram app tests
│
├── agentic_app/       # AGENTIC APP - AI coaching system
│   ├── src/           # Core agentic modules
│   ├── agents/        # Individual AI agents
│   ├── logs/          # Agent discussion logs
│   └── tests/         # Agentic app tests
│
├── shared/            # Shared components
│   ├── database/      # Database models & connections
│   ├── models/        # Pydantic models
│   └── utils/         # Common utilities
│
├── deployment/        # Deployment configurations
│   ├── systemd/       # Service files
│   └── scripts/       # Deployment scripts
│
└── docs/             # Documentation
```

## 🎯 Core Components

### 1. DATA APP
- **Purpose**: Collect and store health/activity data
- **Schedule**: Every 2 hours data sync
- **Sources**: Strava API + Garmin Connect (via garth)
- **Storage**: PostgreSQL database
- **Monitoring**: Daily health checks with comprehensive logging

### 2. TELEGRAM APP  
- **Purpose**: User interface and daily interactions
- **Features**: Daily messages, feedback collection, interactive menus
- **Integration**: Reads from PostgreSQL via API

### 3. AGENTIC APP
- **Purpose**: AI-powered coaching system
- **Agents**: Project Manager, Running Coach, Strength Coach, Nutritionist, Sports Psychologist
- **Output**: Daily detailed workouts and coaching advice
- **Logging**: Complete agent discussion transcripts

## 🚀 Getting Started

1. **Setup Environment**
   ```bash
   python -m venv benjamin_ai_env
   source benjamin_ai_env/bin/activate
   pip install -r requirements.txt
   ```

2. **Configure Services**
   - Copy config templates
   - Set API keys and tokens
   - Configure database connection

3. **Deploy Components**
   ```bash
   ./deployment/scripts/deploy_data_app.sh
   ./deployment/scripts/deploy_telegram_app.sh
   ./deployment/scripts/deploy_agentic_app.sh
   ```

## 📊 Data Flow

```
Strava API ──┐
              ├─► DATA APP ──► PostgreSQL ──► API ──► AGENTIC APP ──► TELEGRAM APP ──► User
Garmin API ──┘
```

## 🔧 Configuration

- Secrets are loaded via per-service environment files referenced by systemd:
  - data_app: `data_app/config/.env` (example: `data_app/config/env_example`)
  - agentic_app: `agentic_app/config/.env` (contains `AGENTIC_APP_*`, `LANGSMITH_API_KEY`)
  - telegram_app: `telegram_app/config/.env` (contains `TELEGRAM_APP_*`)
- Do not commit real secrets. Rotate any credentials previously checked into the repo.
 - Ensure `.env` files are not committed (`.gitignore` added). Use the `env_example` files as templates:
   - `data_app/config/env_example`
   - `agentic_app/config/env_example`
   - `telegram_app/config/env_example`
- Logging: Comprehensive logs in each component's `logs/` directory
- Scheduling: systemd services with timers

## 📈 Monitoring

- Daily data sync health checks
- API endpoint monitoring  
- Agent discussion logging
- User interaction tracking