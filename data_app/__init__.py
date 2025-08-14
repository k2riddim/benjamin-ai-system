"""
Benjamin AI System - Data Application

The DATA APP is responsible for:
1. Collecting data from Strava API and Garmin Connect every 2 hours
2. Storing data in PostgreSQL database
3. Monitoring data quality and system health
4. Providing API endpoints for other applications

Components:
- GarminCollector: Collects health data from Garmin Connect
- StravaCollector: Collects activity data from Strava API  
- DataSynchronizer: Orchestrates data collection and storage
- HealthMonitor: Monitors system health and data freshness
- API Server: Provides REST API for data access
"""

__version__ = "1.0.0"
__author__ = "Benjamin AI System"