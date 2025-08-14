"""
Clean Strava API data collector for Benjamin AI System
"""

import logging
import time
from datetime import date, datetime, timedelta
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

import requests

from ..config.settings import settings


@dataclass
class StravaActivity:
    id: str
    name: str
    activity_type: str
    sport_type: str
    start_date: datetime
    start_date_local: datetime
    timezone: str
    duration: float  # minutes
    moving_time: float  # minutes
    distance: float  # km
    average_pace: Optional[float] = None  # min/km
    average_speed: Optional[float] = None  # km/h
    max_speed: Optional[float] = None
    elevation_gain: Optional[float] = None
    elev_high: Optional[float] = None
    elev_low: Optional[float] = None
    average_heart_rate: Optional[float] = None
    max_heart_rate: Optional[float] = None
    has_heartrate: bool = False
    average_watts: Optional[float] = None
    max_watts: Optional[float] = None
    weighted_average_watts: Optional[float] = None
    kilojoules: Optional[float] = None
    calories: Optional[float] = None
    suffer_score: Optional[float] = None
    device_name: Optional[str] = None
    trainer: bool = False
    commute: bool = False
    manual: bool = False
    raw_data: Dict[str, Any] = None


class StravaCollector:
    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)
        self.access_token = settings.strava_access_token
        self.refresh_token = settings.strava_refresh_token
        self.client_id = settings.strava_client_id
        self.client_secret = settings.strava_client_secret
        self.expires_at = settings.strava_token_expires_at
        self._minute_start = time.time()
        self._requests = 0

    def _rate_limit(self) -> None:
        now = time.time()
        if now - self._minute_start >= 60:
            self._minute_start = now
            self._requests = 0
        if self._requests >= settings.strava_requests_per_minute:
            wait_s = 60 - (now - self._minute_start)
            if wait_s > 0:
                self.logger.info(f"⏱️ Strava RL sleep {wait_s:.1f}s")
                time.sleep(wait_s)
            self._minute_start = time.time()
            self._requests = 0
        self._requests += 1

    def _make_request(self, url: str, params: Dict[str, Any] = None) -> Optional[Dict]:
        self._rate_limit()
        headers = {"Authorization": f"Bearer {self.access_token}", "Content-Type": "application/json"}
        try:
            r = requests.get(url, headers=headers, params=params, timeout=30)
            if r.status_code == 401:
                self.logger.warning("🔄 Strava token expired; refreshing")
                if self.refresh_access_token():
                    headers["Authorization"] = f"Bearer {self.access_token}"
                    r = requests.get(url, headers=headers, params=params, timeout=30)
                else:
                    return None
            if r.status_code == 429:
                wait_s = int(r.headers.get('Retry-After', 60))
                time.sleep(wait_s)
                return self._make_request(url, params)
            if r.status_code != 200:
                self.logger.error(f"❌ Strava {r.status_code}: {r.text}")
                return None
            return r.json()
        except requests.exceptions.RequestException as e:
            self.logger.error(f"❌ Strava request failed: {e}")
            return None

    def refresh_access_token(self) -> bool:
        try:
            url = "https://www.strava.com/oauth/token"
            data = {
                'client_id': self.client_id,
                'client_secret': self.client_secret,
                'refresh_token': self.refresh_token,
                'grant_type': 'refresh_token'
            }
            r = requests.post(url, data=data, timeout=30)
            if r.status_code != 200:
                self.logger.error(f"❌ Token refresh failed: {r.status_code} {r.text}")
                return False
            td = r.json()
            self.access_token = td['access_token']
            self.refresh_token = td['refresh_token']
            self.expires_at = td['expires_at']
            self._persist_tokens_to_env()
            self.logger.info("✅ Strava access token refreshed")
            return True
        except Exception as e:
            self.logger.error(f"❌ Error refreshing token: {e}")
            return False

    def _persist_tokens_to_env(self) -> None:
        try:
            from pathlib import Path
            env_path = Path(__file__).resolve().parents[1] / 'config' / '.env'
            existing = {}
            if env_path.exists():
                for line in env_path.read_text().splitlines():
                    if '=' in line and not line.strip().startswith('#'):
                        k, v = line.split('=', 1)
                        existing[k.strip()] = v.strip()
            existing['DATA_APP_STRAVA_ACCESS_TOKEN'] = self.access_token
            existing['DATA_APP_STRAVA_REFRESH_TOKEN'] = self.refresh_token
            existing['DATA_APP_STRAVA_TOKEN_EXPIRES_AT'] = str(self.expires_at)
            env_path.write_text("\n".join([f"{k}={v}" for k, v in existing.items()]) + "\n")
        except Exception:
            pass

    def get_athlete_info(self) -> Optional[Dict[str, Any]]:
        return self._make_request(f"{settings.strava_api_base_url}/athlete")

    def test_connection(self) -> bool:
        try:
            info = self.get_athlete_info()
            if info:
                self.logger.info(f"✅ Connected to Strava as {info.get('firstname','')} {info.get('lastname','')}")
                return True
            return False
        except Exception:
            return False

    def collect_activities(self, start_date: date, end_date: date, per_page: int = 100) -> List[StravaActivity]:
        results: List[StravaActivity] = []
        after = int(datetime.combine(start_date, datetime.min.time()).timestamp())
        before = int(datetime.combine(end_date, datetime.max.time()).timestamp())
        page = 1
        while True:
            params = {"after": after, "before": before, "page": page, "per_page": per_page}
            data = self._make_request(f"{settings.strava_api_base_url}/athlete/activities", params)
            if not data:
                break
            if isinstance(data, list) and not data:
                break
            for item in data or []:
                act = self._parse_activity(item)
                if act:
                    results.append(act)
            if len(data or []) < per_page:
                break
            page += 1
        return results

    def _parse_activity(self, data: Dict[str, Any]) -> Optional[StravaActivity]:
        try:
            sd = data.get('start_date')
            sdl = data.get('start_date_local')
            start_date = datetime.fromisoformat(sd) if sd else None
            start_date_local = datetime.fromisoformat(sdl) if sdl else start_date
            duration_minutes = (data.get('elapsed_time') or 0) / 60.0
            moving_time_minutes = (data.get('moving_time') or 0) / 60.0
            distance_km = (data.get('distance') or 0) / 1000.0
            average_pace = None
            if moving_time_minutes > 0 and distance_km > 0 and data.get('type') in ['Run', 'TrailRun']:
                average_pace = moving_time_minutes / distance_km
            average_speed = (data.get('average_speed') or 0) * 3.6 if data.get('average_speed') else None
            max_speed = (data.get('max_speed') or 0) * 3.6 if data.get('max_speed') else None
            return StravaActivity(
                id=str(data['id']),
                name=data.get('name', ''),
                activity_type=data.get('type', ''),
                sport_type=data.get('sport_type', data.get('type', '')),
                start_date=start_date,
                start_date_local=start_date_local,
                timezone=data.get('timezone', ''),
                duration=duration_minutes,
                moving_time=moving_time_minutes,
                distance=distance_km,
                average_pace=average_pace,
                average_speed=average_speed,
                max_speed=max_speed,
                elevation_gain=data.get('total_elevation_gain'),
                elev_high=data.get('elev_high'),
                elev_low=data.get('elev_low'),
                average_heart_rate=data.get('average_heartrate'),
                max_heart_rate=data.get('max_heartrate'),
                has_heartrate=data.get('has_heartrate', False),
                average_watts=data.get('average_watts'),
                max_watts=data.get('max_watts'),
                weighted_average_watts=data.get('weighted_average_watts'),
                kilojoules=data.get('kilojoules'),
                calories=data.get('calories'),
                suffer_score=data.get('suffer_score'),
                device_name=data.get('device_name'),
                trainer=data.get('trainer', False),
                commute=data.get('commute', False),
                manual=data.get('manual', False),
                raw_data=data,
            )
        except Exception as e:
            self.logger.error(f"❌ Error parsing activity: {e}")
            return None

    def get_activity_details(self, activity_id: str) -> Optional[Dict[str, Any]]:
        return self._make_request(f"{settings.strava_api_base_url}/activities/{activity_id}")
    
    def collect_recent_activities(self, days_back: int = 7) -> List[StravaActivity]:
        """Collect activities from the last N days"""
        end_date = date.today()
        start_date = end_date - timedelta(days=days_back)
        
        return self.collect_activities(start_date, end_date)
    
    def test_connection(self) -> bool:
        """Test Strava API connectivity"""
        try:
            athlete_info = self.get_athlete_info()
            if athlete_info:
                self.logger.info(f"✅ Connected to Strava as {athlete_info.get('firstname', 'Unknown')} {athlete_info.get('lastname', '')}")
                return True
            return False
        except Exception as e:
            self.logger.error(f"❌ Strava connection test failed: {e}")
            return False
    
    def get_activity_details(self, activity_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed information for a specific activity"""
        url = f"{settings.strava_api_base_url}/activities/{activity_id}"
        return self._make_request(url)