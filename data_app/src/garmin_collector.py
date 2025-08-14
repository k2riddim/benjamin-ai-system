"""
Garmin health data collector for Benjamin AI System
Directly uses garminconnect (with garth tokens via GARTH_HOME) to fetch data
"""

import logging
import os
import json
import traceback
from datetime import date, datetime, timedelta
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from pathlib import Path

from tenacity import retry, stop_after_attempt, wait_exponential
from garminconnect import Garmin, GarminConnectConnectionError, GarminConnectAuthenticationError

from ..config.settings import settings


@dataclass
class GarminHealthData:
    """Structured Garmin health data"""
    date: date
    steps: Optional[int] = None
    resting_heart_rate: Optional[int] = None
    sleep_score: Optional[float] = None
    vo2max: Optional[float] = None
    fitness_age: Optional[float] = None
    hrv_score: Optional[float] = None
    hrv_weekly_avg: Optional[float] = None
    hrv_status: Optional[str] = None
    stress_score: Optional[float] = None
    max_stress: Optional[float] = None
    body_battery: Optional[float] = None
    training_readiness: Optional[float] = None
    training_status: Optional[str] = None
    sleep_hours: Optional[float] = None
    deep_sleep_minutes: Optional[int] = None
    rem_sleep_minutes: Optional[int] = None
    body_weight_kg: Optional[float] = None
    raw_stats: Dict[str, Any] = None
    raw_hr: Dict[str, Any] = None
    raw_sleep: Dict[str, Any] = None
    raw_hrv: Dict[str, Any] = None
    raw_training_readiness: Dict[str, Any] = None
    raw_training_status: Dict[str, Any] = None
    raw_body_battery: Dict[str, Any] = None
    raw_stress: Dict[str, Any] = None
    raw_respiration: Dict[str, Any] = None
    raw_spo2: Dict[str, Any] = None
    raw_body_composition: Dict[str, Any] = None


class GarminCollector:
    """Collects health data directly from Garmin Connect using token-based auth"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        # Ensure GARTH_HOME is set so garth finds tokens
        if settings.garmin_token_dir:
            os.environ["GARTH_HOME"] = settings.garmin_token_dir
        self._client: Optional[Garmin] = None
        self._authenticated: bool = False
        # Log environment state at startup
        self._log_env_state()

    def _log_env_state(self) -> None:
        garth_home = os.environ.get("GARTH_HOME", "not set")
        self.logger.info(f"🧭 GARTH_HOME={garth_home}")
        try:
            if garth_home != "not set":
                token_dir = Path(garth_home)
                if not token_dir.exists():
                    self.logger.warning(f"⚠️ GARTH_HOME directory does not exist: {token_dir}")
                    return
                for name in ["oauth1_token.json", "oauth2_token.json"]:
                    p = token_dir / name
                    if p.exists():
                        stat = p.stat()
                        self.logger.info(
                            f"🗂️ {name}: size={stat.st_size} mtime={datetime.fromtimestamp(stat.st_mtime)}"
                        )
                        try:
                            data = json.loads(p.read_text())
                            if isinstance(data, dict):
                                self.logger.info(f"🔑 {name} keys: {list(data.keys())}")
                        except Exception as e:
                            self.logger.debug(f"Could not read {name}: {e}")
                    else:
                        self.logger.warning(f"⚠️ Missing token file: {name}")
        except Exception as e:
            self.logger.debug(f"Env state logging failed: {e}")

    def _login_if_needed(self) -> bool:
        if self._authenticated and self._client is not None:
            return True
        # Prefer token-based initialization (no fresh login) if tokens exist
        try:
            garth_home = os.environ.get('GARTH_HOME', '')
            oauth1 = Path(garth_home) / 'oauth1_token.json'
            oauth2 = Path(garth_home) / 'oauth2_token.json'
            if garth_home and oauth1.exists() and oauth2.exists():
                import garth as garth_pkg
                gclient = Garmin()
                http_client = garth_pkg.Client()
                http_client.load(garth_home)
                gclient.garth = http_client
                # Ensure display/profile names are set for endpoints using displayName
                try:
                    profile = http_client.profile or {}
                    gclient.display_name = profile.get("displayName")
                    gclient.full_name = profile.get("fullName")
                    self.logger.info(
                        f"👤 Garmin profile displayName={gclient.display_name} fullName={gclient.full_name}"
                    )
                except Exception as _e:
                    self.logger.debug(f"Could not set display/full name from tokens: {_e}")
                self._client = gclient
                self._authenticated = True
                self.logger.info("🔓 Initialized Garmin client from existing tokens (no login)")
                # Optional: quick sanity check that tokens look valid
                try:
                    name = self._client.get_full_name()
                    self.logger.info(f"👤 Profile name: {name}")
                except Exception as e:
                    self.logger.debug(f"Could not read profile name with tokens: {e}")
                return True
            else:
                self.logger.warning("⚠️ Missing token files; cannot initialize from tokens")
                return False
        except Exception as e:
            self.logger.error(f"❌ Token-based init failed: {e}")
            self.logger.error("🔎 Traceback:\n" + traceback.format_exc())
            self._client = None
            self._authenticated = False
            return False

    def test_connection(self) -> bool:
        """Test Garmin token-based login"""
        return self._login_if_needed()
    
    def collect_health_data(self, target_date: date) -> Optional[GarminHealthData]:
        """
        Collect health data for a specific date directly from Garmin.
        Simplified approach, calling API methods directly.
        """
        try:
            self.logger.info(f"📥 Collecting Garmin health data for {target_date}")
            if not self._login_if_needed():
                self.logger.warning("⚠️ Garmin client not authenticated")
                return None

            client = self._client
            iso_date = target_date.isoformat()

            # Initialize raw data containers
            raw_stats: dict = {}
            raw_hr: dict = {}
            raw_sleep: dict = {}
            raw_hrv: dict = {}
            raw_training_readiness: dict = {}
            raw_body_battery: dict = {}
            raw_stress: dict = {}
            raw_respiration: dict = {}
            raw_spo2: dict = {}
            raw_body_composition: dict = {}
            raw_fitnessage: dict = {}
            raw_training_status: dict = {}

            # --- Core Data Fetching ---
            # 1. Basic Stats (Steps)
            try:
                raw_stats = client.get_stats(iso_date)
            except Exception as e:
                self.logger.debug(f"Could not fetch stats for {iso_date}: {e}")

            # 2. Heart Rate (Resting HR)
            try:
                raw_hr = client.get_heart_rates(iso_date)
            except Exception as e:
                self.logger.debug(f"Could not fetch heart rates for {iso_date}: {e}")

            # 3. Sleep Data
            try:
                raw_sleep = client.get_sleep_data(iso_date)
            except Exception as e:
                self.logger.debug(f"Could not fetch sleep data for {iso_date}: {e}")

            # 4. HRV Data
            try:
                raw_hrv = client.get_hrv_data(iso_date)
            except Exception as e:
                self.logger.debug(f"Could not fetch HRV data for {iso_date}: {e}")

            # 5. Training Readiness
            try:
                tr = client.get_training_readiness(iso_date)
                if isinstance(tr, list) and len(tr) > 0:
                    raw_training_readiness = tr[0] if isinstance(tr[0], dict) else {}
                else:
                    raw_training_readiness = tr if isinstance(tr, dict) else {}
            except Exception as e:
                self.logger.debug(f"Could not fetch training readiness for {iso_date}: {e}")

            # 6. Body Battery (range call with same start/end)
            try:
                bb = client.get_body_battery(iso_date, iso_date)
                if isinstance(bb, list) and len(bb) > 0:
                    raw_body_battery = bb[0]
                elif isinstance(bb, dict):
                    raw_body_battery = bb
            except Exception as e:
                self.logger.debug(f"Could not fetch body battery for {iso_date}: {e}")

            # 7. Stress Data
            # 7b. Training Status (optional)
            try:
                ts = client.get_training_status(iso_date)
                if isinstance(ts, list) and len(ts) > 0:
                    raw_training_status = ts[0] if isinstance(ts[0], dict) else {}
                elif isinstance(ts, dict):
                    raw_training_status = ts
            except Exception as e:
                self.logger.debug(f"Could not fetch training status for {iso_date}: {e}")
            try:
                raw_stress = client.get_stress_data(iso_date)
            except Exception as e:
                self.logger.debug(f"Could not fetch stress data for {iso_date}: {e}")

            # 8. Respiration Data
            try:
                raw_respiration = client.get_respiration_data(iso_date)
            except Exception as e:
                self.logger.debug(f"Could not fetch respiration data for {iso_date}: {e}")

            # 9. SpO2 Data
            try:
                raw_spo2 = client.get_spo2_data(iso_date)
            except Exception as e:
                self.logger.debug(f"Could not fetch SpO2 data for {iso_date}: {e}")

            # 10. Body Composition (Weight) - often range based
            try:
                bc = client.get_body_composition(iso_date, iso_date)
                if isinstance(bc, dict) and 'dateWeightList' in bc:
                    weight_list = bc['dateWeightList']
                    if isinstance(weight_list, list) and len(weight_list) > 0:
                        raw_body_composition = weight_list[0]
                elif isinstance(bc, list) and len(bc) > 0:
                    raw_body_composition = bc[0]
                elif isinstance(bc, dict):
                    raw_body_composition = bc
            except Exception as e:
                self.logger.debug(f"Could not fetch body composition for {iso_date}: {e}")

            # 11. Max Metrics (VO2 Max)
            try:
                raw_max_metrics: Any = client.get_max_metrics(iso_date)
                if isinstance(raw_max_metrics, list) and len(raw_max_metrics) > 0:
                    raw_max_metrics = raw_max_metrics[0]
            except Exception as e:
                self.logger.debug(f"Could not fetch max metrics for {iso_date}: {e}")
                raw_max_metrics = {}

            # 11b. Fitness Age (and sometimes VO2-related info)
            try:
                fit = client.get_max_metrics(iso_date)
                if isinstance(fit, dict):
                    raw_fitnessage = fit
            except Exception as e:
                self.logger.debug(f"Could not fetch fitnessage for {iso_date}: {e}")

            # 12. Steps Data (Alternative/Supplemental source)
            steps_data: Any = None
            try:
                steps_data = client.get_steps_data(iso_date)
            except Exception as e:
                self.logger.debug(f"Could not fetch steps data directly for {iso_date}: {e}")
            if not steps_data:
                try:
                    steps_data_list = client.get_daily_steps(iso_date, iso_date)
                    if isinstance(steps_data_list, list) and len(steps_data_list) > 0:
                        steps_data = steps_data_list[0]
                except Exception as e:
                    self.logger.debug(f"Could not fetch daily steps (range) for {iso_date}: {e}")

            # --- Simple Data Extraction ---
            steps = None
            if isinstance(raw_stats, dict):
                ts = raw_stats.get("totalSteps")
                if isinstance(ts, (int, float)):
                    steps = int(ts)
            if steps is None and isinstance(steps_data, dict):
                v = steps_data.get("totalSteps") or steps_data.get("steps") or steps_data.get("value")
                if isinstance(v, (int, float)):
                    steps = int(v)
            if steps is None and isinstance(steps_data, list):
                try:
                    total_calc = 0
                    for interval in steps_data:
                        if isinstance(interval, dict):
                            interval_steps = interval.get("steps")
                            if isinstance(interval_steps, (int, float)):
                                total_calc += int(interval_steps)
                    if total_calc > 0:
                        steps = total_calc
                except Exception as e:
                    self.logger.debug(f"Error calculating steps from list: {e}")

            resting_heart_rate = None
            if isinstance(raw_hr, dict):
                r = raw_hr.get("restingHeartRate")
                if isinstance(r, (int, float)):
                    resting_heart_rate = int(r)
            # Fallback: sometimes RHR is in the user summary stats
            if resting_heart_rate is None and isinstance(raw_stats, dict):
                r = raw_stats.get("restingHeartRate")
                if isinstance(r, (int, float)):
                    resting_heart_rate = int(r)
            # Fallback to dedicated endpoint which may return arrays
            if resting_heart_rate is None:
                try:
                    rhr_data = client.get_rhr_day(iso_date)
                    if isinstance(rhr_data, dict):
                        v = rhr_data.get("restingHeartRate") or rhr_data.get("value")
                        if isinstance(v, (int, float)):
                            resting_heart_rate = int(v)
                        if resting_heart_rate is None:
                            for key in ("metrics", "allMetrics", "userDailySummaryList"):
                                arr = rhr_data.get(key)
                                if isinstance(arr, list):
                                    for item in arr:
                                        if not isinstance(item, dict):
                                            continue
                                        metric_id = item.get("metricId") or item.get("metric")
                                        if metric_id in (60, "RESTING_HEART_RATE"):
                                            if isinstance(item.get("value"), (int, float)):
                                                resting_heart_rate = int(item["value"]) 
                                                break
                                            values = item.get("values")
                                            if isinstance(values, list) and len(values) > 0:
                                                val = values[0]
                                                if isinstance(val, dict):
                                                    vv = val.get("value")
                                                    if isinstance(vv, (int, float)):
                                                        resting_heart_rate = int(vv)
                                                        break
                                    if resting_heart_rate is not None:
                                        break
                except Exception as e:
                    self.logger.debug(f"Could not fetch RHR from dedicated endpoint for {iso_date}: {e}")

            sleep_score = None
            sleep_hours = None
            deep_sleep_minutes = None
            rem_sleep_minutes = None
            if isinstance(raw_sleep, dict):
                sleep_scores = raw_sleep.get("sleepScores", {})
                if isinstance(sleep_scores, dict):
                    overall_score = sleep_scores.get("overall", {})
                    if isinstance(overall_score, dict) and isinstance(overall_score.get("value"), (int, float)):
                        sleep_score = float(overall_score.get("value"))
                daily_sleep_dto = raw_sleep.get("dailySleepDTO", {})
                if isinstance(daily_sleep_dto, dict):
                    # Also support nested score under dailySleepDTO.sleepScores.overall.value
                    if sleep_score is None:
                        dto_scores = daily_sleep_dto.get("sleepScores", {})
                        if isinstance(dto_scores, dict):
                            overall = dto_scores.get("overall", {})
                            if isinstance(overall, dict):
                                val = overall.get("value")
                                if isinstance(val, (int, float)):
                                    sleep_score = float(val)
                    sleep_seconds = daily_sleep_dto.get("sleepTimeSeconds") or daily_sleep_dto.get("overallSleepDuration")
                    if isinstance(sleep_seconds, (int, float)) and sleep_seconds > 0:
                        sleep_hours = float(sleep_seconds) / 3600.0
                    deep_seconds = daily_sleep_dto.get("deepSleepSeconds")
                    rem_seconds = daily_sleep_dto.get("remSleepSeconds")
                    if isinstance(deep_seconds, (int, float)):
                        deep_sleep_minutes = int(deep_seconds / 60)
                    if isinstance(rem_seconds, (int, float)):
                        rem_sleep_minutes = int(rem_seconds / 60)

            hrv_score = None
            hrv_weekly_avg = None
            hrv_status = None
            if isinstance(raw_hrv, dict):
                hrv_summary = raw_hrv.get("hrvSummary", {})
                if isinstance(hrv_summary, dict):
                    hrv_score_val = hrv_summary.get("lastNightAvg") or hrv_summary.get("hrvScore")
                    if isinstance(hrv_score_val, (int, float)):
                        hrv_score = float(hrv_score_val)
                    weekly_avg_val = hrv_summary.get("weeklyAvg")
                    if isinstance(weekly_avg_val, (int, float)):
                        hrv_weekly_avg = float(weekly_avg_val)
                    hrv_status = hrv_summary.get("status")

            training_readiness = None
            training_status_str = None
            # Normalize list -> first dict
            if isinstance(raw_training_readiness, list) and len(raw_training_readiness) > 0:
                raw_tr_first = raw_training_readiness[0] if isinstance(raw_training_readiness[0], dict) else {}
            elif isinstance(raw_training_readiness, dict):
                raw_tr_first = raw_training_readiness
            else:
                raw_tr_first = {}
            if isinstance(raw_tr_first, dict):
                # Try multiple common shapes/keys
                candidates = [
                    raw_tr_first.get("overallScore"),
                    raw_tr_first.get("score"),
                    raw_tr_first.get("dailyScore"),
                    raw_tr_first.get("trainingReadinessLevel"),
                    raw_tr_first.get("readinessScore"),
                ]
                overall_obj = raw_tr_first.get("overall") or raw_tr_first.get("summary")
                if isinstance(overall_obj, dict):
                    candidates.append(overall_obj.get("value"))
                    candidates.append(overall_obj.get("score"))
                for cand in candidates:
                    if isinstance(cand, (int, float)):
                        training_readiness = float(cand)
                        break

            # Parse training status string
            if isinstance(raw_training_status, dict):
                # Common keys observed in Garmin payloads
                for k in (
                    "trainingStatus",
                    "status",
                    "summaryStatus",
                    "training_status",
                    "value",
                ):
                    v = raw_training_status.get(k)
                    if isinstance(v, str) and v:
                        training_status_str = v
                        break

            body_battery = None
            if isinstance(raw_body_battery, dict):
                for key in ("overallAverage", "average", "averageBodyBattery", "avgBodyBattery", "bodyBatteryAverage"):
                    bb_v = raw_body_battery.get(key)
                    if isinstance(bb_v, (int, float)):
                        body_battery = float(bb_v)
                        break
                if body_battery is None:
                    values = raw_body_battery.get("bodyBatteryValues") or raw_body_battery.get("values")
                    if isinstance(values, list) and len(values) > 0:
                        total = 0.0
                        count = 0
                        for item in values:
                            # accept dict with 'value' or pair [ts, value]
                            if isinstance(item, dict):
                                v = item.get("value")
                                if isinstance(v, (int, float)) and v >= 0:
                                    total += float(v)
                                    count += 1
                            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                                v = item[1]
                                if isinstance(v, (int, float)) and v >= 0:
                                    total += float(v)
                                    count += 1
                        if count > 0:
                            body_battery = total / count

            stress_score = None
            max_stress = None
            if isinstance(raw_stress, dict):
                ss_v = raw_stress.get("stressScore") or raw_stress.get("avgStressLevel")
                if isinstance(ss_v, (int, float)):
                    stress_score = float(ss_v)
                ms_v = raw_stress.get("maxStressLevel")
                if isinstance(ms_v, (int, float)):
                    max_stress = float(ms_v)

            vo2max = None
            fitness_age = None
            def find_first_numeric(d: Any, candidate_keys: List[str]) -> Optional[float]:
                try:
                    if isinstance(d, dict):
                        for k in candidate_keys:
                            v = d.get(k)
                            if isinstance(v, (int, float)):
                                return float(v)
                        for v in d.values():
                            r = find_first_numeric(v, candidate_keys)
                            if isinstance(r, (int, float)):
                                return float(r)
                    elif isinstance(d, list):
                        for item in d:
                            r = find_first_numeric(item, candidate_keys)
                            if isinstance(r, (int, float)):
                                return float(r)
                except Exception:
                    return None
                return None

            if isinstance(raw_max_metrics, dict):
                vx = find_first_numeric(raw_max_metrics, ["vo2MaxPreciseValue", "vo2MaxValue", "vo2max"])
                if isinstance(vx, (int, float)):
                    vo2max = float(vx)
            # Extract from fitnessage if available
            if isinstance(raw_fitnessage, dict):
                fa = find_first_numeric(raw_fitnessage, ["fitnessAge", "fitnessAgeYears"])
                if isinstance(fa, (int, float)):
                    fitness_age = float(fa)
                if vo2max is None:
                    v2 = find_first_numeric(raw_fitnessage, ["vo2MaxPreciseValue", "vo2MaxValue", "vo2max"])
                    if isinstance(v2, (int, float)):
                        vo2max = float(v2)
            # Final fallback: GraphQL vo2MaxScalar which includes both
            if vo2max is None or fitness_age is None:
                try:
                    q = {
                        "query": f'query{{vo2MaxScalar(startDate:"{iso_date}", endDate:"{iso_date}")}}'
                    }
                    gql = client.query_garmin_graphql(q)
                    data = (gql or {}).get("data", {})
                    arr = data.get("vo2MaxScalar")
                    if isinstance(arr, list) and len(arr) > 0 and isinstance(arr[0], dict):
                        first = arr[0]
                        if vo2max is None:
                            vx = first.get("vo2MaxPreciseValue") or first.get("vo2MaxValue")
                            if isinstance(vx, (int, float)):
                                vo2max = float(vx)
                        if fitness_age is None:
                            fa = first.get("fitnessAge")
                            if isinstance(fa, (int, float)):
                                fitness_age = float(fa)
                except Exception as e:
                    self.logger.debug(f"GraphQL vo2MaxScalar failed for {iso_date}: {e}")

            body_weight_kg = None
            if isinstance(raw_body_composition, dict):
                body_weight_kg = (
                    raw_body_composition.get("weight")
                    or raw_body_composition.get("weightKg")
                    or raw_body_composition.get("weight_kg")
                )
                if not isinstance(body_weight_kg, (int, float)):
                    body_weight_kg = None

            health_data = GarminHealthData(
                date=target_date,
                steps=steps if isinstance(steps, int) else None,
                resting_heart_rate=resting_heart_rate if isinstance(resting_heart_rate, int) else None,
                sleep_score=sleep_score,
                vo2max=vo2max,
                fitness_age=fitness_age,
                hrv_score=hrv_score,
                hrv_weekly_avg=hrv_weekly_avg,
                hrv_status=hrv_status,
                stress_score=stress_score,
                max_stress=max_stress,
                body_battery=body_battery,
                training_readiness=training_readiness,
                training_status=training_status_str,
                sleep_hours=sleep_hours,
                deep_sleep_minutes=deep_sleep_minutes,
                rem_sleep_minutes=rem_sleep_minutes,
                body_weight_kg=body_weight_kg,
                raw_stats=raw_stats,
                raw_hr=raw_hr,
                raw_sleep=raw_sleep,
                raw_hrv=raw_hrv,
                raw_training_readiness=raw_training_readiness,
                raw_training_status=raw_training_status,
                raw_body_battery=raw_body_battery,
                raw_stress=raw_stress,
                raw_respiration=raw_respiration,
                raw_spo2=raw_spo2,
                raw_body_composition=raw_body_composition,
            )

            if any([
                health_data.steps is not None,
                health_data.resting_heart_rate is not None,
                health_data.sleep_score is not None,
                health_data.vo2max is not None,
                bool(health_data.raw_stats),
                bool(health_data.raw_sleep),
                bool(health_data.raw_hr),
            ]):
                self.logger.info(f"✅ Successfully collected simplified Garmin health data for {target_date}")
                return health_data
            else:
                self.logger.warning(f"⚠️ No meaningful simplified health data found for {target_date}")
                return None

        except GarminConnectConnectionError as e:
            status_code = getattr(e, "status_code", None)
            self.logger.error(
                f"❌ Garmin Connection Error for {target_date}: {e}" 
                + (f" (status={status_code})" if status_code else "")
            )
            self.logger.error("🔎 Traceback:\n" + traceback.format_exc())
            return None
        except GarminConnectAuthenticationError as e:
            self.logger.error(f"❌ Garmin Authentication Error for {target_date}: {e}")
            self.logger.error("🔎 Traceback:\n" + traceback.format_exc())
            return None
        except Exception as e:
            self.logger.error(f"❌ A generic error occurred for {target_date}: {e}")
            self.logger.error("🔎 Traceback:\n" + traceback.format_exc())
            return None
    
    def collect_date_range(self, start_date: date, end_date: date) -> List[GarminHealthData]:
        """
        Collect health data for a date range
        """
        results = []
        current_date = start_date
        
        self.logger.info(f"📅 Collecting Garmin data from {start_date} to {end_date}")
        
        while current_date <= end_date:
            data = self.collect_health_data(current_date)
            if data:
                results.append(data)
            current_date += timedelta(days=1)
        
        self.logger.info(f"✅ Collected {len(results)} days of Garmin health data")
        return results




