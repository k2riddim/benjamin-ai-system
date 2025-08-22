from __future__ import annotations

from typing import Dict, Any, List, Optional
from langsmith.run_helpers import traceable


class DataAnalystTool:
    """Tool for retrieving structured health and fitness metrics data.
    
    This is a non-conversational tool that returns structured data
    for other agents to use in their reasoning.
    """
    
    @traceable(name="data_analyst_tool.get_metric", run_type="tool")
    def get_metric(self, context: Dict[str, Any], metric: str) -> Dict[str, Any]:
        """Retrieve a specific metric from the latest health data.
        
        Returns:
            Dictionary with keys:
            - metric: The requested metric name
            - value: The metric value (None if not found)
            - unit: The unit of measurement (if applicable)
            - label: Human-readable label
            - found: Boolean indicating if metric was found
        """
        latest = context.get("latest_health", {})
        
        # Handle case where the classifier might return the metric as a list
        metric_to_process = metric
        if isinstance(metric, list) and metric:
            metric_to_process = metric[0]

        # Normalize metric name
        metric_lc = (metric_to_process or "").strip().lower()
        
        # Canonical key mapping with common synonyms
        canonical_map: Dict[str, str] = {
            # Resting heart rate
            "resting_heart_rate": "resting_heart_rate",
            "resting hr": "resting_heart_rate",
            "rhr": "resting_heart_rate",
            "restingheartrate": "resting_heart_rate",
            # HRV
            "hrv": "hrv_score",
            "hrv score": "hrv_score",
            # Sleep
            "sleep score": "sleep_score",
            "sleep": "sleep_score",
            "sleep hours": "sleep_hours",
            # Stress/body battery
            "stress": "stress_score",
            "stress score": "stress_score",
            "body battery": "body_battery",
            # Steps
            "steps": "steps",
            # Weight
            "weight": "body_weight_kg",
            "body weight": "body_weight_kg",
            # Training readiness
            "training readiness": "training_readiness",
            "readiness": "training_readiness",
            # Fitness age
            "fitness age": "fitness_age",
            # Performance analysis (special handling)
            "performance": "_performance_analysis",
            "run performance": "_performance_analysis",
            "workout performance": "_performance_analysis",
            "activity performance": "_performance_analysis",
            "running performance": "_performance_analysis",
            # Common conversational metric terms
            "endurance": "_performance_analysis",
            "speed": "_performance_analysis",
            "pace": "_performance_analysis",
            "power": "_performance_analysis",
            "intensity": "_performance_analysis",
            "effort": "_performance_analysis",
            "fitness": "_performance_analysis",
            "condition": "_performance_analysis",
            "form": "_performance_analysis",
            "recovery": "training_readiness",
            "energy": "body_battery",
            "fatigue": "training_readiness",
            "freshness": "training_readiness",
        }
        
        # VO2max: allow modality selection and common spellings
        vo2_aliases = {"vo2max", "vo2 max", "vo2", "vo2max running", "vo2max cycling"}
        is_vo2 = any(a in metric_lc for a in vo2_aliases)
        
        if is_vo2:
            # Disambiguate based on hints present in the metric string
            if "cycl" in metric_lc or "bike" in metric_lc:
                key = "vo2max_cycling"
            elif "run" in metric_lc:
                key = "vo2max_running"
            else:
                # Fallback: prefer cycling if running is missing, else running
                key = "vo2max_cycling" if latest.get("vo2max_cycling") is not None else "vo2max_running"
        else:
            key = canonical_map.get(metric_lc, metric_to_process)
        
        # Special handling for performance analysis
        if key == "_performance_analysis":
            return self._analyze_recent_performance(context)
        
        value = latest.get(key)
        
        # Determine unit based on metric type
        unit = None
        if key.startswith("vo2max"):
            unit = "ml/kg/min"
        elif key == "resting_heart_rate":
            unit = "bpm"
        elif key == "body_weight_kg":
            unit = "kg"
        elif key == "sleep_hours":
            unit = "hours"
        elif key == "steps":
            unit = "steps"
        elif key in ["hrv_score", "sleep_score", "stress_score", "body_battery", "training_readiness"]:
            unit = "score"
        elif key == "fitness_age":
            unit = "years"
        
        # Provide helpful suggestions for unknown metrics
        if value is None and key not in ["_performance_analysis"] and key not in latest:
            # Check if this might be a fuzzy match for available metrics
            available_metrics = [k for k in latest.keys() if latest[k] is not None]
            suggestion = self._suggest_similar_metric(metric_to_process, available_metrics)
            
            return {
                "metric": key,
                "value": None,
                "unit": unit,
                "label": key.replace("_", " ").title(),
                "found": False,
                "suggestion": suggestion,
                "available_metrics": available_metrics[:5]  # Top 5 available metrics
            }
        
        return {
            "metric": key,
            "value": value,
            "unit": unit,
            "label": key.replace("_", " ").title(),
            "found": value is not None
        }
    
    @traceable(name="data_analyst_tool.get_training_load", run_type="tool")
    def get_training_load(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Retrieve training load metrics.
        
        Returns:
            Dictionary with training load data including acute and chronic loads
        """
        tl = context.get("training_load", []) or []
        if not tl or not isinstance(tl, list):
            return {
                "acute_load_7d": None,
                "chronic_load_28d": None,
                "ratio": None,
                "found": False
            }
        
        try:
            last = tl[-1] if tl else {}
            acute = float(last.get("acute_load_7d") or 0)
            chronic = float(last.get("chronic_load_28d") or 1)
            ratio = (acute / chronic) if chronic > 0 else 0
            
            return {
                "acute_load_7d": acute,
                "chronic_load_28d": chronic,
                "ratio": round(ratio, 2),
                "found": True
            }
        except Exception:
            return {
                "acute_load_7d": None,
                "chronic_load_28d": None,
                "ratio": None,
                "found": False
            }
    
    @traceable(name="data_analyst_tool.get_fitness_trends", run_type="tool")
    def get_fitness_trends(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze fitness trends from available data.
        
        Returns:
            Dictionary with trend analysis including VO2max trends, weight changes, etc.
        """
        trends = {}
        latest = context.get("latest_health", {}) or {}
        wt = context.get("weight_recent", []) or []
        acts = context.get("activities_recent", []) or []
        
        # VO2max values
        vo2_running = latest.get("vo2max_running")
        vo2_cycling = latest.get("vo2max_cycling")
        if vo2_running is not None:
            trends["vo2max_running"] = vo2_running
        if vo2_cycling is not None:
            trends["vo2max_cycling"] = vo2_cycling
        
        # Weight trend
        if wt and isinstance(wt, list) and len(wt) >= 2:
            try:
                start = wt[0].get("weight_kg")
                end = wt[-1].get("weight_kg")
                if isinstance(start, (int, float)) and isinstance(end, (int, float)):
                    trends["weight_change_60d"] = round(end - start, 1)
                    trends["weight_current"] = end
            except Exception:
                pass
        
        # Activity volume
        if acts and isinstance(acts, list):
            try:
                trends["activities_14d"] = len([a for a in acts if a])
            except Exception:
                pass
        
        # Training load
        load_data = self.get_training_load(context)
        if load_data["found"]:
            trends["training_load"] = {
                "acute_7d": load_data["acute_load_7d"],
                "chronic_28d": load_data["chronic_load_28d"],
                "ratio": load_data["ratio"]
            }
        
        return trends
    
    def _analyze_recent_performance(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze recent activity performance and provide insights."""
        activities = context.get("activities_recent", []) or []
        latest_health = context.get("latest_health", {}) or {}
        
        if not activities:
            return {
                "metric": "performance",
                "value": None,
                "unit": None,
                "label": "Performance Analysis",
                "found": False,
                "message": "No recent activities found to analyze"
            }
        
        # Get the most recent activity (likely what user is asking about)
        latest_activity = activities[0] if activities else None
        
        if not latest_activity:
            return {
                "metric": "performance",
                "value": None,
                "unit": None,
                "label": "Performance Analysis",
                "found": False,
                "message": "No recent activity data available"
            }
        
        # Build performance summary
        activity_type = latest_activity.get("activity_type", "Activity")
        duration_min = latest_activity.get("duration", 0)
        distance_km = latest_activity.get("distance", 0)
        avg_hr = latest_activity.get("average_heart_rate")
        avg_watts = latest_activity.get("average_watts")
        
        # Calculate pace if it's a run
        pace_per_km = None
        if activity_type == "Run" and distance_km and duration_min:
            pace_per_km = duration_min / distance_km
        
        # Build analysis message
        performance_parts = []
        performance_parts.append(f"{activity_type}: {distance_km:.1f}km in {duration_min:.0f} minutes")
        
        if pace_per_km:
            pace_min = int(pace_per_km)
            pace_sec = int((pace_per_km - pace_min) * 60)
            performance_parts.append(f"Pace: {pace_min}:{pace_sec:02d}/km")
        
        if avg_hr:
            performance_parts.append(f"Avg HR: {avg_hr:.0f} bpm")
        
        if avg_watts:
            performance_parts.append(f"Avg Power: {avg_watts:.0f}W")
        
        # Add readiness context
        readiness = latest_health.get("training_readiness")
        if readiness is not None:
            if readiness >= 70:
                readiness_note = "good readiness"
            elif readiness >= 40:
                readiness_note = "moderate readiness"  
            else:
                readiness_note = "low readiness"
            performance_parts.append(f"Training readiness: {readiness:.0f}/100 ({readiness_note})")
        
        analysis = ". ".join(performance_parts)
        
        return {
            "metric": "performance",
            "value": analysis,
            "unit": None,
            "label": "Latest Activity Performance",
            "found": True
        }
    
    def _suggest_similar_metric(self, requested_metric: str, available_metrics: List[str]) -> str | None:
        """Suggest a similar available metric based on fuzzy matching."""
        if not available_metrics:
            return None
        
        requested_lower = requested_metric.lower()
        
        # Simple keyword matching for suggestions
        for metric in available_metrics:
            metric_lower = metric.lower()
            # Check for partial matches
            if any(word in metric_lower for word in requested_lower.split("_") if len(word) > 2):
                return metric
        
        # Common metric mappings for suggestions
        suggestion_map = {
            "heart": "resting_heart_rate",
            "hr": "resting_heart_rate", 
            "weight": "body_weight_kg",
            "sleep": "sleep_score",
            "stress": "stress_score",
            "ready": "training_readiness",
            "battery": "body_battery",
            "vo2": "vo2max_running",
            "fitness": "vo2max_running"
        }
        
        for keyword, suggested_metric in suggestion_map.items():
            if keyword in requested_lower and suggested_metric in available_metrics:
                return suggested_metric
        
        return None
    
    @traceable(name="data_analyst_tool.get_recovery_metrics", run_type="tool")
    def get_recovery_metrics(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Get recovery-related metrics for decision making.
        
        Returns:
            Dictionary with recovery metrics like readiness, sleep, HRV
        """
        latest = context.get("latest_health", {}) or {}
        
        recovery_data = {}
        
        # Get key recovery metrics
        readiness = latest.get("training_readiness")
        if readiness is not None:
            recovery_data["training_readiness"] = readiness
        
        sleep_score = latest.get("sleep_score")
        if sleep_score is not None:
            recovery_data["sleep_score"] = sleep_score
        
        sleep_hours = latest.get("sleep_hours")
        if sleep_hours is not None:
            recovery_data["sleep_hours"] = sleep_hours
        
        hrv = latest.get("hrv_score")
        if hrv is not None:
            recovery_data["hrv_score"] = hrv
        
        stress = latest.get("stress_score")
        if stress is not None:
            recovery_data["stress_score"] = stress
        
        body_battery = latest.get("body_battery")
        if body_battery is not None:
            recovery_data["body_battery"] = body_battery
        
        rhr = latest.get("resting_heart_rate")
        if rhr is not None:
            recovery_data["resting_heart_rate"] = rhr
        
        # Calculate recovery status
        recovery_data["needs_recovery"] = self._assess_recovery_need(recovery_data)
        
        return recovery_data
    
    def _assess_recovery_need(self, metrics: Dict[str, Any]) -> bool:
        """Assess if recovery is needed based on metrics."""
        readiness = metrics.get("training_readiness", 100)
        sleep_score = metrics.get("sleep_score", 100)
        
        # Recovery triggers
        if readiness is not None and readiness <= 25:
            return True
        if sleep_score is not None and sleep_score <= 50:
            return True
        
        return False
    
    @traceable(name="data_analyst_tool.get_all_metrics", run_type="tool")
    def get_all_metrics(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Get all available metrics in a structured format.
        
        Returns:
            Comprehensive dictionary with all available metrics
        """
        return {
            "latest_health": context.get("latest_health", {}),
            "training_load": self.get_training_load(context),
            "fitness_trends": self.get_fitness_trends(context),
            "recovery_metrics": self.get_recovery_metrics(context),
            "weather": context.get("weather_vincennes", {}),
            "upcoming_events": context.get("upcoming_events", [])
        }


# Global instance for easy access
data_analyst_tool = DataAnalystTool()
