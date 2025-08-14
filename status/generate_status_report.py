#!/usr/bin/env python3
"""
System status report generator for Benjamin AI System.

Checks:
- Data App: DB connectivity, latest health/activity dates, Data API endpoints
- Services: systemd statuses for data/agentic/telegram units
- Agentic App: API ping, Qdrant connectivity, OpenAI key presence
- Telegram: service status and configuration presence

Writes a timestamped Markdown report under status/reports/.
Run inside bbox_env with PYTHONPATH pointing to project root.
"""

from __future__ import annotations

import os
import sys
import json
import socket
import subprocess
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = PROJECT_ROOT / "status" / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def safe_call(cmd: list[str]) -> tuple[int, str, str]:
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return proc.returncode, proc.stdout.strip(), proc.stderr.strip()
    except Exception as e:
        return 1, "", str(e)


def _load_env_file(path: Path) -> None:
    """Best-effort load simple KEY=VALUE lines from a .env file into os.environ if not set."""
    try:
        if not path.exists():
            return
        for raw in path.read_text().splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, val = line.split("=", 1)
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = val
    except Exception:
        pass


def resolve_data_api_url() -> str:
    """Resolve Data API base URL from environment with sensible fallbacks."""
    return (
        os.environ.get("AGENTIC_APP_DATA_API_BASE_URL")
        or os.environ.get("TELEGRAM_APP_DATA_API_BASE_URL")
        or "http://127.0.0.1:8010"
    )


def resolve_agentic_api_url() -> str:
    """Resolve Agentic API base URL from environment with sensible fallbacks."""
    return (
        os.environ.get("TELEGRAM_APP_AGENTIC_API_BASE_URL")
        or "http://127.0.0.1:8012"
    )


def resolve_qdrant_url() -> str:
    """Resolve Qdrant base URL from agentic env, with default to local port 6333."""
    return os.environ.get("AGENTIC_APP_QDRANT_URL", "http://127.0.0.1:6333").rstrip("/")


def qdrant_headers() -> dict:
    """Return headers for Qdrant if API key is configured."""
    key = os.environ.get("AGENTIC_APP_QDRANT_API_KEY", "").strip()
    return {"api-key": key} if key else {}


def read_unit_env_var(unit_file: str, var_name: str) -> Optional[str]:
    """Best-effort parse of a systemd unit file to extract Environment var values."""
    try:
        path = Path(unit_file)
        if not path.exists():
            return None
        for line in path.read_text().splitlines():
            line = line.strip()
            if line.startswith("Environment=") and (var_name + "=") in line:
                # Environment=VAR=value or multiple vars on one line
                parts = line[len("Environment="):].split()
                for part in parts:
                    if part.startswith(var_name + "="):
                        val = part.split("=", 1)[1]
                        return val
        return None
    except Exception:
        return None


def header(title: str) -> str:
    return f"\n## {title}\n\n"


def kv(key: str, value: Any) -> str:
    return f"- **{key}**: {value}\n"


def check_database() -> Dict[str, Any]:
    sys.path.append(str(PROJECT_ROOT))
    out: Dict[str, Any] = {"ok": False}
    try:
        # Ensure service env files are loaded so shared.database can find URLs
        _load_env_file(PROJECT_ROOT / "data_app" / "config" / ".env")
        _load_env_file(PROJECT_ROOT / "agentic_app" / "config" / ".env")
        from shared.database import get_db_session, HealthMetrics, Activities, test_connection
        out["connect_ok"] = bool(test_connection())
        latest_health: Optional[date] = None
        latest_activity: Optional[datetime] = None
        with get_db_session() as db:
            latest_h = db.query(HealthMetrics).order_by(HealthMetrics.date.desc()).first()
            latest_a = db.query(Activities).order_by(Activities.start_date.desc()).first()
            latest_health = latest_h.date if latest_h else None
            latest_activity = latest_a.start_date if latest_a else None
        out["latest_health_date"] = latest_health.isoformat() if latest_health else None
        out["latest_activity_date"] = latest_activity.isoformat() if latest_activity else None
        out["ok"] = out["connect_ok"] is True
    except Exception as e:
        out["error"] = repr(e)
        # Hint which env vars are present to aid debugging
        out["env_DATA_APP_DATABASE_URL_present"] = bool(os.environ.get("DATA_APP_DATABASE_URL"))
        out["env_AGENTIC_APP_DATABASE_URL_present"] = bool(os.environ.get("AGENTIC_APP_DATABASE_URL"))
    return out


def check_data_api() -> Dict[str, Any]:
    sys.path.append(str(PROJECT_ROOT))
    out: Dict[str, Any] = {"ok": False}
    try:
        from data_app.config.settings import settings as data_settings
        base = data_settings.database_url  # for visibility
        api = getattr(data_settings, "strava_api_base_url", "n/a")
        out["config_database_url"] = data_settings.database_url
        out["config_api_garmin_base"] = getattr(data_settings, "garmin_api_base_url", "n/a")
        data_api_url = resolve_data_api_url()
        root = requests.get(f"{data_api_url}/", timeout=5)
        health = requests.get(f"{data_api_url}/health", timeout=5)
        recent = requests.get(f"{data_api_url}/activities/recent", timeout=10)
        out["root_status"] = root.status_code
        out["health_status"] = health.status_code
        out["recent_status"] = recent.status_code
        out["recent_count"] = len(recent.json()) if recent.ok else None
        out["ok"] = root.ok and health.ok
    except Exception as e:
        out["error"] = repr(e)
        out["data_api_base_url"] = resolve_data_api_url()
    return out


def check_services() -> Dict[str, Any]:
    units = [
        "benjamin-data-app.service",
        "benjamin-data-api.service",
        "benjamin-agentic-api.service",
        "benjamin-agentic-discussion.service",
        "benjamin-agentic-discussion.timer",
        "benjamin-telegram-app.service",
        "benjamin-status-report.timer",
    ]
    status: Dict[str, Any] = {}
    for u in units:
        code, out, err = safe_call(["systemctl", "is-active", u])
        # Show whatever systemctl reports, even if non-zero (e.g., inactive/failed)
        status[u] = out or err or "unknown"
    return status


def check_agentic() -> Dict[str, Any]:
    sys.path.append(str(PROJECT_ROOT))
    out: Dict[str, Any] = {"ok": False}
    try:
        # Load agentic env file so URLs/keys are available
        _load_env_file(PROJECT_ROOT / "agentic_app" / "config" / ".env")
        # Ping Agentic API
        ok_root = False
        base = resolve_agentic_api_url().rstrip("/")
        for url in [f"{base}/", f"{base}/health"]:
            try:
                r = requests.get(url, timeout=3)
                if r.ok:
                    ok_root = True
                    break
            except Exception:
                pass
        out["api_ok"] = ok_root

        # Qdrant: query Qdrant server directly (not via Agentic API)
        try:
            qbase = resolve_qdrant_url()
            rq = requests.get(f"{qbase}/collections", headers=qdrant_headers(), timeout=5)
            out["qdrant_url"] = qbase
            if rq.ok:
                payload = rq.json()
                cols: list[str] = []
                if isinstance(payload, dict):
                    if isinstance(payload.get("result"), list):
                        cols = [c.get("name") for c in payload.get("result", []) if isinstance(c, dict) and c.get("name")]
                    elif isinstance(payload.get("collections"), list):
                        cols = [c.get("name") for c in payload.get("collections", []) if isinstance(c, dict) and c.get("name")]
                out["qdrant_collections"] = cols
                out["qdrant_ok"] = True
            else:
                out["qdrant_ok"] = False
                out["qdrant_error"] = f"http_status={rq.status_code}"
        except Exception as e:
            out["qdrant_ok"] = False
            out["qdrant_error"] = repr(e)

        # OpenAI key presence: check env of systemd unit if not in current env
        present = bool(os.environ.get("AGENTIC_APP_OPENAI_API_KEY"))
        if not present:
            unit_val = read_unit_env_var("/etc/systemd/system/benjamin-agentic-api.service", "AGENTIC_APP_OPENAI_API_KEY")
            present = bool(unit_val)
        out["openai_key_present"] = present
        out["agentic_api_base_url"] = base
        out["ok"] = ok_root or out.get("qdrant_ok", False)
    except Exception as e:
        out["error"] = repr(e)
    return out


def check_telegram() -> Dict[str, Any]:
    sys.path.append(str(PROJECT_ROOT))
    out: Dict[str, Any] = {}
    try:
        # Load telegram env file for tokens if not present
        _load_env_file(PROJECT_ROOT / "telegram_app" / "config" / ".env")
        from telegram_app.config.settings import settings as tg_settings
        # Check current env or settings
        token_present = bool(getattr(tg_settings, "telegram_bot_token", "")) or bool(os.environ.get("TELEGRAM_APP_TELEGRAM_BOT_TOKEN"))
        chat_id_present = bool(getattr(tg_settings, "telegram_chat_id", "")) or bool(os.environ.get("TELEGRAM_APP_TELEGRAM_CHAT_ID"))
        # If not present, attempt to read from systemd unit file env
        if not token_present:
            unit_val = read_unit_env_var("/etc/systemd/system/benjamin-telegram-app.service", "TELEGRAM_APP_TELEGRAM_BOT_TOKEN")
            token_present = bool(unit_val)
        if not chat_id_present:
            unit_val = read_unit_env_var("/etc/systemd/system/benjamin-telegram-app.service", "TELEGRAM_APP_TELEGRAM_CHAT_ID")
            chat_id_present = bool(unit_val)
        out["token_present"] = token_present
        out["chat_id_present"] = chat_id_present
        out["data_api_base_url"] = resolve_data_api_url()
        out["agentic_api_base_url"] = resolve_agentic_api_url()
        code, active, _ = safe_call(["systemctl", "is-active", "benjamin-telegram-app.service"])
        out["service_status"] = active or "unknown"
    except Exception as e:
        out["error"] = repr(e)
    return out


def main() -> int:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report_path = REPORTS_DIR / f"system_status_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    lines: list[str] = []
    lines.append(f"# Benjamin AI System Status\n\nGenerated: {ts}\n")

    # Data: DB and API
    lines.append(header("Data App"))
    db = check_database()
    lines.append(kv("DB connect", db.get("connect_ok")))
    lines.append(kv("Latest health date", db.get("latest_health_date")))
    lines.append(kv("Latest activity date", db.get("latest_activity_date")))
    if db.get("error"):
        lines.append(kv("DB error", db["error"]))

    api = check_data_api()
    lines.append(kv("Data API root", api.get("root_status")))
    lines.append(kv("Data API health", api.get("health_status")))
    lines.append(kv("Recent activities count", api.get("recent_count")))
    if api.get("error"):
        lines.append(kv("API error", api["error"]))

    # Services
    lines.append(header("Services"))
    svc = check_services()
    for name, st in svc.items():
        lines.append(kv(name, st))

    # Agentic
    lines.append(header("Agentic App"))
    ag = check_agentic()
    for k, v in ag.items():
        if k != "qdrant_collections":
            lines.append(kv(k, v))
    if ag.get("qdrant_collections"):
        lines.append("- **qdrant_collections**: " + ", ".join(ag["qdrant_collections"]) + "\n")

    # Telegram
    lines.append(header("Telegram"))
    tg = check_telegram()
    for k, v in tg.items():
        lines.append(kv(k, v))

    report_path.write_text("".join(lines))
    print(str(report_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())