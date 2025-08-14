#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import json
import os
import sys
from dataclasses import asdict, is_dataclass
from datetime import date, datetime, timedelta
from typing import Any, Dict

# Ensure project root is on path (two levels up from this scripts directory)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from data_app.src.garmin_collector import GarminCollector  # noqa: E402


def parse_target_date(value: str) -> date:
    v = value.strip().lower()
    if v in ("today", "tod", "t"):
        return date.today()
    if v in ("yesterday", "yday", "y"):
        return date.today() - timedelta(days=1)
    # Support relative offsets like -1, +0, +2 (days from today)
    try:
        if v.startswith("+") or v.startswith("-"):
            offset = int(v)
            return date.today() + timedelta(days=offset)
    except Exception:
        pass
    # Fallback to ISO date
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid date '{value}'. Use YYYY-MM-DD, 'today', 'yesterday', or relative like -1"
        ) from exc


def to_json_serializable(obj: Any, summary_only: bool) -> Dict[str, Any]:
    if is_dataclass(obj):
        data = asdict(obj)
    elif isinstance(obj, dict):
        data = dict(obj)
    else:
        data = {"value": obj}

    # Ensure date becomes ISO string
    if isinstance(data.get("date"), date):
        data["date"] = data["date"].isoformat()

    if summary_only:
        data = {k: v for k, v in data.items() if not str(k).startswith("raw_")}

    return data


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Collect Garmin health data for a single day and print JSON."
    )
    parser.add_argument(
        "date",
        nargs="?",
        default="today",
        type=parse_target_date,
        help="Target day (YYYY-MM-DD | today | yesterday | +/-offset). Default: today",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print summary fields only (omit raw_* payloads)",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON",
    )
    parser.add_argument(
        "--log",
        default="WARNING",
        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"],
        help="Log level (default: WARNING)",
    )

    args = parser.parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log.upper(), logging.WARNING))

    collector = GarminCollector()

    if not collector.test_connection():
        print("Garmin auth failed. Ensure GARTH_HOME tokens are present.", file=sys.stderr)
        return 1

    data = collector.collect_health_data(args.date)
    if not data:
        print(f"No Garmin data found for {args.date.isoformat()}", file=sys.stderr)
        return 2

    payload = to_json_serializable(data, summary_only=args.summary)
    json_kwargs = {"indent": 2, "sort_keys": True} if args.pretty else {}
    print(json.dumps(payload, **json_kwargs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


