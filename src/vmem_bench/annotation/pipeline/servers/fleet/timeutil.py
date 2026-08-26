"""Beijing (Asia/Shanghai) wall-clock helpers for console + fleet timestamps."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

try:
    BEIJING = ZoneInfo("Asia/Shanghai")
except ZoneInfoNotFoundError:
    # Minimal images may lack the tzdata package; Asia/Shanghai is UTC+8 year-round.
    BEIJING = timezone(timedelta(hours=8), name="Asia/Shanghai")


def now_beijing() -> str:
    """Return ``YYYY-MM-DD HH:MM:SS`` in Asia/Shanghai (no timezone suffix)."""
    return datetime.now(BEIJING).strftime("%Y-%m-%d %H:%M:%S")


def now_beijing_iso() -> str:
    """Return ISO-8601 with ``+08:00`` offset."""
    return datetime.now(BEIJING).isoformat(timespec="seconds")
