import logging
import os
import threading
import time
from datetime import datetime, timedelta
from typing import Dict

from app.email_utils import send_email
from app.usage_log import usage_logger

logger = logging.getLogger("centaurweb.usage_reporter")

REPORT_HOUR = int(os.getenv("USAGE_REPORT_HOUR", "2"))
REPORT_MINUTE = int(os.getenv("USAGE_REPORT_MINUTE", "0"))


def _next_run_time() -> (datetime, float):
    now = datetime.now()
    target = now.replace(hour=REPORT_HOUR, minute=REPORT_MINUTE, second=0, microsecond=0)
    if target <= now:
        target += timedelta(days=1)
    return target, (target - now).total_seconds()


def _build_body(day: str, summary: Dict[str, int]) -> str:
    lines = [f"Centaur usage report for {day}"]
    lines.append(f"Sessions created: {summary.get('events_session_created', 0)}")
    lines.append(f"Chunks processed: {summary.get('events_chunk', 0)} (errors: {summary.get('errors_chunk', 0)})")
    lines.append(f"SOAP calls: {summary.get('events_soap', 0)} (errors: {summary.get('errors_soap', 0)})")
    lines.append(f"Billing calls: {summary.get('events_billing', 0)} (errors: {summary.get('errors_billing', 0)})")
    lines.append(f"Patient summaries: {summary.get('events_summary', 0)} (errors: {summary.get('errors_summary', 0)})")
    lines.append(f"Differentials: {summary.get('events_differential', 0)} (errors: {summary.get('errors_differential', 0)})")
    return "\n".join(lines)


def _run_report_loop() -> None:
    while True:
        target, wait = _next_run_time()
        logger.info("Usage reporter sleeping %.0f seconds until %s", wait, target.isoformat())
        time.sleep(wait)
        day = (target - timedelta(days=1)).strftime("%Y-%m-%d")
        summary = usage_logger.summarize_day(day)
        body = _build_body(day, summary)
        success = send_email(f"Centaur usage report {day}", body)
        logger.info("Usage report for %s sent: %s", day, success)


def start_daily_reporter() -> None:
    if os.getenv("DISABLE_USAGE_REPORT", "0") == "1":
        logger.info("Usage reporter disabled via env")
        return
    thread = threading.Thread(target=_run_report_loop, daemon=True)
    thread.start()
