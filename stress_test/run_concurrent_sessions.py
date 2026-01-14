import argparse
import json
import math
import os
import random
import threading
import time
from typing import Dict, List, Optional, Tuple

from stress_test.lib import (
    ApiClient,
    EventLogger,
    SystemMonitor,
    WavSegment,
    collect_audio_files,
    find_uvicorn_pid,
    iter_wav_segments,
    read_jsonl,
)


DEFAULT_BASE_URL = os.getenv("CENTAUR_BASE_URL", "http://127.0.0.1:8000")
DEFAULT_USER = os.getenv("CENTAUR_USERNAME", "")
DEFAULT_PASS = os.getenv("CENTAUR_PASSWORD", "")


def _percentile(values: List[float], pct: float) -> Optional[float]:
    if not values:
        return None
    values_sorted = sorted(values)
    k = int(math.ceil((pct / 100.0) * len(values_sorted))) - 1
    k = max(0, min(k, len(values_sorted) - 1))
    return values_sorted[k]


def _ascii_bar(values: List[float], width: int = 40, max_value: float = 100.0) -> List[str]:
    lines: List[str] = []
    if not values:
        return ["(no data)"]
    for v in values:
        ratio = min(max(v / max_value, 0.0), 1.0)
        filled = int(ratio * width)
        lines.append("#" * filled)
    return lines


def _write_report(report_path: str, events: List[dict]) -> None:
    chunk_latencies = [e["latency_s"] for e in events if e.get("type") == "chunk" and "latency_s" in e]
    chunk_ratios = [
        e["latency_s"] / e["duration_s"]
        for e in events
        if e.get("type") == "chunk" and e.get("latency_s") and e.get("duration_s")
    ]
    soap_lat = [e["latency_s"] for e in events if e.get("type") == "soap"]
    summary_lat = [e["latency_s"] for e in events if e.get("type") == "summary"]
    billing_lat = [e["latency_s"] for e in events if e.get("type") == "billing"]
    coach_lat = [e["latency_s"] for e in events if e.get("type") == "coach"]

    drops = sum(1 for e in events if e.get("type") == "drop")
    errors = sum(1 for e in events if e.get("type") == "error")
    max_queue = max([e.get("queue_depth", 0) for e in events if e.get("type") == "queue_depth"] or [0])

    ratio_p95 = _percentile(chunk_ratios, 95) or 0.0
    chunk_p95 = _percentile(chunk_latencies, 95) or 0.0
    soap_p95 = _percentile(soap_lat, 95) or 0.0

    pass_perf = ratio_p95 < 2.0 and soap_p95 < 8.0
    pass_stability = drops == 0 and errors == 0
    pass_queue = max_queue <= 10
    overall_pass = pass_perf and pass_stability and pass_queue

    sys_cpu = [e.get("sys_cpu_percent") for e in events if e.get("type") == "system" and e.get("sys_cpu_percent") is not None]
    sys_mem = [e.get("sys_mem_percent") for e in events if e.get("type") == "system" and e.get("sys_mem_percent") is not None]

    lines: List[str] = []
    lines.append("# Centaur Stress Test Report")
    lines.append("")
    lines.append("## Summary")
    lines.append(f"- Chunk P95 latency: {chunk_p95:.2f}s")
    lines.append(f"- Chunk latency ratio P95 (latency/duration): {ratio_p95:.2f}x")
    lines.append(f"- SOAP P95 latency: {soap_p95:.2f}s")
    lines.append(f"- Drops: {drops}")
    lines.append(f"- Errors: {errors}")
    lines.append(f"- Max queue depth: {max_queue}")
    lines.append(f"- Result: {'PASS' if overall_pass else 'FAIL'}")
    lines.append("")
    lines.append("## Latencies")
    lines.append(f"- Chunk P50: {_percentile(chunk_latencies, 50) or 0:.2f}s")
    lines.append(f"- Chunk P95: {_percentile(chunk_latencies, 95) or 0:.2f}s")
    lines.append(f"- Chunk P99: {_percentile(chunk_latencies, 99) or 0:.2f}s")
    lines.append(f"- SOAP P50: {_percentile(soap_lat, 50) or 0:.2f}s")
    lines.append(f"- SOAP P95: {_percentile(soap_lat, 95) or 0:.2f}s")
    lines.append(f"- SOAP P99: {_percentile(soap_lat, 99) or 0:.2f}s")
    lines.append(f"- Summary P95: {_percentile(summary_lat, 95) or 0:.2f}s")
    lines.append(f"- Billing P95: {_percentile(billing_lat, 95) or 0:.2f}s")
    lines.append(f"- Coach P95: {_percentile(coach_lat, 95) or 0:.2f}s")
    lines.append("")
    lines.append("## System (sampled)")
    if sys_cpu:
        lines.append("- CPU samples (ascii, 0-100):")
        lines.extend(["  " + s for s in _ascii_bar([v for v in sys_cpu if v is not None], width=30, max_value=100.0)])
    else:
        lines.append("- CPU samples: (no data)")
    if sys_mem:
        lines.append("- RAM samples (ascii, 0-100):")
        lines.extend(["  " + s for s in _ascii_bar([v for v in sys_mem if v is not None], width=30, max_value=100.0)])
    else:
        lines.append("- RAM samples: (no data)")
    lines.append("")
    lines.append("## Conclusion")
    if overall_pass:
        lines.append("Safe for current concurrency under this run.")
    else:
        lines.append("Not safe for current concurrency; see metrics above.")

    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


class SessionRunner:
    def __init__(
        self,
        client: ApiClient,
        logger: EventLogger,
        audio_path: str,
        start_delay_s: float,
        max_queue: int,
        do_coach: bool,
        do_regen: bool,
        session_label: str,
    ) -> None:
        self.client = client
        self.logger = logger
        self.audio_path = audio_path
        self.start_delay_s = start_delay_s
        self.max_queue = max_queue
        self.do_coach = do_coach
        self.do_regen = do_regen
        self.session_label = session_label
        self.session_id = ""
        self._queue: "queue.Queue[dict]" = queue.Queue()
        self._producer_done = threading.Event()
        self._consumer_done = threading.Event()
        self._chunk_count = 0
        self._lock = threading.Lock()

    def run(self) -> None:
        time.sleep(self.start_delay_s)
        self.session_id = self.client.create_session()
        self.logger.log({"type": "session_start", "session_id": self.session_id, "label": self.session_label})

        prod = threading.Thread(target=self._producer, daemon=True)
        cons = threading.Thread(target=self._consumer, daemon=True)
        prod.start()
        cons.start()
        prod.join()
        cons.join()

        self._run_post_actions()
        self.logger.log({"type": "session_done", "session_id": self.session_id, "label": self.session_label})

    def _producer(self) -> None:
        for seg in iter_wav_segments(self.audio_path):
            item = {
                "segment": seg,
                "queued_at": time.time(),
            }
            with self._lock:
                if self._queue.qsize() >= self.max_queue:
                    try:
                        _ = self._queue.get_nowait()
                        self.logger.log({"type": "drop", "session_id": self.session_id, "label": self.session_label})
                    except Exception:
                        pass
                self._queue.put(item)
                self.logger.log({"type": "queue_depth", "session_id": self.session_id, "queue_depth": self._queue.qsize()})
            time.sleep(seg.duration_s)
        self._producer_done.set()

    def _consumer(self) -> None:
        coach_trigger_chunk = None
        try:
            total_segments = sum(1 for _ in iter_wav_segments(self.audio_path))
            coach_trigger_chunk = max(2, total_segments // 3)
        except Exception:
            coach_trigger_chunk = 3

        while not (self._producer_done.is_set() and self._queue.empty()):
            try:
                item = self._queue.get(timeout=1)
            except queue.Empty:
                continue
            seg: WavSegment = item["segment"]
            queued_at = item["queued_at"]
            start = time.time()
            wait_s = start - queued_at
            status, obj, retries = self._post_multipart_with_retry(
                f"/api/transcribe_chunk?session_id={self.session_id}",
                seg,
            )
            latency = time.time() - start
            text_len = len((obj or {}).get("text") or "")
            self._chunk_count += 1
            self.logger.log({
                "type": "chunk",
                "session_id": self.session_id,
                "label": self.session_label,
                "chunk_index": seg.index,
                "duration_s": seg.duration_s,
                "queue_wait_s": wait_s,
                "latency_s": latency,
                "status": status,
                "retries": retries,
                "text_len": text_len,
            })
            if status != 200:
                self.logger.log({"type": "error", "session_id": self.session_id, "label": self.session_label, "stage": "transcribe", "status": status, "detail": obj})

            if self.do_coach and coach_trigger_chunk and self._chunk_count == coach_trigger_chunk:
                self._call_coach()
        self._consumer_done.set()

    def _post_multipart_with_retry(self, path: str, seg: WavSegment) -> Tuple[int, dict, int]:
        attempts = 0
        backoff = 1.0
        last_status = 0
        last_obj: dict = {}
        while attempts < 3:
            status, obj = self.client.post_multipart(
                path,
                field_name="file",
                filename=f"chunk_{seg.index}.wav",
                content_type="audio/wav",
                data=seg.data,
            )
            last_status = status
            last_obj = obj
            if status == 200:
                return status, obj, attempts
            attempts += 1
            time.sleep(backoff)
            backoff = min(backoff * 2, 8.0)
        return last_status, last_obj, attempts

    def _call_coach(self) -> None:
        start = time.time()
        status, obj = self.client.post_json(f"/api/session/{self.session_id}/differential", {})
        latency = time.time() - start
        self.logger.log({
            "type": "coach",
            "session_id": self.session_id,
            "label": self.session_label,
            "latency_s": latency,
            "status": status,
        })
        if status != 200:
            self.logger.log({"type": "error", "session_id": self.session_id, "label": self.session_label, "stage": "coach", "status": status, "detail": obj})

    def _run_post_actions(self) -> None:
        # Patient summary
        start = time.time()
        status, obj, retries = self._post_json_with_retry(f"/api/session/{self.session_id}/patient_summary")
        self.logger.log({"type": "summary", "session_id": self.session_id, "label": self.session_label, "latency_s": time.time() - start, "status": status, "retries": retries})
        if status != 200:
            self.logger.log({"type": "error", "session_id": self.session_id, "label": self.session_label, "stage": "summary", "status": status, "detail": obj})

        # Billing
        start = time.time()
        status, obj, retries = self._post_json_with_retry(f"/api/session/{self.session_id}/billing/bill")
        self.logger.log({"type": "billing", "session_id": self.session_id, "label": self.session_label, "latency_s": time.time() - start, "status": status, "retries": retries})
        if status != 200:
            self.logger.log({"type": "error", "session_id": self.session_id, "label": self.session_label, "stage": "billing", "status": status, "detail": obj})

        # SOAP
        start = time.time()
        status, obj, retries = self._post_json_with_retry(f"/api/session/{self.session_id}/make_soap")
        self.logger.log({"type": "soap", "session_id": self.session_id, "label": self.session_label, "latency_s": time.time() - start, "status": status, "retries": retries})
        if status != 200:
            self.logger.log({"type": "error", "session_id": self.session_id, "label": self.session_label, "stage": "soap", "status": status, "detail": obj})

        if self.do_regen:
            start = time.time()
            status, obj, retries = self._post_json_with_retry(f"/api/session/{self.session_id}/make_soap")
            self.logger.log({"type": "soap_regen", "session_id": self.session_id, "label": self.session_label, "latency_s": time.time() - start, "status": status, "retries": retries})
            if status != 200:
                self.logger.log({"type": "error", "session_id": self.session_id, "label": self.session_label, "stage": "soap_regen", "status": status, "detail": obj})

    def _post_json_with_retry(self, path: str) -> Tuple[int, dict, int]:
        attempts = 0
        backoff = 1.0
        last_status = 0
        last_obj: dict = {}
        while attempts < 3:
            status, obj = self.client.post_json(path, {})
            last_status = status
            last_obj = obj
            if status == 200:
                return status, obj, attempts
            attempts += 1
            time.sleep(backoff)
            backoff = min(backoff * 2, 8.0)
        return last_status, last_obj, attempts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--username", default=DEFAULT_USER)
    parser.add_argument("--password", default=DEFAULT_PASS)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--max-queue", type=int, default=10)
    parser.add_argument("--server-pid", type=int, default=0)
    args = parser.parse_args()

    audio_dir = os.path.join(os.path.dirname(__file__), "audio")
    audio_files = collect_audio_files(audio_dir)
    if len(audio_files) < args.concurrency:
        raise SystemExit(f"Not enough audio files in {audio_dir}. Need {args.concurrency}, found {len(audio_files)}.")

    client = ApiClient(args.base_url)
    if not args.username or not args.password:
        raise SystemExit("Set CENTAUR_USERNAME and CENTAUR_PASSWORD (or pass --username/--password).")
    client.login(args.username, args.password)

    ts = time.strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)
    log_path = os.path.join(results_dir, f"run_{ts}.jsonl")
    logger = EventLogger(log_path)

    server_pid = args.server_pid or find_uvicorn_pid()
    monitor = SystemMonitor(logger, server_pid=server_pid, interval_s=5.0)
    monitor.start()

    runners: List[SessionRunner] = []
    threads: List[threading.Thread] = []

    random.shuffle(audio_files)

    for i in range(args.concurrency):
        start_delay = i * 30
        do_coach = (i % 2 == 0)
        do_regen = (i % 4 == 0)
        runner = SessionRunner(
            client=client,
            logger=logger,
            audio_path=audio_files[i],
            start_delay_s=start_delay,
            max_queue=args.max_queue,
            do_coach=do_coach,
            do_regen=do_regen,
            session_label=f"session_{i+1}",
        )
        runners.append(runner)
        t = threading.Thread(target=runner.run)
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    monitor.stop()

    events = read_jsonl(log_path)
    report_path = os.path.join(os.path.dirname(__file__), "report.md")
    _write_report(report_path, events)

    print(f"Run complete. Log: {log_path}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
