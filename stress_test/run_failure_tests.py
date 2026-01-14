import argparse
import os
import random
import signal
import threading
import time
from typing import List, Optional

from stress_test.lib import (
    ApiClient,
    EventLogger,
    SystemMonitor,
    collect_audio_files,
    find_uvicorn_pid,
    iter_wav_segments,
    read_jsonl,
)


DEFAULT_BASE_URL = os.getenv("CENTAUR_BASE_URL", "http://127.0.0.1:8000")
DEFAULT_USER = os.getenv("CENTAUR_USERNAME", "")
DEFAULT_PASS = os.getenv("CENTAUR_PASSWORD", "")


def _find_child_pid(ppid: int) -> Optional[int]:
    try:
        out = os.popen("ps -ax -o pid=,ppid=,command=").read()
    except Exception:
        return None
    for line in out.splitlines():
        parts = line.strip().split(None, 2)
        if len(parts) < 3:
            continue
        try:
            pid = int(parts[0])
            parent = int(parts[1])
        except Exception:
            continue
        if parent == ppid:
            return pid
    return None


def _burst_regen(client: ApiClient, logger: EventLogger, audio_path: str, session_count: int) -> None:
    sessions: List[str] = []
    for i in range(session_count):
        sid = client.create_session()
        sessions.append(sid)
        logger.log({"type": "session_start", "session_id": sid, "label": f"burst_{i+1}"})

    for sid in sessions:
        for seg in iter_wav_segments(audio_path, min_sec=6.0, max_sec=6.0, max_total_sec=30.0):
            start = time.time()
            status, obj = client.post_multipart(
                f"/api/transcribe_chunk?session_id={sid}",
                field_name="file",
                filename=f"chunk_{seg.index}.wav",
                content_type="audio/wav",
                data=seg.data,
            )
            logger.log({"type": "chunk", "session_id": sid, "label": "burst", "duration_s": seg.duration_s, "latency_s": time.time() - start, "status": status})
            if status != 200:
                logger.log({"type": "error", "session_id": sid, "label": "burst", "stage": "transcribe", "status": status, "detail": obj})
            time.sleep(seg.duration_s)

    def _soap_call(sid: str) -> None:
        start = time.time()
        status, obj = client.post_json(f"/api/session/{sid}/make_soap", {})
        logger.log({"type": "soap", "session_id": sid, "label": "burst", "latency_s": time.time() - start, "status": status})
        if status != 200:
            logger.log({"type": "error", "session_id": sid, "label": "burst", "stage": "soap", "status": status, "detail": obj})

    threads = [threading.Thread(target=_soap_call, args=(sid,)) for sid in sessions]
    for t in threads:
        t.start()
    for t in threads:
        t.join()


def _partial_failure(client: ApiClient, logger: EventLogger, audio_path: str, session_count: int, server_pid: Optional[int]) -> None:
    if not server_pid:
        logger.log({"type": "skip", "stage": "partial_failure", "reason": "server_pid_missing"})
        return

    child_pid = _find_child_pid(server_pid)
    if not child_pid:
        logger.log({"type": "skip", "stage": "partial_failure", "reason": "worker_pid_missing"})
        return

    sessions: List[str] = []
    for i in range(session_count):
        sid = client.create_session()
        sessions.append(sid)
        logger.log({"type": "session_start", "session_id": sid, "label": f"partial_{i+1}"})

    stop_event = threading.Event()

    def _pause_worker():
        time.sleep(10)
        try:
            os.kill(child_pid, signal.SIGSTOP)
            logger.log({"type": "worker_stop", "pid": child_pid})
            time.sleep(5)
            os.kill(child_pid, signal.SIGCONT)
            logger.log({"type": "worker_resume", "pid": child_pid})
        except Exception as e:
            logger.log({"type": "error", "stage": "worker_pause", "detail": str(e)})
        stop_event.set()

    pause_thread = threading.Thread(target=_pause_worker, daemon=True)
    pause_thread.start()

    def _run_session(sid: str):
        for seg in iter_wav_segments(audio_path, min_sec=6.0, max_sec=6.0, max_total_sec=60.0):
            start = time.time()
            status, obj = client.post_multipart(
                f"/api/transcribe_chunk?session_id={sid}",
                field_name="file",
                filename=f"chunk_{seg.index}.wav",
                content_type="audio/wav",
                data=seg.data,
            )
            logger.log({"type": "chunk", "session_id": sid, "label": "partial", "duration_s": seg.duration_s, "latency_s": time.time() - start, "status": status})
            if status != 200:
                logger.log({"type": "error", "session_id": sid, "label": "partial", "stage": "transcribe", "status": status, "detail": obj})
            time.sleep(seg.duration_s)

    threads = [threading.Thread(target=_run_session, args=(sid,)) for sid in sessions]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    stop_event.wait(1)


def _network_blip(client: ApiClient, logger: EventLogger, audio_path: str, session_count: int) -> None:
    sessions: List[str] = []
    for i in range(session_count):
        sid = client.create_session()
        sessions.append(sid)
        logger.log({"type": "session_start", "session_id": sid, "label": f"blip_{i+1}"})

    down_client = ApiClient("http://127.0.0.1:9999")
    down_client.set_token(client._token or "")

    blip_start = time.time() + 10
    blip_end = blip_start + 30

    def _run_session(sid: str):
        for seg in iter_wav_segments(audio_path, min_sec=6.0, max_sec=6.0, max_total_sec=60.0):
            now = time.time()
            active_client = down_client if blip_start <= now <= blip_end else client
            start = time.time()
            try:
                status, obj = active_client.post_multipart(
                    f"/api/transcribe_chunk?session_id={sid}",
                    field_name="file",
                    filename=f"chunk_{seg.index}.wav",
                    content_type="audio/wav",
                    data=seg.data,
                )
            except Exception as e:
                status, obj = 0, {"error": str(e)}
            logger.log({"type": "chunk", "session_id": sid, "label": "blip", "duration_s": seg.duration_s, "latency_s": time.time() - start, "status": status})
            if status != 200:
                logger.log({"type": "error", "session_id": sid, "label": "blip", "stage": "transcribe", "status": status, "detail": obj})
            time.sleep(seg.duration_s)

    threads = [threading.Thread(target=_run_session, args=(sid,)) for sid in sessions]
    for t in threads:
        t.start()
    for t in threads:
        t.join()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--username", default=DEFAULT_USER)
    parser.add_argument("--password", default=DEFAULT_PASS)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--server-pid", type=int, default=0)
    args = parser.parse_args()

    audio_dir = os.path.join(os.path.dirname(__file__), "audio")
    audio_files = collect_audio_files(audio_dir)
    if not audio_files:
        raise SystemExit(f"No audio files in {audio_dir}.")

    client = ApiClient(args.base_url)
    if not args.username or not args.password:
        raise SystemExit("Set CENTAUR_USERNAME and CENTAUR_PASSWORD (or pass --username/--password).")
    client.login(args.username, args.password)

    ts = time.strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)
    log_path = os.path.join(results_dir, f"failure_{ts}.jsonl")
    logger = EventLogger(log_path)

    server_pid = args.server_pid or find_uvicorn_pid()
    monitor = SystemMonitor(logger, server_pid=server_pid, interval_s=5.0)
    monitor.start()

    audio_path = random.choice(audio_files)

    _burst_regen(client, logger, audio_path, args.concurrency)
    _partial_failure(client, logger, audio_path, args.concurrency, server_pid)
    _network_blip(client, logger, audio_path, args.concurrency)

    monitor.stop()

    print(f"Failure tests complete. Log: {log_path}")


if __name__ == "__main__":
    main()
