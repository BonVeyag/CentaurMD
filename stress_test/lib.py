import base64
import io
import json
import os
import queue
import random
import ssl
import subprocess
import threading
import time
import uuid
import wave
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple
from urllib.parse import urlparse
import http.client


@dataclass
class WavSegment:
    index: int
    duration_s: float
    data: bytes


def collect_audio_files(audio_dir: str) -> List[str]:
    paths: List[str] = []
    if not os.path.isdir(audio_dir):
        return paths
    for name in sorted(os.listdir(audio_dir)):
        if not name.lower().endswith(".wav"):
            continue
        paths.append(os.path.join(audio_dir, name))
    return paths


def _write_wav_bytes(
    frames: bytes,
    nchannels: int,
    sampwidth: int,
    framerate: int,
) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(nchannels)
        wf.setsampwidth(sampwidth)
        wf.setframerate(framerate)
        wf.writeframes(frames)
    return buf.getvalue()


def iter_wav_segments(
    path: str,
    min_sec: float = 6.0,
    max_sec: float = 12.0,
    max_total_sec: Optional[float] = None,
) -> Iterable[WavSegment]:
    with wave.open(path, "rb") as wf:
        nchannels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        total_frames = wf.getnframes()
        max_frames = total_frames
        if max_total_sec is not None:
            max_frames = min(total_frames, int(max_total_sec * framerate))

        index = 0
        frames_read = 0
        while frames_read < max_frames:
            seg_sec = random.uniform(min_sec, max_sec)
            seg_frames = min(int(seg_sec * framerate), max_frames - frames_read)
            if seg_frames <= 0:
                break
            frames = wf.readframes(seg_frames)
            frames_read += seg_frames
            duration_s = seg_frames / float(framerate)
            data = _write_wav_bytes(frames, nchannels, sampwidth, framerate)
            yield WavSegment(index=index, duration_s=duration_s, data=data)
            index += 1


def _multipart_encode_file(field_name: str, filename: str, content_type: str, data: bytes) -> Tuple[str, bytes]:
    boundary = uuid.uuid4().hex
    header = (
        f"--{boundary}\r\n"
        f"Content-Disposition: form-data; name=\"{field_name}\"; filename=\"{filename}\"\r\n"
        f"Content-Type: {content_type}\r\n\r\n"
    ).encode("utf-8")
    footer = f"\r\n--{boundary}--\r\n".encode("utf-8")
    body = header + data + footer
    return boundary, body


class ApiClient:
    def __init__(self, base_url: str, timeout: int = 60):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._token: Optional[str] = None

        parsed = urlparse(self.base_url)
        self.scheme = parsed.scheme or "http"
        self.host = parsed.hostname or "127.0.0.1"
        self.port = parsed.port or (443 if self.scheme == "https" else 80)

    def _connection(self) -> http.client.HTTPConnection:
        if self.scheme == "https":
            context = ssl.create_default_context()
            return http.client.HTTPSConnection(self.host, self.port, timeout=self.timeout, context=context)
        return http.client.HTTPConnection(self.host, self.port, timeout=self.timeout)

    def set_token(self, token: str) -> None:
        self._token = token

    def _auth_header(self) -> Dict[str, str]:
        if not self._token:
            return {}
        return {"Authorization": f"Bearer {self._token}"}

    def post_json(self, path: str, payload: Optional[dict] = None) -> Tuple[int, dict]:
        body = json.dumps(payload or {}).encode("utf-8")
        headers = {"Content-Type": "application/json", **self._auth_header()}
        conn = self._connection()
        try:
            conn.request("POST", path, body=body, headers=headers)
            resp = conn.getresponse()
            data = resp.read()
        finally:
            conn.close()
        try:
            obj = json.loads(data.decode("utf-8")) if data else {}
        except Exception:
            obj = {"raw": data.decode("utf-8", errors="ignore")}
        return resp.status, obj

    def post_multipart(self, path: str, field_name: str, filename: str, content_type: str, data: bytes) -> Tuple[int, dict]:
        boundary, body = _multipart_encode_file(field_name, filename, content_type, data)
        headers = {
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "Content-Length": str(len(body)),
            **self._auth_header(),
        }
        conn = self._connection()
        try:
            conn.request("POST", path, body=body, headers=headers)
            resp = conn.getresponse()
            raw = resp.read()
        finally:
            conn.close()
        try:
            obj = json.loads(raw.decode("utf-8")) if raw else {}
        except Exception:
            obj = {"raw": raw.decode("utf-8", errors="ignore")}
        return resp.status, obj

    def login(self, username: str, password: str) -> str:
        status, obj = self.post_json("/api/auth/login", {"username": username, "password": password})
        if status != 200:
            raise RuntimeError(f"Login failed ({status}): {obj}")
        token = obj.get("token") or ""
        if not token:
            raise RuntimeError(f"Login response missing token: {obj}")
        self.set_token(token)
        return token

    def create_session(self) -> str:
        status, obj = self.post_json("/api/session/create", {})
        if status != 200:
            raise RuntimeError(f"Session create failed ({status}): {obj}")
        sid = obj.get("session_id")
        if not sid:
            raise RuntimeError(f"Session create missing session_id: {obj}")
        return sid


class EventLogger:
    def __init__(self, path: str):
        self.path = path
        self._lock = threading.Lock()
        os.makedirs(os.path.dirname(path), exist_ok=True)

    def log(self, event: dict) -> None:
        event = dict(event)
        event["ts"] = time.time()
        line = json.dumps(event, ensure_ascii=True)
        with self._lock:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(line + "\n")


def _parse_ps_cpu_mem(pid: int) -> Tuple[Optional[float], Optional[float]]:
    try:
        out = subprocess.check_output(["ps", "-o", "%cpu,%mem", "-p", str(pid)], text=True)
        lines = [ln.strip() for ln in out.splitlines() if ln.strip()]
        if len(lines) < 2:
            return None, None
        cpu_str, mem_str = lines[1].split()
        return float(cpu_str), float(mem_str)
    except Exception:
        return None, None


def _system_mem_bytes() -> Tuple[Optional[int], Optional[int]]:
    try:
        total = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip())
    except Exception:
        return None, None
    try:
        vm = subprocess.check_output(["vm_stat"], text=True)
        page_size = 4096
        free = 0
        speculative = 0
        for line in vm.splitlines():
            line = line.strip().lower()
            if line.startswith("pages free"):
                free = int(line.split(":")[1].strip().strip(".")) * page_size
            if line.startswith("pages speculative"):
                speculative = int(line.split(":")[1].strip().strip(".")) * page_size
        used = total - (free + speculative)
        return total, used
    except Exception:
        return total, None


class SystemMonitor:
    def __init__(self, logger: EventLogger, server_pid: Optional[int], interval_s: float = 5.0):
        self.logger = logger
        self.server_pid = server_pid
        self.interval_s = interval_s
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._psutil = None
        try:
            import psutil  # type: ignore
            self._psutil = psutil
        except Exception:
            self._psutil = None

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5)

    def _run(self) -> None:
        while not self._stop.is_set():
            payload = {"type": "system"}
            if self.server_pid:
                if self._psutil:
                    try:
                        proc = self._psutil.Process(self.server_pid)
                        cpu = proc.cpu_percent(interval=None)
                        mem = proc.memory_info().rss
                        payload.update({"server_cpu_percent": cpu, "server_mem_bytes": mem})
                    except Exception:
                        pass
                else:
                    cpu, mem = _parse_ps_cpu_mem(self.server_pid)
                    if cpu is not None:
                        payload["server_cpu_percent"] = cpu
                    if mem is not None:
                        payload["server_mem_percent"] = mem
            if self._psutil:
                try:
                    payload["sys_cpu_percent"] = self._psutil.cpu_percent(interval=None)
                    payload["sys_mem_percent"] = self._psutil.virtual_memory().percent
                except Exception:
                    pass
            else:
                try:
                    load1, load5, load15 = os.getloadavg()
                    payload["sys_loadavg"] = [load1, load5, load15]
                except Exception:
                    pass
                total, used = _system_mem_bytes()
                if total is not None:
                    payload["sys_mem_total_bytes"] = total
                if used is not None:
                    payload["sys_mem_used_bytes"] = used
            self.logger.log(payload)
            self._stop.wait(self.interval_s)


def find_uvicorn_pid() -> Optional[int]:
    try:
        out = subprocess.check_output(["ps", "-ax", "-o", "pid=,command="], text=True)
    except Exception:
        return None
    for line in out.splitlines():
        if "uvicorn" in line and "main:app" in line:
            parts = line.strip().split(None, 1)
            try:
                return int(parts[0])
            except Exception:
                continue
    return None


def read_jsonl(path: str) -> List[dict]:
    items: List[dict] = []
    if not os.path.exists(path):
        return items
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except Exception:
                continue
    return items
