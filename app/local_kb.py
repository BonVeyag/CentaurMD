from __future__ import annotations

import hashlib
import html
import json
import logging
import mimetypes
import os
import re
import sqlite3
import threading
import time
import urllib.parse
import urllib.request
import urllib.robotparser
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from html.parser import HTMLParser
from typing import Any, Dict, Iterable, List, Optional, Tuple


logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
DB_PATH = os.path.join(DATA_DIR, "local_kb.sqlite")

KB_ENABLED = os.getenv("CENTAUR_KB_ENABLED", "1").strip().lower() not in {"0", "false", "no"}
KB_MAX_PAGES = int(os.getenv("CENTAUR_KB_MAX_PAGES", "25"))
KB_MAX_DEPTH = int(os.getenv("CENTAUR_KB_MAX_DEPTH", "2"))
KB_MIN_TEXT_LEN = int(os.getenv("CENTAUR_KB_MIN_TEXT_LEN", "300"))
KB_MAX_TEXT_LEN = int(os.getenv("CENTAUR_KB_MAX_TEXT_LEN", "20000"))
KB_CHUNK_CHARS = int(os.getenv("CENTAUR_KB_CHUNK_CHARS", "1200"))
KB_CHUNK_OVERLAP = int(os.getenv("CENTAUR_KB_CHUNK_OVERLAP", "150"))
KB_REFRESH_DAYS = int(os.getenv("CENTAUR_KB_REFRESH_DAYS", "30"))
KB_REFRESH_INTERVAL_SECONDS = int(os.getenv("CENTAUR_KB_REFRESH_INTERVAL_SECONDS", "86400"))
KB_ASSET_DIR = os.path.join(DATA_DIR, "kb_assets")
KB_MAX_ASSETS = int(os.getenv("CENTAUR_KB_MAX_ASSETS", "60"))
KB_GUIDELINE_MAX_NODES = int(os.getenv("CENTAUR_KB_GUIDELINE_MAX_NODES", "120"))
KB_GUIDELINE_MAX_EDGES = int(os.getenv("CENTAUR_KB_GUIDELINE_MAX_EDGES", "180"))
KB_GUIDELINE_VISION_MODEL = os.getenv("CENTAUR_KB_VISION_MODEL", "gpt-4o-mini").strip()
KB_GUIDELINE_LLM_EXTRACT = os.getenv("CENTAUR_KB_ENABLE_VISION", "0").strip().lower() in {"1", "true", "yes"}

_INDEX_LOCK = threading.Lock()
_REFRESH_THREAD: Optional[threading.Thread] = None


@dataclass
class KbSite:
    id: int
    url: str
    domain: str
    created_at_utc: str
    last_indexed_utc: str
    last_status: str
    last_error: str


@dataclass
class KbPage:
    url: str
    title: str
    text: str
    links: List[str]
    assets: List[str]
    inline_svgs: List[str]


class _HtmlExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._texts: List[str] = []
        self._links: List[str] = []
        self._images: List[str] = []
        self._embeds: List[str] = []
        self._title: str = ""
        self._in_script = False
        self._in_style = False
        self._in_title = False

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, str]]) -> None:
        t = (tag or "").lower()
        if t == "script":
            self._in_script = True
        elif t == "style":
            self._in_style = True
        elif t == "title":
            self._in_title = True
        elif t == "a":
            for k, v in attrs:
                if (k or "").lower() == "href" and v:
                    self._links.append(v)
        elif t == "img":
            for k, v in attrs:
                if (k or "").lower() == "src" and v:
                    self._images.append(v)
        elif t in {"embed", "object"}:
            for k, v in attrs:
                if (k or "").lower() in {"src", "data"} and v:
                    self._embeds.append(v)

    def handle_endtag(self, tag: str) -> None:
        t = (tag or "").lower()
        if t == "script":
            self._in_script = False
        elif t == "style":
            self._in_style = False
        elif t == "title":
            self._in_title = False

    def handle_data(self, data: str) -> None:
        if self._in_script or self._in_style:
            return
        if self._in_title:
            self._title += data
            return
        if data and data.strip():
            self._texts.append(data.strip())

    def extract(self) -> Tuple[str, str, List[str], List[str], List[str]]:
        text = " ".join(self._texts)
        text = html.unescape(text)
        text = re.sub(r"\s+", " ", text).strip()
        title = html.unescape(self._title or "").strip()
        return title, text, list(self._links), list(self._images), list(self._embeds)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ensure_dirs() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(KB_ASSET_DIR, exist_ok=True)


def _get_db() -> sqlite3.Connection:
    _ensure_dirs()
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with _get_db() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS kb_sites (
                id INTEGER PRIMARY KEY,
                url TEXT UNIQUE,
                domain TEXT,
                created_at_utc TEXT,
                last_indexed_utc TEXT,
                last_status TEXT,
                last_error TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS kb_chunks (
                id INTEGER PRIMARY KEY,
                site_id INTEGER,
                url TEXT,
                title TEXT,
                content TEXT,
                created_at_utc TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS kb_chunks_fts USING fts5(
                content,
                title,
                url,
                site_id
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS kb_assets (
                id INTEGER PRIMARY KEY,
                site_url TEXT,
                asset_url TEXT,
                asset_type TEXT,
                sha256 TEXT,
                fetched_at_utc TEXT,
                updated_at_utc TEXT,
                status TEXT,
                last_error TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS kb_guidelines (
                id INTEGER PRIMARY KEY,
                site_url TEXT,
                guideline_id TEXT UNIQUE,
                title TEXT,
                jurisdiction TEXT,
                version_date TEXT,
                source_url TEXT,
                created_at_utc TEXT,
                updated_at_utc TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS kb_guideline_graphs (
                id INTEGER PRIMARY KEY,
                guideline_id TEXT,
                graph_json TEXT,
                extraction_method TEXT,
                extraction_confidence REAL,
                created_at_utc TEXT,
                updated_at_utc TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS kb_guideline_graph_patches (
                id INTEGER PRIMARY KEY,
                guideline_id TEXT,
                patch_json TEXT,
                created_at_utc TEXT,
                updated_at_utc TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS kb_guideline_graph_index USING fts5(
                guideline_id,
                text,
                metadata_json
            )
            """
        )
        conn.commit()


def _normalize_url(raw: str) -> str:
    url = (raw or "").strip()
    if not url:
        return ""
    if not re.match(r"^https?://", url, flags=re.IGNORECASE):
        url = "https://" + url
    parsed = urllib.parse.urlparse(url)
    if not parsed.netloc:
        return ""
    norm = parsed._replace(fragment="").geturl()
    return norm.rstrip("/")


def _is_allowed_url(url: str) -> bool:
    if not url:
        return False
    if not re.match(r"^https?://", url, flags=re.IGNORECASE):
        return False
    if re.search(r"\.(pdf|jpg|jpeg|png|gif|svg|zip|doc|docx|ppt|pptx)$", url, flags=re.IGNORECASE):
        return False
    return True


def _is_asset_url(url: str) -> bool:
    if not url:
        return False
    return bool(re.search(r"\.(pdf|jpg|jpeg|png|gif|svg)$", url, flags=re.IGNORECASE))


def _extract_inline_svgs(html_text: str) -> List[str]:
    if not html_text:
        return []
    return re.findall(r"<svg[^>]*>.*?</svg>", html_text, flags=re.IGNORECASE | re.DOTALL)


def _guess_asset_type(url: str, content_type: str = "") -> str:
    ctype = (content_type or "").lower()
    if "svg" in ctype:
        return "svg"
    if "pdf" in ctype:
        return "pdf"
    if "image" in ctype:
        return "image"
    ext = (os.path.splitext(urllib.parse.urlparse(url).path)[-1] or "").lower()
    if ext in {".svg"}:
        return "svg"
    if ext in {".pdf"}:
        return "pdf"
    if ext in {".png", ".jpg", ".jpeg", ".gif"}:
        return "image"
    return ""


def _split_chunks(text: str) -> List[str]:
    if not text:
        return []
    if len(text) <= KB_CHUNK_CHARS:
        return [text]
    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + KB_CHUNK_CHARS)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(0, end - KB_CHUNK_OVERLAP)
    return chunks


def _fetch_html(url: str) -> Tuple[str, str]:
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "CentaurMD/1.0"},
        method="GET",
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        content_type = (resp.headers.get("Content-Type") or "").lower()
        if "text/html" not in content_type:
            return "", ""
        raw = resp.read()
        charset = "utf-8"
        m = re.search(r"charset=([a-z0-9\-]+)", content_type)
        if m:
            charset = m.group(1)
        try:
            html_text = raw.decode(charset, errors="ignore")
        except Exception:
            html_text = raw.decode("utf-8", errors="ignore")
        return html_text, content_type


def _get_robot_parser(start_url: str) -> urllib.robotparser.RobotFileParser:
    parsed = urllib.parse.urlparse(start_url)
    robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
    rp = urllib.robotparser.RobotFileParser()
    try:
        rp.set_url(robots_url)
        rp.read()
    except Exception:
        pass
    return rp


def _crawl_site(start_url: str, max_pages: int, max_depth: int) -> List[Tuple[str, str, str]]:
    parsed = urllib.parse.urlparse(start_url)
    root_domain = parsed.netloc
    rp = _get_robot_parser(start_url)

    queue = deque([(start_url, 0)])
    visited: set[str] = set()
    pages: List[Tuple[str, str, str]] = []

    while queue and len(pages) < max_pages:
        url, depth = queue.popleft()
        url = _normalize_url(url)
        if not url or url in visited:
            continue
        if depth > max_depth:
            continue
        if not _is_allowed_url(url):
            continue
        if urllib.parse.urlparse(url).netloc != root_domain:
            continue
        if rp and not rp.can_fetch("CentaurMD/1.0", url):
            visited.add(url)
            continue

        visited.add(url)
        try:
            html_text, _ = _fetch_html(url)
        except Exception as e:
            logger.info(f"KB fetch failed: {url} ({e})")
            continue

        if not html_text:
            continue

        parser = _HtmlExtractor()
        try:
            parser.feed(html_text)
        except Exception:
            continue
        title, text, links = parser.extract()

        if len(text) < KB_MIN_TEXT_LEN:
            continue

        text = text[:KB_MAX_TEXT_LEN].strip()
        pages.append((url, title, text))

        for href in links:
            if not href:
                continue
            if href.startswith("#"):
                continue
            abs_url = urllib.parse.urljoin(url, href)
            abs_url = _normalize_url(abs_url)
            if not abs_url or abs_url in visited:
                continue
            if urllib.parse.urlparse(abs_url).netloc != root_domain:
                continue
            queue.append((abs_url, depth + 1))

    return pages


def _upsert_site(conn: sqlite3.Connection, url: str, domain: str) -> int:
    now = _utc_now_iso()
    cur = conn.execute("SELECT id FROM kb_sites WHERE url = ?", (url,))
    row = cur.fetchone()
    if row:
        return int(row["id"])
    conn.execute(
        "INSERT INTO kb_sites (url, domain, created_at_utc, last_indexed_utc, last_status, last_error) VALUES (?, ?, ?, ?, ?, ?)",
        (url, domain, now, "", "new", ""),
    )
    conn.commit()
    cur = conn.execute("SELECT id FROM kb_sites WHERE url = ?", (url,))
    row = cur.fetchone()
    return int(row["id"])


def index_site(url: str) -> Dict[str, str]:
    if not KB_ENABLED:
        raise RuntimeError("Local knowledge base is disabled.")

    normalized = _normalize_url(url)
    if not normalized:
        raise ValueError("Invalid URL.")

    with _INDEX_LOCK:
        init_db()
        parsed = urllib.parse.urlparse(normalized)
        domain = parsed.netloc
        pages: List[Tuple[str, str, str]] = []
        error = ""
        status = "ok"
        now = _utc_now_iso()

        with _get_db() as conn:
            site_id = _upsert_site(conn, normalized, domain)
            conn.execute("DELETE FROM kb_chunks WHERE site_id = ?", (site_id,))
            conn.execute("DELETE FROM kb_chunks_fts WHERE site_id = ?", (site_id,))
            conn.commit()

        try:
            pages = _crawl_site(normalized, KB_MAX_PAGES, KB_MAX_DEPTH)
        except Exception as e:
            status = "error"
            error = str(e)
            pages = []

        chunk_count = 0
        with _get_db() as conn:
            for page_url, title, text in pages:
                for chunk in _split_chunks(text):
                    chunk_count += 1
                    created = _utc_now_iso()
                    cur = conn.execute(
                        "INSERT INTO kb_chunks (site_id, url, title, content, created_at_utc) VALUES (?, ?, ?, ?, ?)",
                        (site_id, page_url, title, chunk, created),
                    )
                    chunk_id = cur.lastrowid
                    conn.execute(
                        "INSERT INTO kb_chunks_fts (rowid, content, title, url, site_id) VALUES (?, ?, ?, ?, ?)",
                        (chunk_id, chunk, title, page_url, site_id),
                    )

            conn.execute(
                "UPDATE kb_sites SET last_indexed_utc = ?, last_status = ?, last_error = ? WHERE id = ?",
                (now, status, error, site_id),
            )
            conn.commit()

        return {
            "url": normalized,
            "domain": domain,
            "last_indexed_utc": now,
            "last_status": status,
            "last_error": error,
            "pages_indexed": str(len(pages)),
            "chunks_indexed": str(chunk_count),
        }


def list_sites() -> List[Dict[str, str]]:
    if not os.path.exists(DB_PATH):
        return []
    init_db()
    with _get_db() as conn:
        cur = conn.execute(
            "SELECT id, url, domain, created_at_utc, last_indexed_utc, last_status, last_error FROM kb_sites ORDER BY url"
        )
        rows = cur.fetchall()
    out: List[Dict[str, str]] = []
    for row in rows:
        out.append(
            {
                "id": str(row["id"]),
                "url": row["url"] or "",
                "domain": row["domain"] or "",
                "created_at_utc": row["created_at_utc"] or "",
                "last_indexed_utc": row["last_indexed_utc"] or "",
                "last_status": row["last_status"] or "",
                "last_error": row["last_error"] or "",
            }
        )
    return out


def _sanitize_query(query: str) -> str:
    q = (query or "").lower()
    q = re.sub(r"[^a-z0-9\s]", " ", q)
    q = re.sub(r"\s+", " ", q).strip()
    return q


def search_kb(query: str, limit: int = 5) -> List[Dict[str, str]]:
    if not KB_ENABLED:
        return []
    if not os.path.exists(DB_PATH):
        return []
    init_db()
    q = _sanitize_query(query)
    if not q:
        return []
    tokens = [t for t in q.split(" ") if t]
    if not tokens:
        return []
    match = " OR ".join([f"{t}*" for t in tokens[:6]])
    try:
        with _get_db() as conn:
            cur = conn.execute(
                "SELECT url, title, content, bm25(kb_chunks_fts) AS score FROM kb_chunks_fts WHERE kb_chunks_fts MATCH ? ORDER BY score LIMIT ?",
                (match, int(limit)),
            )
            rows = cur.fetchall()
    except Exception:
        return []

    results: List[Dict[str, str]] = []
    for row in rows:
        content = (row["content"] or "").strip()
        results.append(
            {
                "url": row["url"] or "",
                "title": row["title"] or "",
                "content": content[:900],
            }
        )
    return results


def format_kb_context(results: Iterable[Dict[str, str]]) -> str:
    blocks: List[str] = []
    for idx, rec in enumerate(results):
        url = (rec.get("url") or "").strip()
        title = (rec.get("title") or "").strip()
        content = (rec.get("content") or "").strip()
        if not content:
            continue
        label = f"{title} ({url})" if title else url
        blocks.append(f"{idx + 1}) {label}\n{content}")
    return "\n\n".join(blocks).strip()


def refresh_due_sites() -> None:
    if not KB_ENABLED:
        return
    init_db()
    cutoff = time.time() - (KB_REFRESH_DAYS * 86400)
    due: List[str] = []
    with _get_db() as conn:
        cur = conn.execute("SELECT url, last_indexed_utc FROM kb_sites")
        for row in cur.fetchall():
            last = row["last_indexed_utc"] or ""
            if not last:
                due.append(row["url"])
                continue
            try:
                ts = datetime.fromisoformat(last.replace("Z", "+00:00")).timestamp()
            except Exception:
                ts = 0
            if ts < cutoff:
                due.append(row["url"])

    for url in due:
        try:
            logger.info(f"KB refresh: {url}")
            index_site(url)
        except Exception as e:
            logger.warning(f"KB refresh failed: {url} ({e})")


def start_refresh_thread() -> None:
    global _REFRESH_THREAD
    if _REFRESH_THREAD or not KB_ENABLED:
        return

    def _runner() -> None:
        while True:
            try:
                refresh_due_sites()
            except Exception as e:
                logger.warning(f"KB refresh loop error: {e}")
            time.sleep(max(3600, KB_REFRESH_INTERVAL_SECONDS))

    t = threading.Thread(target=_runner, daemon=True)
    t.start()
    _REFRESH_THREAD = t
