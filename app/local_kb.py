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


def _crawl_site(start_url: str, max_pages: int, max_depth: int) -> List[KbPage]:
    parsed = urllib.parse.urlparse(start_url)
    root_domain = parsed.netloc
    rp = _get_robot_parser(start_url)

    queue = deque([(start_url, 0)])
    visited: set[str] = set()
    pages: List[KbPage] = []

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
        title, text, links, images, embeds = parser.extract()
        inline_svgs = _extract_inline_svgs(html_text)
        assets: List[str] = []
        for raw in (links + images + embeds):
            if not raw:
                continue
            abs_url = urllib.parse.urljoin(url, raw)
            abs_url = _normalize_url(abs_url)
            if not abs_url:
                continue
            if urllib.parse.urlparse(abs_url).netloc != root_domain:
                continue
            if _is_asset_url(abs_url):
                assets.append(abs_url)

        if len(text) < KB_MIN_TEXT_LEN:
            continue

        text = text[:KB_MAX_TEXT_LEN].strip()
        pages.append(KbPage(url=url, title=title, text=text, links=links, assets=assets, inline_svgs=inline_svgs))

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


def _collect_asset_candidates(pages: List[KbPage], site_url: str) -> List[Dict[str, Any]]:
    assets: List[Dict[str, Any]] = []
    for page in pages:
        for idx, svg in enumerate(page.inline_svgs):
            if not svg.strip():
                continue
            assets.append(
                {
                    "site_url": site_url,
                    "asset_url": f"{page.url}#inline-svg-{idx + 1}",
                    "asset_type": "svg",
                    "inline_text": svg,
                    "page_title": page.title,
                    "page_url": page.url,
                }
            )
        for raw in page.assets:
            asset_type = _guess_asset_type(raw)
            if not asset_type:
                continue
            assets.append(
                {
                    "site_url": site_url,
                    "asset_url": raw,
                    "asset_type": asset_type,
                    "page_title": page.title,
                    "page_url": page.url,
                }
            )
    return assets[:KB_MAX_ASSETS]


def _asset_path(sha256_hex: str, ext: str) -> str:
    safe_ext = ext if ext.startswith(".") else f".{ext}"
    return os.path.join(KB_ASSET_DIR, f"{sha256_hex}{safe_ext}")


def _fetch_asset_bytes(url: str) -> Tuple[bytes, str]:
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "CentaurMD/1.0"},
        method="GET",
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        content_type = (resp.headers.get("Content-Type") or "").lower()
        data = resp.read()
    return data, content_type


def _upsert_asset(
    conn: sqlite3.Connection,
    site_url: str,
    asset_url: str,
    asset_type: str,
    sha256_hex: str,
    status: str,
    error: str,
) -> None:
    now = _utc_now_iso()
    cur = conn.execute(
        "SELECT id FROM kb_assets WHERE asset_url = ?",
        (asset_url,),
    )
    row = cur.fetchone()
    if row:
        conn.execute(
            "UPDATE kb_assets SET asset_type = ?, sha256 = ?, updated_at_utc = ?, status = ?, last_error = ? WHERE id = ?",
            (asset_type, sha256_hex, now, status, error, int(row["id"])),
        )
        return
    conn.execute(
        """
        INSERT INTO kb_assets (site_url, asset_url, asset_type, sha256, fetched_at_utc, updated_at_utc, status, last_error)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (site_url, asset_url, asset_type, sha256_hex, now, now, status, error),
    )


def _guideline_id(source_url: str, version_date: str) -> str:
    base = f"{source_url}|{version_date or ''}".encode("utf-8")
    return hashlib.sha256(base).hexdigest()[:16]


def _detect_version_date(text: str) -> str:
    if not text:
        return ""
    m = re.search(r"\b(20\d{2}[-/]\d{1,2}[-/]\d{1,2})\b", text)
    if m:
        return m.group(1)
    m = re.search(r"\b(20\d{2})\b", text)
    return m.group(1) if m else ""


def _detect_jurisdiction(text: str) -> str:
    t = (text or "").lower()
    if "alberta" in t or "ahs" in t:
        return "Alberta, Canada"
    if "canada" in t or "canadian" in t:
        return "Canada"
    return "Canada"


def _guess_node_type(label: str) -> str:
    t = (label or "").strip().lower()
    if not t:
        return "note"
    if "refer" in t:
        return "referral"
    if "order" in t or "test" in t or "investigation" in t or "labs" in t:
        return "investigation"
    if "if " in t or t.endswith("?") or t.startswith("is ") or t.startswith("does ") or t.startswith("any "):
        return "decision"
    if "start" in t or "treat" in t or "manage" in t or "give" in t or "consider" in t:
        return "action"
    if t.startswith("stop") or t.startswith("avoid"):
        return "action"
    return "note"


def _build_graph_base(
    guideline_id: str,
    title: str,
    jurisdiction: str,
    version_date: str,
    source_url: str,
) -> Dict[str, Any]:
    return {
        "guideline_id": guideline_id,
        "title": title,
        "jurisdiction": jurisdiction,
        "version_date": version_date,
        "source_url": source_url,
        "nodes": [],
        "edges": [],
        "variables": [],
    }


def _graph_from_blocks(
    blocks: List[Dict[str, Any]],
    guideline_id: str,
    title: str,
    jurisdiction: str,
    version_date: str,
    source_url: str,
    asset_url: str,
    asset_type: str,
) -> Dict[str, Any]:
    graph = _build_graph_base(guideline_id, title, jurisdiction, version_date, source_url)
    nodes = []
    edges = []
    for idx, block in enumerate(blocks[:KB_GUIDELINE_MAX_NODES]):
        label = (block.get("text") or "").strip()
        if len(label) < 3:
            continue
        node_id = f"node_{idx + 1}"
        node_type = _guess_node_type(label)
        evidence = {
            "asset_url": asset_url,
            "asset_type": asset_type,
            "page": block.get("page"),
            "bbox": block.get("bbox"),
            "excerpt": label[:200],
        }
        nodes.append(
            {
                "id": node_id,
                "type": node_type,
                "label": label[:400],
                "actions": [label[:400]] if node_type in {"action", "investigation", "referral"} else [],
                "logic": None,
                "evidence_spans": [evidence],
            }
        )
    for idx in range(len(nodes) - 1):
        if idx >= KB_GUIDELINE_MAX_EDGES:
            break
        edges.append(
            {
                "from": nodes[idx]["id"],
                "to": nodes[idx + 1]["id"],
                "condition_text": "",
                "condition_logic": None,
                "evidence_spans": nodes[idx + 1].get("evidence_spans", []),
            }
        )
    graph["nodes"] = nodes
    graph["edges"] = edges
    return graph


def _extract_pdf_blocks(data: bytes) -> Tuple[List[Dict[str, Any]], bool]:
    blocks: List[Dict[str, Any]] = []
    has_text = False
    try:
        import fitz  # type: ignore
        doc = fitz.open(stream=data, filetype="pdf")
        for page_index, page in enumerate(doc, start=1):
            for b in page.get_text("blocks"):
                text = (b[4] or "").strip()
                if not text:
                    continue
                has_text = True
                blocks.append(
                    {
                        "text": text,
                        "page": page_index,
                        "bbox": [float(b[0]), float(b[1]), float(b[2]), float(b[3])],
                    }
                )
        return blocks, has_text
    except Exception:
        pass
    try:
        from pypdf import PdfReader  # type: ignore
    except Exception:
        try:
            from PyPDF2 import PdfReader  # type: ignore
        except Exception:
            return blocks, False
    try:
        reader = PdfReader(data)
        for page_index, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if not text:
                continue
            has_text = True
            blocks.append(
                {
                    "text": text,
                    "page": page_index,
                    "bbox": None,
                }
            )
    except Exception:
        return blocks, False
    return blocks, has_text


def _extract_svg_blocks(svg_text: str) -> List[Dict[str, Any]]:
    blocks: List[Dict[str, Any]] = []
    if not svg_text:
        return blocks
    try:
        import xml.etree.ElementTree as ET
        root = ET.fromstring(svg_text)
        for elem in root.iter():
            if elem.tag.lower().endswith("text"):
                txt = "".join(elem.itertext()).strip()
                if not txt:
                    continue
                x = elem.attrib.get("x")
                y = elem.attrib.get("y")
                bbox = None
                if x is not None and y is not None:
                    try:
                        bbox = [float(x), float(y), float(x), float(y)]
                    except Exception:
                        bbox = None
                blocks.append({"text": txt, "page": None, "bbox": bbox})
    except Exception:
        pass
    return blocks


def _extract_html_blocks(text: str) -> List[Dict[str, Any]]:
    blocks: List[Dict[str, Any]] = []
    if not text:
        return blocks
    for line in re.split(r"(?:\n|\\n)", text):
        line = (line or "").strip()
        if len(line) < 6:
            continue
        blocks.append({"text": line, "page": None, "bbox": None})
    if not blocks:
        parts = re.split(r"\.\s+", text)
        for part in parts:
            part = part.strip()
            if len(part) < 12:
                continue
            blocks.append({"text": part, "page": None, "bbox": None})
    return blocks[:KB_GUIDELINE_MAX_NODES]


def _validate_guideline_graph(graph: Dict[str, Any]) -> Dict[str, Any]:
    required = {"guideline_id", "title", "jurisdiction", "version_date", "source_url", "nodes", "edges", "variables"}
    missing = required - set(graph.keys())
    if missing:
        raise ValueError(f"Guideline graph missing keys: {sorted(missing)}")
    if not isinstance(graph.get("nodes"), list) or not isinstance(graph.get("edges"), list):
        raise ValueError("Guideline graph nodes/edges must be lists.")
    return graph


def _flatten_graph_text(graph: Dict[str, Any]) -> str:
    parts: List[str] = []
    for node in graph.get("nodes", []):
        parts.append(str(node.get("label", "")))
        for action in node.get("actions", []) or []:
            parts.append(str(action))
    for edge in graph.get("edges", []):
        parts.append(str(edge.get("condition_text", "")))
    for var in graph.get("variables", []):
        parts.append(str(var.get("name", "")))
        for syn in var.get("synonyms", []) or []:
            parts.append(str(syn))
    text = " ".join([p for p in parts if p])
    return re.sub(r"\s+", " ", text).strip()


def _apply_patch_ops(graph: Dict[str, Any], patch_ops: List[Dict[str, Any]]) -> Dict[str, Any]:
    out = graph
    for op in patch_ops:
        if op.get("op") != "replace":
            continue
        path = op.get("path", "")
        if path in {"", "/"}:
            out = op.get("value", out)
            continue
        parts = [p for p in path.split("/") if p]
        cur = out
        for key in parts[:-1]:
            if isinstance(cur, list):
                key = int(key)
            cur = cur[key]
        last = parts[-1]
        if isinstance(cur, list):
            last = int(last)
        cur[last] = op.get("value")
    return out


def _store_guideline_graph(
    conn: sqlite3.Connection,
    guideline: Dict[str, Any],
    graph: Dict[str, Any],
    extraction_method: str,
    confidence: float,
) -> None:
    now = _utc_now_iso()
    guideline_id = guideline["guideline_id"]
    cur = conn.execute("SELECT id FROM kb_guidelines WHERE guideline_id = ?", (guideline_id,))
    row = cur.fetchone()
    if row:
        conn.execute(
            """
            UPDATE kb_guidelines
            SET title = ?, jurisdiction = ?, version_date = ?, source_url = ?, updated_at_utc = ?, site_url = ?
            WHERE guideline_id = ?
            """,
            (
                guideline.get("title", ""),
                guideline.get("jurisdiction", ""),
                guideline.get("version_date", ""),
                guideline.get("source_url", ""),
                now,
                guideline.get("site_url", ""),
                guideline_id,
            ),
        )
    else:
        conn.execute(
            """
            INSERT INTO kb_guidelines (site_url, guideline_id, title, jurisdiction, version_date, source_url, created_at_utc, updated_at_utc)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                guideline.get("site_url", ""),
                guideline_id,
                guideline.get("title", ""),
                guideline.get("jurisdiction", ""),
                guideline.get("version_date", ""),
                guideline.get("source_url", ""),
                now,
                now,
            ),
        )
    conn.execute("DELETE FROM kb_guideline_graphs WHERE guideline_id = ?", (guideline_id,))
    conn.execute(
        """
        INSERT INTO kb_guideline_graphs (guideline_id, graph_json, extraction_method, extraction_confidence, created_at_utc, updated_at_utc)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (guideline_id, json.dumps(graph), extraction_method, confidence, now, now),
    )
    conn.execute("DELETE FROM kb_guideline_graph_index WHERE guideline_id = ?", (guideline_id,))
    conn.execute(
        "INSERT INTO kb_guideline_graph_index (guideline_id, text, metadata_json) VALUES (?, ?, ?)",
        (
            guideline_id,
            _flatten_graph_text(graph),
            json.dumps(
                {
                    "title": guideline.get("title", ""),
                    "jurisdiction": guideline.get("jurisdiction", ""),
                    "version_date": guideline.get("version_date", ""),
                    "source_url": guideline.get("source_url", ""),
                    "extraction_method": extraction_method,
                    "confidence": confidence,
                }
            ),
        ),
    )


def _extract_guideline_graph_from_asset(
    asset: Dict[str, Any],
    site_url: str,
) -> Optional[Tuple[Dict[str, Any], str, float]]:
    asset_url = asset.get("asset_url", "")
    asset_type = asset.get("asset_type", "")
    inline_text = asset.get("inline_text", "")
    title = os.path.basename(urllib.parse.urlparse(asset_url).path) or asset_url
    version_date = _detect_version_date(title)
    jurisdiction = _detect_jurisdiction(title)
    guideline_id = _guideline_id(asset_url, version_date)
    guideline = {
        "guideline_id": guideline_id,
        "title": title,
        "jurisdiction": jurisdiction,
        "version_date": version_date,
        "source_url": asset_url,
        "site_url": site_url,
    }

    if asset_type == "svg":
        svg_text = inline_text
        if not svg_text:
            data = asset.get("bytes")
            if data:
                try:
                    svg_text = data.decode("utf-8", errors="ignore")
                except Exception:
                    svg_text = ""
        if not svg_text:
            return None
        blocks = _extract_svg_blocks(svg_text)
        graph = _graph_from_blocks(blocks, guideline_id, title, jurisdiction, version_date, asset_url, asset_url, "svg")
        return _validate_guideline_graph(graph), "svg", 0.7

    if asset_type == "pdf":
        data = asset.get("bytes")
        if not data:
            return None
        blocks, has_text = _extract_pdf_blocks(data)
        if has_text:
            graph = _graph_from_blocks(blocks, guideline_id, title, jurisdiction, version_date, asset_url, asset_url, "pdf")
            return _validate_guideline_graph(graph), "pdf_layout", 0.6
        if KB_GUIDELINE_LLM_EXTRACT:
            graph = _vision_graph_from_image(data, asset_url, "pdf")
            if graph:
                graph.update(
                    {
                        "guideline_id": guideline_id,
                        "title": title,
                        "jurisdiction": jurisdiction,
                        "version_date": version_date,
                        "source_url": asset_url,
                    }
                )
                graph.setdefault("variables", [])
                return _validate_guideline_graph(graph), "vision", 0.45
        return None

    if asset_type == "image":
        data = asset.get("bytes")
        if not data:
            return None
        if KB_GUIDELINE_LLM_EXTRACT:
            graph = _vision_graph_from_image(data, asset_url, "image")
            if graph:
                graph.update(
                    {
                        "guideline_id": guideline_id,
                        "title": title,
                        "jurisdiction": jurisdiction,
                        "version_date": version_date,
                        "source_url": asset_url,
                    }
                )
                graph.setdefault("variables", [])
                return _validate_guideline_graph(graph), "vision", 0.4
        return None
    return None


def _vision_graph_from_image(data: bytes, asset_url: str, asset_type: str) -> Optional[Dict[str, Any]]:
    try:
        from openai import OpenAI  # type: ignore
    except Exception:
        return None
    if not data:
        return None
    prompt = (
        "You are extracting a clinical guideline flowchart into structured JSON. "
        "Return JSON only matching GuidelineGraph schema with nodes and edges. "
        "Include evidence_spans with asset_url, asset_type, page (if known), bbox (approximate), excerpt. "
        "Keep labels concise and faithful."
    )
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    try:
        import base64
        b64 = base64.b64encode(data).decode("ascii")
        mime = "image/png" if asset_type == "image" else "application/pdf"
        resp = client.chat.completions.create(
            model=KB_GUIDELINE_VISION_MODEL,
            temperature=0.1,
            messages=[
                {"role": "system", "content": prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Extract the flowchart."},
                        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                    ],
                },
            ],
        )
    except Exception:
        return None
    try:
        content = resp.choices[0].message.content or ""
        graph = json.loads(content)
        return graph if isinstance(graph, dict) else None
    except Exception:
        return None


def _is_guideline_asset(asset: Dict[str, Any]) -> bool:
    url = (asset.get("asset_url") or "").lower()
    title = (asset.get("page_title") or "").lower()
    keywords = ["pathway", "guideline", "algorithm", "flow", "care map", "clinical pathway"]
    return any(k in url for k in keywords) or any(k in title for k in keywords)


def _index_guidelines_for_site(site_url: str, pages: List[KbPage]) -> None:
    assets = _collect_asset_candidates(pages, site_url)
    if not assets:
        return
    now = _utc_now_iso()
    with _get_db() as conn:
        for asset in assets:
            if not _is_guideline_asset(asset):
                continue
            asset_url = asset.get("asset_url", "")
            asset_type = asset.get("asset_type", "")
            inline_text = asset.get("inline_text", "")
            status = "ok"
            error = ""
            sha = ""
            data = b""
            if inline_text:
                data = inline_text.encode("utf-8")
                sha = hashlib.sha256(data).hexdigest()
            else:
                try:
                    data, content_type = _fetch_asset_bytes(asset_url)
                    asset_type = asset_type or _guess_asset_type(asset_url, content_type)
                    sha = hashlib.sha256(data).hexdigest()
                except Exception as e:
                    status = "error"
                    error = f"{e.__class__.__name__}: {e}"
            if sha:
                ext = ".bin"
                if asset_type == "pdf":
                    ext = ".pdf"
                elif asset_type == "svg":
                    ext = ".svg"
                elif asset_type == "image":
                    ext = os.path.splitext(urllib.parse.urlparse(asset_url).path)[-1] or ".png"
                path = _asset_path(sha, ext)
                if data and not os.path.exists(path):
                    try:
                        with open(path, "wb") as f:
                            f.write(data)
                    except Exception:
                        pass
            _upsert_asset(conn, site_url, asset_url, asset_type, sha, status, error)
            conn.commit()
            if status != "ok" or not data:
                continue
            asset["bytes"] = data
            graph_tuple = _extract_guideline_graph_from_asset(asset, site_url)
            if not graph_tuple:
                continue
            graph, method, confidence = graph_tuple
            guideline = {
                "guideline_id": graph.get("guideline_id"),
                "title": graph.get("title"),
                "jurisdiction": graph.get("jurisdiction"),
                "version_date": graph.get("version_date"),
                "source_url": graph.get("source_url"),
                "site_url": site_url,
            }
            try:
                _store_guideline_graph(conn, guideline, graph, method, confidence)
            except Exception as e:
                logger.warning(f"Guideline graph store failed: {asset_url} ({e})")
        conn.execute("UPDATE kb_assets SET updated_at_utc = ? WHERE site_url = ?", (now, site_url))
        conn.commit()


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
        pages: List[KbPage] = []
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
            for page in pages:
                page_url = page.url
                title = page.title
                text = page.text
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

        try:
            _index_guidelines_for_site(normalized, pages)
        except Exception as e:
            logger.warning(f"KB guideline extraction failed: {e}")

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


def list_guidelines() -> List[Dict[str, Any]]:
    if not os.path.exists(DB_PATH):
        return []
    init_db()
    with _get_db() as conn:
        cur = conn.execute(
            """
            SELECT g.guideline_id, g.title, g.jurisdiction, g.version_date, g.source_url,
                   gg.extraction_method, gg.extraction_confidence, gg.updated_at_utc
            FROM kb_guidelines g
            LEFT JOIN kb_guideline_graphs gg ON gg.guideline_id = g.guideline_id
            ORDER BY g.updated_at_utc DESC
            """
        )
        rows = cur.fetchall()
    out: List[Dict[str, Any]] = []
    for row in rows:
        out.append(
            {
                "guideline_id": row["guideline_id"] or "",
                "title": row["title"] or "",
                "jurisdiction": row["jurisdiction"] or "",
                "version_date": row["version_date"] or "",
                "source_url": row["source_url"] or "",
                "extraction_method": row["extraction_method"] or "",
                "confidence": float(row["extraction_confidence"] or 0),
                "updated_at_utc": row["updated_at_utc"] or "",
            }
        )
    return out


def _get_guideline_assets(conn: sqlite3.Connection, site_url: str) -> List[Dict[str, Any]]:
    cur = conn.execute(
        "SELECT asset_url, asset_type, sha256, updated_at_utc FROM kb_assets WHERE site_url = ? ORDER BY updated_at_utc DESC",
        (site_url,),
    )
    return [
        {
            "asset_url": row["asset_url"] or "",
            "asset_type": row["asset_type"] or "",
            "sha256": row["sha256"] or "",
            "updated_at_utc": row["updated_at_utc"] or "",
        }
        for row in cur.fetchall()
    ]


def get_guideline_graph(guideline_id: str, apply_patches: bool = True) -> Optional[Dict[str, Any]]:
    if not guideline_id:
        return None
    init_db()
    with _get_db() as conn:
        cur = conn.execute(
            "SELECT graph_json FROM kb_guideline_graphs WHERE guideline_id = ?",
            (guideline_id,),
        )
        row = cur.fetchone()
        if not row:
            return None
        try:
            graph = json.loads(row["graph_json"] or "{}")
        except Exception:
            return None
        if not apply_patches:
            return graph
        cur = conn.execute(
            "SELECT patch_json FROM kb_guideline_graph_patches WHERE guideline_id = ? ORDER BY updated_at_utc DESC",
            (guideline_id,),
        )
        patch_row = cur.fetchone()
        if patch_row and patch_row["patch_json"]:
            try:
                patch_ops = json.loads(patch_row["patch_json"])
                if isinstance(patch_ops, list):
                    graph = _apply_patch_ops(graph, patch_ops)
            except Exception:
                pass
        return graph


def get_guideline_detail(guideline_id: str) -> Optional[Dict[str, Any]]:
    init_db()
    with _get_db() as conn:
        cur = conn.execute(
            "SELECT guideline_id, title, jurisdiction, version_date, source_url, site_url FROM kb_guidelines WHERE guideline_id = ?",
            (guideline_id,),
        )
        row = cur.fetchone()
        if not row:
            return None
        graph = get_guideline_graph(guideline_id, apply_patches=True)
        assets = _get_guideline_assets(conn, row["site_url"] or "")
    return {
        "guideline": {
            "guideline_id": row["guideline_id"] or "",
            "title": row["title"] or "",
            "jurisdiction": row["jurisdiction"] or "",
            "version_date": row["version_date"] or "",
            "source_url": row["source_url"] or "",
            "site_url": row["site_url"] or "",
        },
        "graph": graph,
        "assets": assets,
    }


def save_guideline_patch(guideline_id: str, patch_ops: List[Dict[str, Any]]) -> None:
    init_db()
    now = _utc_now_iso()
    with _get_db() as conn:
        conn.execute("DELETE FROM kb_guideline_graph_patches WHERE guideline_id = ?", (guideline_id,))
        conn.execute(
            """
            INSERT INTO kb_guideline_graph_patches (guideline_id, patch_json, created_at_utc, updated_at_utc)
            VALUES (?, ?, ?, ?)
            """,
            (guideline_id, json.dumps(patch_ops), now, now),
        )
        conn.commit()


def reextract_guideline(guideline_id: str) -> None:
    init_db()
    with _get_db() as conn:
        cur = conn.execute("SELECT source_url, site_url FROM kb_guidelines WHERE guideline_id = ?", (guideline_id,))
        row = cur.fetchone()
        if not row:
            raise ValueError("Guideline not found.")
        site_url = row["site_url"] or ""
    if not site_url:
        raise ValueError("Site URL missing.")
    pages = _crawl_site(site_url, KB_MAX_PAGES, KB_MAX_DEPTH)
    _index_guidelines_for_site(site_url, pages)


def search_guideline_graphs(query: str, limit: int = 3) -> List[Dict[str, Any]]:
    if not query:
        return []
    init_db()
    q = _sanitize_query(query)
    if not q:
        return []
    tokens = [t for t in q.split(" ") if t]
    if not tokens:
        return []
    match = " OR ".join([f"{t}*" for t in tokens[:6]])
    with _get_db() as conn:
        try:
            cur = conn.execute(
                "SELECT guideline_id, metadata_json, bm25(kb_guideline_graph_index) AS score FROM kb_guideline_graph_index WHERE kb_guideline_graph_index MATCH ? ORDER BY score LIMIT ?",
                (match, int(limit)),
            )
        except Exception:
            return []
        rows = cur.fetchall()
    out: List[Dict[str, Any]] = []
    for row in rows:
        meta = {}
        try:
            meta = json.loads(row["metadata_json"] or "{}")
        except Exception:
            meta = {}
        out.append(
            {
                "guideline_id": row["guideline_id"] or "",
                "metadata": meta,
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
