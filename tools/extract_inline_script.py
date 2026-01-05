import re
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT / "frontend" / "index.html"
OUTJS  = ROOT / "frontend" / "legacy_app.js"

html = INDEX.read_text(encoding="utf-8", errors="ignore")

# Find inline <script> blocks with no src= (we only want large embedded JS)
script_re = re.compile(
    r"<script(?![^>]*\bsrc=)[^>]*>(.*?)</script>",
    re.IGNORECASE | re.DOTALL,
)

matches = list(script_re.finditer(html))
if not matches:
    raise SystemExit("No inline <script> blocks without src= found. Nothing to extract.")

# Choose the largest inline script block (usually your 2000+ lines)
largest = max(matches, key=lambda m: len(m.group(1) or ""))

script_body = (largest.group(1) or "").strip()
if len(script_body) < 2000:
    raise SystemExit(
        f"Largest inline script is only {len(script_body)} chars. "
        "Refusing to extract automatically (to avoid extracting the wrong thing)."
    )

# Backup again (belt-and-suspenders)
ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
bak = INDEX.with_suffix(f".html.extractbak.{ts}")
bak.write_text(html, encoding="utf-8")
print(f"Backup written: {bak}")

# Write extracted JS
OUTJS.write_text(script_body + "\n", encoding="utf-8")
print(f"Extracted JS written: {OUTJS} ({len(script_body)} chars)")

# Replace that script block with external script include (same location / order)
replacement = '<script src="/frontend/legacy_app.js"></script>'
new_html = html[:largest.start()] + replacement + html[largest.end():]
INDEX.write_text(new_html, encoding="utf-8")
print("index.html updated to load /frontend/legacy_app.js")
