# CentaurMD

## SMTP (Feedback + Signup Requests)

CentaurMD sends feedback and signup-request emails via SMTP. It will **not** send email unless SMTP is configured.

### Option A — `data/smtp.json` (Mac mini friendly)
Create and edit `data/smtp.json` (not committed):

```json
{
  "host": "smtp.gmail.com",
  "port": 587,
  "user": "your@gmail.com",
  "pass": "app_password_here",
  "tls": true,
  "ssl": false,
  "timeout": 10,
  "from_name": "CentaurMD",
  "from_email": "your@gmail.com",
  "admin_email": "thapa.rajat@gmail.com"
}
```

### Option B — Environment Variables
Environment variables take precedence over the file:

```
CENTAUR_SMTP_HOST
CENTAUR_SMTP_PORT
CENTAUR_SMTP_USER
CENTAUR_SMTP_PASS
CENTAUR_SMTP_TLS
CENTAUR_SMTP_SSL
CENTAUR_SMTP_TIMEOUT
CENTAUR_SMTP_FROM_NAME
CENTAUR_SMTP_FROM_EMAIL
CENTAUR_ADMIN_EMAIL
```

### Notes
- The UI **never** shows the admin email.
- `/api/feedback` returns a clear error code if SMTP isn’t configured.
- Admin-only endpoints:
  - `GET /api/admin/smtp_status`
  - `POST /api/admin/test_email`

## Local Knowledge Base (LKB)

Centaur can index local guideline sources and use them first during Clinical Query.

### Storage
- SQLite: `data/local_kb.sqlite`
- Assets (PDF/SVG/image): `data/kb_assets/`

### Admin endpoints
- `GET /api/admin/local_kb/sites`
- `POST /api/admin/local_kb/index`
- `GET /api/admin/local_kb/guidelines`
- `GET /api/admin/local_kb/guidelines/{guideline_id}`
- `POST /api/admin/local_kb/guidelines/{guideline_id}/patch`
- `POST /api/admin/local_kb/guidelines/{guideline_id}/reextract`

### Environment toggles
- `CENTAUR_KB_ENABLED=1`
- `CENTAUR_KB_MAX_PAGES=25`
- `CENTAUR_KB_MAX_DEPTH=2`
- `CENTAUR_KB_REFRESH_DAYS=30`
- `CENTAUR_KB_ENABLE_VISION=1` (optional, enables vision extraction for raster-only diagrams)
- `CENTAUR_KB_VISION_MODEL=gpt-4o-mini`
- `CENTAUR_GUIDELINE_LLM_VARS=1` (optional, LLM-assisted variable extraction for guideline runner)

### Notes
- Structured sources are preferred: SVG/HTML text > PDF text layer > vision fallback.
- Guideline graphs are deterministic at runtime; patches persist across refresh runs.

## Ambient Mode

Ambient Mode enables continuous room audio capture with automatic segmentation and transcription.

### Behavior
- Opt-in toggle in Personalization.
- Consent required at the start of each encounter before any audio is processed or uploaded.
- No raw audio is stored server-side; only transcript text and segment metadata.

### API endpoints
- `POST /api/ambient/start_encounter`
- `POST /api/ambient/upload_segment`
- `POST /api/ambient/stop_encounter`
