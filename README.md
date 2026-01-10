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
