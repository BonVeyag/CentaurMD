import smtplib
from email.message import EmailMessage
from typing import Dict, Any

from app.auth import _load_smtp_config
from app.auth import _format_from_header
import logging

logger = logging.getLogger("centaurweb.email")


def send_email(subject: str, body: str) -> bool:
    config: Dict[str, Any] = _load_smtp_config()
    host = config.get("host", "")
    admin_email = config.get("admin_email", "")
    if not host or not admin_email or not config.get("configured"):
        logger.warning("Admin report email skipped: SMTP not configured.")
        return False

    port = int(config.get("port", 587))
    smtp_user = config.get("user", "")
    smtp_pass = config.get("pass", "")
    use_tls = bool(config.get("tls", True))
    use_ssl = bool(config.get("ssl", False))
    from_header = (config.get("from_header") or _format_from_header("", smtp_user)).strip()
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = from_header or smtp_user
    msg["To"] = admin_email
    msg.set_content(body)

    try:
        if use_ssl:
            server = smtplib.SMTP_SSL(host, port, timeout=int(config.get("timeout", 10)))
        else:
            server = smtplib.SMTP(host, port, timeout=int(config.get("timeout", 10)))
        server.ehlo()
        if use_tls:
            server.starttls()
            server.ehlo()
        server.login(smtp_user, smtp_pass)
        server.send_message(msg)
        server.quit()
        logger.info("Admin report email sent to %s", admin_email)
        return True
    except Exception as exc:
        logger.exception("Failed to send admin report email: %s", exc)
        return False
