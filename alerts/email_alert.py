"""Email alert channel using smtplib with an HTML template.

Example:
    >>> from alerts.email_alert import EmailAlert
    >>> _ = EmailAlert(smtp_host="smtp.gmail.com", smtp_port=587, user="u", password="p", recipient="r")
"""

from __future__ import annotations

import smtplib
from dataclasses import dataclass
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Optional

from utils.logger import logger


@dataclass
class EmailResult:
    ok: bool
    error: Optional[str]


class EmailAlert:
    """Send a styled HTML email."""

    def __init__(self, *, smtp_host: str, smtp_port: int, user: str, password: str, recipient: str) -> None:
        self.smtp_host = smtp_host
        self.smtp_port = int(smtp_port)
        self.user = user
        self.password = password
        self.recipient = recipient
        self.template_path = Path("alerts/templates/email_template.html")

    def send(
        self,
        *,
        subject: str,
        timestamp: str,
        location: str,
        class_name: str,
        confidence_pct: float,
        risk_score: float,
        risk_severity: str,
        llm_summary: str,
    ) -> EmailResult:
        """Send an HTML email using the local template."""

        try:
            html = self.template_path.read_text(encoding="utf-8")
        except Exception as e:
            return EmailResult(ok=False, error=f"Email template missing: {e}")

        try:
            body = html.format(
                timestamp=timestamp,
                location=location,
                class_name_upper=class_name.upper(),
                confidence_pct=float(confidence_pct),
                risk_score=float(risk_score),
                risk_severity=risk_severity,
                llm_summary=llm_summary,
            )
        except Exception as e:
            return EmailResult(ok=False, error=f"Template format error: {e}")

        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = self.user
        msg["To"] = self.recipient
        msg.attach(MIMEText(body, "html"))

        try:
            server = smtplib.SMTP(self.smtp_host, self.smtp_port, timeout=10)
            server.starttls()
            server.login(self.user, self.password)
            server.sendmail(self.user, [self.recipient], msg.as_string())
            server.quit()
            logger.info("Email alert sent.")
            return EmailResult(ok=True, error=None)
        except Exception as e:
            logger.warning(f"Email alert failed: {e}")
            return EmailResult(ok=False, error=str(e))

