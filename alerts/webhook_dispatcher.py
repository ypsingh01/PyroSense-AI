"""Async webhook dispatcher for PyroSense AI alerts.

Example:
    >>> import asyncio
    >>> from alerts.webhook_dispatcher import WebhookDispatcher
    >>> async def _t():\n...     d = WebhookDispatcher("https://example.com")\n...     await d.send({"hello": "world"})\n... \n>>> True
    True
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import httpx

from utils.logger import logger


@dataclass
class WebhookResult:
    ok: bool
    status_code: Optional[int]
    error: Optional[str]


class WebhookDispatcher:
    """Send alert payloads to an arbitrary webhook endpoint."""

    def __init__(self, url: str, timeout_s: float = 8.0) -> None:
        self.url = url
        self.timeout_s = float(timeout_s)

    async def send(self, payload: Dict[str, Any]) -> WebhookResult:
        """POST a JSON payload to the webhook URL."""

        try:
            async with httpx.AsyncClient(timeout=self.timeout_s) as client:
                resp = await client.post(self.url, json=payload)
            ok = 200 <= resp.status_code < 300
            if not ok:
                return WebhookResult(ok=False, status_code=resp.status_code, error=resp.text[:500])
            return WebhookResult(ok=True, status_code=resp.status_code, error=None)
        except Exception as e:
            logger.warning(f"Webhook dispatch failed: {e}")
            return WebhookResult(ok=False, status_code=None, error=str(e))

