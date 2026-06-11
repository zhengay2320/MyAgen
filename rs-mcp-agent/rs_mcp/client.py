from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any

DEFAULT_SERVICE_URL = "http://127.0.0.1:8765"
DEFAULT_TIMEOUT_SECONDS = 600.0
SERVICE_NOT_RUNNING_MESSAGE = "FastAPI service is not running, start with python scripts/run_api.py"


class ServiceClientError(RuntimeError):
    """Raised when the MCP server cannot call the FastAPI service."""


class RsServiceClient:
    """Small HTTP client for the rs_service FastAPI backend."""

    def __init__(self, base_url: str | None = None, timeout: float = DEFAULT_TIMEOUT_SECONDS) -> None:
        self.base_url = (base_url or os.getenv("RS_SERVICE_URL") or DEFAULT_SERVICE_URL).rstrip("/")
        self.timeout = timeout

    def get(self, path: str) -> dict[str, Any]:
        """Send a GET request to the FastAPI service."""
        return self._request("GET", path)

    def post(self, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        """Send a POST request to the FastAPI service."""
        return self._request("POST", path, payload or {})

    def _request(self, method: str, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        """Dispatch a request, preferring httpx and falling back to urllib when unavailable."""
        try:
            import httpx  # type: ignore

            try:
                with httpx.Client(base_url=self.base_url, timeout=self.timeout) as client:
                    response = client.request(method, path, json=payload if method != "GET" else None)
                    response.raise_for_status()
                    return response.json()
            except (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout) as exc:
                raise ServiceClientError(SERVICE_NOT_RUNNING_MESSAGE) from exc
            except httpx.HTTPStatusError as exc:
                raise ServiceClientError(_format_http_error(exc.response.status_code, exc.response.text)) from exc
        except ImportError:
            return self._request_urllib(method, path, payload)

    def _request_urllib(self, method: str, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        """Fallback request implementation used only when httpx is not installed."""
        url = f"{self.base_url}{path}"
        data = None
        headers: dict[str, str] = {}
        if method != "GET":
            data = json.dumps(payload or {}).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError) as exc:
            raise ServiceClientError(SERVICE_NOT_RUNNING_MESSAGE) from exc
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise ServiceClientError(_format_http_error(exc.code, body)) from exc


def unwrap_result(payload: dict[str, Any]) -> Any:
    """Return `result` from wrapped API responses, otherwise return the payload unchanged."""
    if payload.get("ok") is True and "result" in payload:
        return payload["result"]
    return payload


def _format_http_error(status_code: int, body: str) -> str:
    """Format FastAPI error responses for MCP clients."""
    try:
        parsed = json.loads(body)
        detail = parsed.get("detail", parsed)
    except Exception:
        detail = body
    return f"FastAPI service returned HTTP {status_code}: {detail}"
