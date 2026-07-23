from __future__ import annotations

import time
from collections.abc import Callable

import requests


TRANSIENT_HTTP_STATUS_CODES = {408, 409, 429, 500, 502, 503, 504}


def post_json_with_retry(
    post: Callable,
    url: str,
    *,
    headers: dict,
    json: dict,
    timeout: int,
    max_attempts: int = 3,
    sleep_func: Callable[[float], None] = time.sleep,
):
    """POST JSON with bounded retries for transient network and HTTP failures."""
    attempts = max(1, int(max_attempts))
    for attempt in range(attempts):
        try:
            response = post(
                url,
                headers=headers,
                json=json,
                timeout=timeout,
            )
        except (requests.ConnectionError, requests.Timeout):
            if attempt + 1 >= attempts:
                raise
            response = None

        status_code = int(getattr(response, "status_code", 0) or 0)
        if response is not None and status_code not in TRANSIENT_HTTP_STATUS_CODES:
            return response
        if attempt + 1 >= attempts:
            return response

        delay = min(2 ** attempt, 4)
        retry_after = getattr(response, "headers", {}).get("Retry-After") if response else None
        try:
            delay = min(max(float(retry_after), delay), 10)
        except (TypeError, ValueError):
            pass
        sleep_func(delay)

    raise RuntimeError("模型接口重试流程异常结束。")
