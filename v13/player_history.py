"""Paginated BALLDONTLIE player box-score downloader with safe retry behavior."""
import os
import time
import requests

URL = "https://api.balldontlie.io/v1/stats"


def fetch_pages(season, max_pages=10, session=None, sleep=time.sleep):
    if max_pages < 1:
        raise ValueError("max_pages must be positive")
    key = os.environ.get("BALLDONTLIE_API_KEY")
    if not key:
        raise RuntimeError("BALLDONTLIE_API_KEY is required")
    client = session or requests.Session()
    cursor = None
    seen = set()
    for page in range(max_pages):
        params = {"seasons[]": season, "per_page": 100}
        if cursor is not None:
            params["cursor"] = cursor
        for attempt in range(8):
            response = client.get(URL, headers={"Authorization": key}, params=params, timeout=35)
            if response.status_code in (429, 500, 502, 503, 504):
                if attempt == 7:
                    response.raise_for_status()
                raw = response.headers.get("Retry-After", "")
                try:
                    delay = max(1, min(float(raw), 120))
                except ValueError:
                    delay = min(10 * (attempt + 1), 120)
                print(f"HTTP {response.status_code}; retrying after {delay}s", flush=True)
                sleep(delay)
                continue
            response.raise_for_status()
            break
        payload = response.json()
        rows = payload.get("data", [])
        if not isinstance(rows, list):
            raise ValueError("Unexpected stats response data")
        next_cursor = payload.get("meta", {}).get("next_cursor")
        if next_cursor is not None:
            if str(next_cursor) in seen:
                raise RuntimeError("Repeated cursor")
            seen.add(str(next_cursor))
        yield page + 1, rows, next_cursor
        if next_cursor is None:
            return
        cursor = next_cursor
        if session is None:
            sleep(0.5)
