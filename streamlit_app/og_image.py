"""페이지 HTML에서 og:image URL 추출 (인프런 강의 썸네일 등)."""

from __future__ import annotations

import re

import httpx

from streamlit_app.config import HTTP_TIMEOUT, USER_AGENT

_OG1 = re.compile(
    r'<meta\s+property=["\']og:image["\']\s+content=["\']([^"\']+)["\']',
    re.I,
)
_OG2 = re.compile(
    r'<meta\s+content=["\']([^"\']+)["\']\s+property=["\']og:image["\']',
    re.I,
)


def fetch_og_image_url(page_url: str) -> str | None:
    if not page_url or not page_url.startswith("http"):
        return None
    headers = {"User-Agent": USER_AGENT, "Accept": "text/html,application/xhtml+xml"}
    try:
        with httpx.Client(timeout=min(HTTP_TIMEOUT, 15.0), follow_redirects=True, headers=headers) as client:
            r = client.get(page_url)
            r.raise_for_status()
            text = r.text[:800_000]
    except Exception:
        return None
    m = _OG1.search(text) or _OG2.search(text)
    if not m:
        return None
    url = m.group(1).strip()
    if url.startswith("//"):
        return "https:" + url
    return url or None
