"""NewsAPI.org 선택 수집 — RSS와 동일한 기사 dict 형식으로 병합."""

from __future__ import annotations

import os
from typing import Any

import httpx

from streamlit_app.config import HTTP_TIMEOUT, NEWSAPI_TOP_HEADLINES_URL, NEWSAPI_PAGE_SIZE, USER_AGENT


def fetch_newsapi_top_headlines() -> tuple[list[dict[str, Any]], str | None]:
    """NEWS_API_KEY가 있으면 technology 헤드라인을 가져옵니다. 없거나 실패하면 ([], 오류메시지)."""
    key = (os.getenv("NEWS_API_KEY") or os.getenv("NEWSAPI_API_KEY") or "").strip()
    if not key:
        return [], None
    params: dict[str, str | int] = {
        "apiKey": key,
        "category": "technology",
        "pageSize": NEWSAPI_PAGE_SIZE,
        "language": "en",
    }
    headers = {"User-Agent": USER_AGENT}
    try:
        with httpx.Client(timeout=HTTP_TIMEOUT, follow_redirects=True, headers=headers) as client:
            r = client.get(NEWSAPI_TOP_HEADLINES_URL, params=params)
            r.raise_for_status()
            data = r.json()
    except Exception as e:
        return [], str(e)

    if data.get("status") != "ok":
        return [], str(data.get("message") or data.get("code") or "NewsAPI 비정상 응답")

    raw = data.get("articles")
    if not isinstance(raw, list):
        return [], None

    out: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        title = (item.get("title") or "").strip()
        link = (item.get("url") or "").strip()
        if not title or not link:
            continue
        desc = (item.get("description") or item.get("content") or "").strip()
        if len(desc) > 800:
            desc = desc[:799] + "…"
        pub = (item.get("publishedAt") or "").strip()
        src = item.get("source") or {}
        src_name = src.get("name") if isinstance(src, dict) else ""
        out.append(
            {
                "title": title,
                "link": link,
                "summary": desc or f"(NewsAPI) {src_name}".strip(),
                "published": pub,
                "source_feed": f"newsapi:top-headlines:{src_name}",
            }
        )
    return out, None
