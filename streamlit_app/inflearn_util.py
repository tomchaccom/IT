"""인프런: 공식 API 없음 — 로컬 큐레이션 JSON + 검색 URL 생성."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any
from urllib.parse import quote

_CATALOG_PATH = Path(__file__).resolve().parent / "data" / "inflearn_catalog.json"


def _load_catalog() -> list[dict[str, Any]]:
    if not _CATALOG_PATH.is_file():
        return []
    try:
        data = json.loads(_CATALOG_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []
    courses = data.get("courses")
    return courses if isinstance(courses, list) else []


def _tokens(*text_parts: str) -> set[str]:
    blob = " ".join(t for t in text_parts if t).lower()
    parts = re.split(r"[^\w가-힣]+", blob, flags=re.UNICODE)
    return {p for p in parts if len(p) >= 2}


def inflearn_search_url(query: str) -> str:
    q = query.strip()
    if not q:
        q = "프로그래밍"
    return f"https://www.inflearn.com/courses?search={quote(q)}"


def match_curated_courses(
    user_profile: dict[str, Any],
    articles: list[dict[str, Any]],
    *,
    max_courses: int = 5,
) -> list[tuple[dict[str, Any], int]]:
    catalog = _load_catalog()
    if not catalog:
        return []
    p = user_profile or {}
    blob_profile = " ".join(
        str(p.get(k) or "")
        for k in ("role", "stack", "topics")
    )
    blob_articles = []
    for a in articles[:8]:
        blob_articles.append(f"{a.get('title', '')} {a.get('summary', '')}")
    blob_all = blob_profile + " " + " ".join(blob_articles)
    blob_lower = blob_all.lower()
    keys = _tokens(blob_all)

    scored: list[tuple[dict[str, Any], int]] = []
    for c in catalog:
        if not isinstance(c, dict):
            continue
        tags = c.get("tags") or []
        if not isinstance(tags, list):
            continue
        tagset = {str(t).lower() for t in tags if t}
        score = 0
        for t in tagset:
            if t in blob_lower or t in keys:
                score += 1
        if score > 0:
            scored.append((c, score))
    scored.sort(key=lambda x: -x[1])
    return scored[:max(1, min(max_courses, 12))]


def format_inflearn_tool_output(
    user_profile: dict[str, Any],
    articles: list[dict[str, Any]],
    *,
    max_courses: int = 5,
) -> str:
    p = user_profile or {}
    stack = (p.get("stack") or "").strip()
    topics = (p.get("topics") or "").strip()
    role = (p.get("role") or "").strip()

    matches = match_curated_courses(user_profile, articles, max_courses=max_courses)
    lines: list[str] = ["인프런 학습 추천", ""]
    if matches:
        lines.append("추천 강의")
        for c, _ in matches:
            title = str(c.get("title") or "")
            url = str(c.get("url") or "")
            tags = c.get("tags") or []
            tagstr = ", ".join(str(t) for t in tags) if isinstance(tags, list) else ""
            lines.append(f"- {title}\n  링크: {url}\n  관련 키워드: {tagstr}")
        lines.append("")
    else:
        lines.append("프로필과 겹치는 추천 강의가 없어 검색으로 안내합니다.")
        lines.append("")

    search_queries: list[str] = []
    if topics:
        search_queries.append(topics.split(",")[0].strip()[:40])
    if stack:
        search_queries.append(stack.split(",")[0].strip()[:40])
    if role:
        search_queries.append(role[:30])
    if not search_queries:
        search_queries.append("IT 프로그래밍")

    lines.append("인프런에서 검색하기")
    seen: set[str] = set()
    for q in search_queries:
        u = inflearn_search_url(q)
        if u not in seen:
            seen.add(u)
            lines.append(f"- «{q}» {u}")
        if len(seen) >= 3:
            break

    return "\n".join(lines)


def inflearn_search_link_pairs(user_profile: dict[str, Any], *, max_links: int = 3) -> list[tuple[str, str]]:
    """(표시용 라벨, 검색 URL) 목록. 강의 카드 옆 빠른 검색 버튼에 사용."""
    p = user_profile or {}
    stack = (p.get("stack") or "").strip()
    topics = (p.get("topics") or "").strip()
    role = (p.get("role") or "").strip()
    search_queries: list[str] = []
    if topics:
        search_queries.append(topics.split(",")[0].strip()[:40])
    if stack:
        search_queries.append(stack.split(",")[0].strip()[:40])
    if role:
        search_queries.append(role[:30])
    if not search_queries:
        search_queries.append("IT 프로그래밍")
    out: list[tuple[str, str]] = []
    seen: set[str] = set()
    for q in search_queries:
        u = inflearn_search_url(q)
        if u not in seen:
            seen.add(u)
            out.append((q, u))
        if len(out) >= max_links:
            break
    return out
