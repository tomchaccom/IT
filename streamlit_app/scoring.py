"""사용자 프로필과 기사 텍스트 간 단순 관련도 (토큰 중첩)."""

from __future__ import annotations

import re
from typing import Any


_TOKEN = re.compile(r"[\w가-힣]+", re.UNICODE)


def _profile_tokens(profile: dict[str, Any]) -> set[str]:
    blobs: list[str] = []
    role = profile.get("role") or ""
    if isinstance(role, str):
        blobs.append(role)
    stack = profile.get("stack") or ""
    if isinstance(stack, str):
        blobs.append(stack)
    topics = profile.get("topics") or ""
    if isinstance(topics, str):
        blobs.append(topics)
    blob = " ".join(blobs).lower()
    return set(_TOKEN.findall(blob)) if blob.strip() else set()


def score_article(profile: dict[str, Any], article: dict[str, Any]) -> float:
    ptoks = _profile_tokens(profile)
    if not ptoks:
        return 0.0
    text = f"{article.get('title', '')} {article.get('summary', '')}".lower()
    atoks = set(_TOKEN.findall(text))
    if not atoks:
        return 0.0
    inter = len(ptoks & atoks)
    union = len(ptoks | atoks)
    return inter / union if union else 0.0
