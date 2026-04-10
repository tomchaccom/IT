"""ReAct 단계에서 사용하는 도구 (상태 주입으로 기사·프로필 참조)."""

from __future__ import annotations

from typing import Annotated, Any

from langchain.tools import InjectedState, tool

from streamlit_app.inflearn_util import format_inflearn_tool_output


@tool
def get_ranked_headlines(
    state: Annotated[dict[str, Any], InjectedState],
    top_n: int = 12,
) -> str:
    """수집된 IT 뉴스 중 관련도 순으로 상위 N개의 제목·링크·짧은 요약을 가져옵니다. 요약 작성 전에 반드시 호출하세요."""
    articles = state.get("articles") or []
    scores = state.get("article_scores") or {}
    lines: list[str] = []
    for i, a in enumerate(articles[: max(1, min(top_n, 30))], start=1):
        sc = scores.get(a.get("link", ""), 0.0)
        lines.append(
            f"{i}. [{sc:.3f}] {a.get('title', '')}\n   URL: {a.get('link', '')}\n"
            f"   요약: {a.get('summary', '')[:240]}"
        )
    if not lines:
        return "(기사 없음 — 수집 단계를 확인하세요.)"
    return "\n\n".join(lines)


@tool
def get_user_interest_profile(
    state: Annotated[dict[str, Any], InjectedState],
) -> str:
    """사용자의 구조화된 관심사(역할·스택·토픽)를 텍스트로 반환합니다. 요약에 맞춤화할 때 사용하세요."""
    p = state.get("user_profile") or {}
    role = p.get("role") or ""
    stack = p.get("stack") or ""
    topics = p.get("topics") or ""
    parts = [
        f"역할: {role}",
        f"기술 스택: {stack}",
        f"관심 토픽: {topics}",
    ]
    return "\n".join(parts)


@tool
def get_inflearn_learning_suggestions(
    state: Annotated[dict[str, Any], InjectedState],
    max_courses: int = 5,
) -> str:
    """인프런 강의 링크·검색 URL 목록을 반환합니다. 답변에 넣을 인프런 주소는 이 출력에 있는 것만 사용하세요."""
    profile = state.get("user_profile") or {}
    articles = list(state.get("articles") or [])
    return format_inflearn_tool_output(
        profile,
        articles,
        max_courses=max(1, min(max_courses, 12)),
    )


def agent_tools_list():
    return [
        get_ranked_headlines,
        get_user_interest_profile,
        get_inflearn_learning_suggestions,
    ]
