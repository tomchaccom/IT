"""RSS 수집 → 관심사 매칭 → (저매칭 시) 피드 확장 → ReAct 요약 그래프."""

from __future__ import annotations

import json
import re
from typing import Any, Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import create_react_agent

from streamlit_app.agent_tools import agent_tools_list
from streamlit_app.config import DEFAULT_FEEDS, EXTENDED_FEEDS, MATCH_THRESHOLD
from streamlit_app.news_api_util import fetch_newsapi_top_headlines
from streamlit_app.rss_util import fetch_feed_items
from streamlit_app.scoring import score_article
from streamlit_app.state import NewsAgentState

SYSTEM_KO = """당신은 IT 뉴스 큐레이터이자 커리어·기술 전략 코치입니다.
반드시 다음 도구를 모두 한 번 이상 호출해 근거를 확인한 뒤, 한국어로만 마크다운을 작성합니다:
  • `get_ranked_headlines` — 오늘의 기사 근거
  • `get_user_interest_profile` — 사용자 맞춤
  • `get_inflearn_learning_suggestions` — 인프런 강의·검색 링크 (도구에 없는 URL을 만들지 않음)

출력은 반드시 아래 **다섯 줄의 ## 제목을 그대로** 사용해 섹션을 나눕니다 (순서 동일, 오타 없이):
## IT 트렌드 인사이트
## 나에게 맞는 조언
## 핵심 이슈
## 추천 기사
## 인프런 학습

각 섹션 안에서:
- **IT 트렌드 인사이트**: 최근 흐름을 한두 문단으로 압축해 “요즘 IT 업계에서는 이런 방향이 두드러진다”는 **인사이트**를 제시합니다. 도구에서 본 기사와 모순되지 않게 씁니다.
- **나에게 맞는 조언**: `get_user_interest_profile`의 역할·스택·토픽을 명시적으로 언급하고, 그 프로필을 전제로 **어떤 역량·학습·실무 습관을 강화하면 좋은지** 3~6문장으로 구체적으로 조언합니다.
- **핵심 이슈**: 불릿 위주로 오늘 기사 묶음의 핵심만 정리합니다.
- **추천 기사**: 3~5개, 제목 + URL (`get_ranked_headlines`에 있는 URL만).
- **인프런 학습**: `get_inflearn_learning_suggestions`에 나온 제목·URL·검색 링크만 인용합니다.

`get_ranked_headlines`의 제목·요약은 이미 한국어입니다. 기사·인프런 URL은 도구 출력만 사용하고 임의로 만들지 마세요."""

LOCALIZE_TOP_N = 22

_LOCALIZE_SYSTEM = """당신은 IT 뉴스 번역·요약 전문가입니다.
입력에 있는 각 기사에 대해 자연스러운 한국어 제목(title_ko)과 본문 스니펫을 바탕으로 한국어 요약(summary_ko, 2~4문장)을 만듭니다.
고유명사는 필요 시 원문 표기를 괄호로 병기해도 됩니다.
응답은 반드시 유효한 JSON 배열만 출력합니다. 각 원소는 {"link": "<입력과 동일한 URL>", "title_ko": "...", "summary_ko": "..."} 형식입니다.
입력에 없는 link를 만들거나 빈 항목을 넣지 마세요."""

_SUMMARY_SNIP = 400


def _ai_message_text(msg: Any) -> str:
    c = getattr(msg, "content", msg)
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        parts: list[str] = []
        for block in c:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text") or ""))
        return "".join(parts)
    return str(c)


def _parse_localization_json(raw: str) -> list[dict[str, Any]]:
    text = raw.strip()
    fence = re.search(r"```(?:json)?\s*\n(.*)\n```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    start = text.find("[")
    if start == -1:
        return []
    try:
        data, _ = json.JSONDecoder().raw_decode(text, start)
    except json.JSONDecodeError:
        return []
    if not isinstance(data, list):
        return []
    out: list[dict[str, Any]] = []
    for item in data:
        if isinstance(item, dict) and item.get("link"):
            out.append(item)
    return out


def ingest_rss(state: NewsAgentState) -> dict[str, Any]:
    urls = list(state.get("feed_urls") or list(DEFAULT_FEEDS))
    merged: list[dict[str, Any]] = []
    errors: list[str] = []
    for url in urls:
        try:
            merged.extend(fetch_feed_items(url))
        except Exception as e:
            errors.append(f"{url}: {e}")
    api_articles, api_err = fetch_newsapi_top_headlines()
    if api_err:
        errors.append(f"NewsAPI: {api_err}")
    elif api_articles:
        merged.extend(api_articles)
    seen: set[str] = set()
    uniq: list[dict[str, Any]] = []
    for a in merged:
        link = a.get("link") or ""
        if link in seen:
            continue
        seen.add(link)
        uniq.append(a)
    detail = f"{len(uniq)}개 기사, 피드 {len(urls)}개"
    if api_articles and not api_err:
        detail += f", NewsAPI 병합 {len(api_articles)}건 소스"
    if errors:
        detail += f" | 오류 {len(errors)}건"
    trace = list(state.get("tool_trace") or [])
    trace.append(
        {
            "node": "ingest_rss",
            "detail_ko": detail,
            "errors": errors[:5] if errors else [],
        }
    )
    return {
        "articles": uniq,
        "tool_trace": trace,
    }


def score_articles(state: NewsAgentState) -> dict[str, Any]:
    profile = state.get("user_profile") or {}
    articles = list(state.get("articles") or [])
    scores: dict[str, float] = {}
    for a in articles:
        link = a.get("link") or ""
        scores[link] = score_article(profile, a)
    max_score = max(scores.values()) if scores else 0.0
    ranked = sorted(articles, key=lambda x: scores.get(x.get("link") or "", 0.0), reverse=True)
    trace = list(state.get("tool_trace") or [])
    trace.append(
        {
            "node": "score_articles",
            "detail_ko": f"최고 관련도 {max_score:.3f} (임계값 {MATCH_THRESHOLD})",
        }
    )
    return {
        "articles": ranked,
        "article_scores": scores,
        "max_match_score": max_score,
        "tool_trace": trace,
    }


def route_after_score(state: NewsAgentState) -> Literal["expand_scope", "localize_articles"]:
    max_score = float(state.get("max_match_score") or 0.0)
    expanded = bool(state.get("scope_expanded"))
    if max_score >= MATCH_THRESHOLD:
        return "localize_articles"
    if not expanded:
        return "expand_scope"
    return "localize_articles"


def expand_scope(state: NewsAgentState) -> dict[str, Any]:
    if state.get("scope_expanded"):
        return {}
    urls = list(state.get("feed_urls") or list(DEFAULT_FEEDS))
    for u in EXTENDED_FEEDS:
        if u not in urls:
            urls.append(u)
    max_score = float(state.get("max_match_score") or 0.0)
    events = list(state.get("branch_events") or [])
    events.append(
        {
            "type": "low_match_expand",
            "detail_ko": (
                f"관련도 최고치 {max_score:.3f}가 임계값 {MATCH_THRESHOLD} 미만이어서 "
                "추가 RSS 피드를 포함했습니다."
            ),
        }
    )
    trace = list(state.get("tool_trace") or [])
    trace.append({"node": "expand_scope", "detail_ko": f"피드 수 {len(urls)}개로 확장"})
    return {
        "feed_urls": urls,
        "scope_expanded": True,
        "branch_events": events,
        "tool_trace": trace,
    }


def prepare_react(state: NewsAgentState) -> dict[str, Any]:
    msg = HumanMessage(
        content=(
            "수집·정렬된 IT 뉴스와 사용자 프로필을 바탕으로, 시스템 프롬프트에 정한 다섯 개 ## 섹션 제목을 "
            "정확히 지켜 작성하세요. 특히 «IT 트렌드 인사이트»에서는 요즘 업계 동향에 대한 통찰을, "
            "«나에게 맞는 조언»에서는 해당 사용자가 강화하면 좋은 역량과 다음 스텝을 알려 주세요."
        )
    )
    trace = list(state.get("tool_trace") or [])
    trace.append({"node": "prepare_react", "detail_ko": "ReAct 단계용 사용자 메시지 추가"})
    return {
        "messages": [msg],
        "tool_trace": trace,
    }


def build_news_workflow(llm: ChatOpenAI):
    def localize_articles(state: NewsAgentState) -> dict[str, Any]:
        articles = list(state.get("articles") or [])
        trace = list(state.get("tool_trace") or [])
        if not articles:
            trace.append({"node": "localize_articles", "detail_ko": "기사 없음 — 건너뜀"})
            return {"tool_trace": trace}
        batch = articles[:LOCALIZE_TOP_N]
        lines: list[str] = []
        for i, a in enumerate(batch, start=1):
            link = (a.get("link") or "").strip()
            title = (a.get("title") or "").strip()
            snip = (a.get("summary") or "").replace("\n", " ").strip()
            if len(snip) > _SUMMARY_SNIP:
                snip = snip[: _SUMMARY_SNIP - 1] + "…"
            lines.append(f"{i}. link: {link}\n   title: {title}\n   snippet: {snip}")
        user_block = "다음 기사들을 처리하세요:\n\n" + "\n\n".join(lines)
        detail_ok = ""
        try:
            resp = llm.invoke(
                [SystemMessage(content=_LOCALIZE_SYSTEM), HumanMessage(content=user_block)]
            )
            parsed = _parse_localization_json(_ai_message_text(resp))
            by_link = {
                str(p.get("link", "")).strip(): p
                for p in parsed
                if isinstance(p, dict) and p.get("link")
            }
            merged: list[dict[str, Any]] = []
            for a in articles:
                link = (a.get("link") or "").strip()
                patch = by_link.get(link)
                if not patch:
                    merged.append(a)
                    continue
                na = dict(a)
                tko = (patch.get("title_ko") or "").strip()
                sko = (patch.get("summary_ko") or "").strip()
                if tko:
                    na["title"] = tko
                if sko:
                    na["summary"] = sko
                merged.append(na)
            n_done = sum(1 for a in batch if (a.get("link") or "").strip() in by_link)
            detail_ok = f"상위 {len(batch)}건 중 {n_done}건 한국어 제목·요약 반영"
            trace.append({"node": "localize_articles", "detail_ko": detail_ok})
            return {"articles": merged, "tool_trace": trace}
        except Exception as e:
            trace.append(
                {
                    "node": "localize_articles",
                    "detail_ko": f"한국어 변환 실패 — 원문 유지 ({e})",
                }
            )
            return {"tool_trace": trace}

    react_agent = create_react_agent(
        llm,
        agent_tools_list(),
        prompt=SYSTEM_KO,
        state_schema=NewsAgentState,
    )
    g = StateGraph(NewsAgentState)
    g.add_node("ingest_rss", ingest_rss)
    g.add_node("score_articles", score_articles)
    g.add_node("expand_scope", expand_scope)
    g.add_node("localize_articles", localize_articles)
    g.add_node("prepare_react", prepare_react)
    g.add_node("react_agent", react_agent)
    g.add_edge(START, "ingest_rss")
    g.add_edge("ingest_rss", "score_articles")
    g.add_conditional_edges(
        "score_articles",
        route_after_score,
        {
            "expand_scope": "expand_scope",
            "localize_articles": "localize_articles",
        },
    )
    g.add_edge("expand_scope", "ingest_rss")
    g.add_edge("localize_articles", "prepare_react")
    g.add_edge("prepare_react", "react_agent")
    g.add_edge("react_agent", END)
    return g.compile()
