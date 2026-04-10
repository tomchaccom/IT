"""LangGraph 공유 상태 (ReAct 서브그래프 + 수집·분기 노드)."""

from __future__ import annotations

from typing import Annotated, Any, NotRequired, Sequence, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from langgraph.managed import RemainingSteps


class NewsAgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    remaining_steps: NotRequired[RemainingSteps]

    user_profile: NotRequired[dict[str, Any]]
    feed_urls: NotRequired[list[str]]
    articles: NotRequired[list[dict[str, Any]]]
    article_scores: NotRequired[dict[str, float]]
    max_match_score: NotRequired[float]
    scope_expanded: NotRequired[bool]
    branch_events: NotRequired[list[dict[str, Any]]]
    tool_trace: NotRequired[list[dict[str, Any]]]
