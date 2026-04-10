"""메시지 히스토리에서 ReAct 도구 호출 타임라인 추출."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage


def react_tool_timeline(messages: Sequence[BaseMessage] | None) -> list[dict[str, Any]]:
    if not messages:
        return []
    out: list[dict[str, Any]] = []
    for m in messages:
        if isinstance(m, AIMessage) and m.tool_calls:
            for tc in m.tool_calls:
                if isinstance(tc, dict):
                    name = tc.get("name", "")
                    args = tc.get("args") or tc.get("arguments")
                else:
                    name = getattr(tc, "name", "") or ""
                    args = getattr(tc, "args", None)
                out.append({"kind": "tool_call", "name": name, "args": args})
        elif isinstance(m, ToolMessage):
            name = m.name or "tool"
            content = (m.content or "")[:2000]
            out.append({"kind": "tool_result", "name": name, "content_preview": content})
    return out
