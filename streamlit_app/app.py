"""IT 뉴스 RSS(+선택 NewsAPI) → 매칭·확장 루프 → 한국어 정규화 → ReAct(기사·프로필·인프런 도구) — PRD v1.2."""

from __future__ import annotations

import hashlib
import os
import re
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from streamlit_app import config
from streamlit_app.inflearn_util import (
    inflearn_search_link_pairs,
    inflearn_search_url,
    match_curated_courses,
)
from streamlit_app.og_image import fetch_og_image_url
from streamlit_app.trace_util import react_tool_timeline
from streamlit_app.workflow import build_news_workflow

_APP_DIR = Path(__file__).resolve().parent

# 스크립트 리런 시에도 이전 분석 결과 유지 (다른 버튼·슬라이더 클릭 시 요약이 사라지지 않도록)
_SIK_FULL = "digest_full_state"
_SIK_PROFILE = "digest_saved_profile"
_SIK_NEWS_LINK = "news_reader_selected_link"

load_dotenv(_ROOT / ".env")
load_dotenv(_ROOT / "notebooks" / ".env")
load_dotenv(_APP_DIR / ".env", override=True)

st.set_page_config(
    page_title="IT 뉴스 트렌드",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _hydrate_env_from_streamlit_secrets() -> None:
    """Streamlit Community Cloud의 Secrets는 os.environ에 주입되지 않음 → LangChain/OpenAI 호환용.

    로컬에는 secrets.toml이 없을 수 있음. 이때 st.secrets 접근은 첫 get/__getitem__에서
    StreamlitSecretNotFoundError(FileNotFoundError)가 나므로 전체를 한 번에 감싼다.
    """

    def _set_if_empty(env_name: str, value: object | None) -> None:
        if value is None:
            return
        s = str(value).strip()
        if not s:
            return
        if (os.getenv(env_name) or "").strip():
            return
        os.environ[env_name] = s

    try:
        sec = st.secrets
        # 최상위 키 (앱 UI에서 쓰는 이름과 동일하게 두는 것을 권장)
        _set_if_empty("OPENAI_API_KEY", sec.get("OPENAI_API_KEY"))
        _set_if_empty("OPENAI_API_KEY", sec.get("openai_api_key"))
        _set_if_empty("NEWS_API_KEY", sec.get("NEWS_API_KEY"))
        _set_if_empty("NEWS_API_KEY", sec.get("NEWSAPI_API_KEY"))
        _set_if_empty("NEWS_API_KEY", sec.get("news_api_key"))
        _set_if_empty("NEWSAPI_API_KEY", sec.get("NEWSAPI_API_KEY"))

        # TOML 섹션 예: [openai] \n api_key = "sk-..."
        openai_sec = sec.get("openai")
        if isinstance(openai_sec, dict):
            _set_if_empty(
                "OPENAI_API_KEY",
                openai_sec.get("api_key") or openai_sec.get("API_KEY"),
            )
    except (OSError, FileNotFoundError):
        # StreamlitSecretNotFoundError → FileNotFoundError 서브클래스
        return
    except Exception:
        return


_hydrate_env_from_streamlit_secrets()

# Tailwind Plus 마케팅 블록 느낌(타이포·여백·카드·차분한 톤)을 참고한 커스텀 테마 — UI 자산은 복사하지 않음.
# 주의: st.markdown의 unsafe_allow_html는 <style>을 제거한 뒤 내용만 텍스트로 남겨 화면에 CSS가 그대로 노출될 수 있음.
# Streamlit 1.56+는 st.html(…)<style>만 있을 때 이벤트 컨테이너로 보내 전역 스타일을 적용함.
_APP_CSS = """
@import url("https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap");

  :root {
    --app-bg-top: #f1f5f9;
    --app-bg-mid: #f8fafc;
    --app-surface: #ffffff;
    --app-border: #e2e8f0;
    --app-text: #334155;
    --app-muted: #64748b;
    --app-heading: #0f172a;
    --app-accent: #0d9488;
    --app-accent-deep: #0f766e;
    --app-shadow: 0 1px 3px rgba(15, 23, 42, 0.06);
    --app-shadow-lg: 0 12px 40px -12px rgba(15, 23, 42, 0.12);
    --app-panel-bg: rgba(255, 255, 255, 0.72);
    --app-accent-soft: rgba(13, 148, 136, 0.08);
  }

  .stApp {
    font-family: "Inter", ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif;
    background: linear-gradient(180deg, var(--app-bg-top) 0%, var(--app-bg-mid) 28%, var(--app-surface) 72%);
  }

  header[data-testid="stHeader"] {
    background: rgba(248, 250, 252, 0.75);
    backdrop-filter: blur(10px);
    border-bottom: 1px solid rgba(226, 232, 240, 0.6);
  }

  .main .block-container {
    padding-top: 1.75rem;
    padding-bottom: 3rem;
    max-width: 1140px;
  }

  .app-hero {
    margin: 0 0 2rem 0;
    padding: 2rem 2.25rem;
    border-radius: 1.25rem;
    background: linear-gradient(135deg, rgba(255, 255, 255, 0.92) 0%, rgba(241, 245, 249, 0.85) 100%);
    border: 1px solid var(--app-border);
    box-shadow: var(--app-shadow), var(--app-shadow-lg);
  }
  .app-hero-badge {
    display: inline-block;
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.07em;
    text-transform: uppercase;
    color: var(--app-accent-deep);
    background: rgba(13, 148, 136, 0.12);
    padding: 0.38rem 0.85rem;
    border-radius: 9999px;
    margin: 0 0 0.85rem 0;
  }
  .app-hero-title {
    font-size: clamp(1.65rem, 4.2vw, 2.35rem);
    font-weight: 700;
    letter-spacing: -0.038em;
    line-height: 1.12;
    color: var(--app-heading);
    margin: 0 0 0.55rem 0;
  }
  .app-hero-lead {
    font-size: 1.06rem;
    color: var(--app-muted);
    line-height: 1.6;
    margin: 0;
    max-width: 40rem;
  }

  h2, h3 {
    font-weight: 600 !important;
    letter-spacing: -0.02em !important;
    color: var(--app-heading) !important;
  }

  .stCaption, [data-testid="stCaptionContainer"] {
    color: var(--app-muted) !important;
  }

  /* 탭: 세그먼트형 툴바 */
  .stTabs [data-baseweb="tab-list"] {
    gap: 0.2rem;
    padding: 0.35rem;
    background: rgba(241, 245, 249, 0.85);
    border-radius: 0.85rem;
    border: 1px solid var(--app-border);
  }
  .stTabs [data-baseweb="tab"] {
    border-radius: 0.55rem;
    font-weight: 500;
    color: var(--app-muted);
  }
  .stTabs [aria-selected="true"] {
    background: var(--app-surface) !important;
    color: var(--app-heading) !important;
    box-shadow: var(--app-shadow);
  }

  /* 탭 패널: 마케팅 / 뉴스레터 섹션 같은 올인원 서피스 (Plus 블록 느낌 참고) */
  .stTabs [data-baseweb="tab-panel"] {
    padding: 1.35rem 1.4rem 1.5rem;
    margin-top: 0.4rem;
    border-radius: 1.05rem;
    border: 1px solid var(--app-border);
    background: linear-gradient(
      165deg,
      rgba(255, 255, 255, 0.88) 0%,
      rgba(248, 250, 252, 0.75) 45%,
      rgba(241, 245, 249, 0.5) 100%
    );
    box-shadow: var(--app-shadow), 0 1px 0 rgba(255, 255, 255, 0.8) inset;
  }

  /* 탭 안 섹션 제목 */
  .stTabs h2,
  .stTabs h3 {
    font-size: 1.05rem !important;
    margin-top: 1.1rem !important;
    margin-bottom: 0.6rem !important;
    padding-bottom: 0.35rem;
    border-bottom: 1px solid rgba(226, 232, 240, 0.85);
  }

  /* 마케팅형 인트로 (HTML로 삽입) */
  .app-mkt-intro {
    display: flex;
    flex-wrap: wrap;
    align-items: baseline;
    gap: 0.5rem 1rem;
    margin-bottom: 1rem;
    padding: 1rem 1.15rem;
    border-radius: 0.85rem;
    background: var(--app-surface);
    border: 1px solid var(--app-border);
    box-shadow: var(--app-shadow);
  }
  .app-mkt-intro__eyebrow {
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.09em;
    text-transform: uppercase;
    color: var(--app-accent-deep);
    background: var(--app-accent-soft);
    padding: 0.28rem 0.65rem;
    border-radius: 9999px;
    white-space: nowrap;
  }
  .app-mkt-intro__hint {
    font-size: 0.9rem;
    color: var(--app-muted);
    line-height: 1.45;
    flex: 1;
    min-width: 200px;
  }

  /* 기사 읽기 구역 리본 */
  .app-mkt-reader-ribbon {
    font-size: 0.78rem;
    font-weight: 500;
    color: var(--app-muted);
    margin: -0.35rem 0 0.85rem 0;
    padding: 0.45rem 0.75rem;
    border-radius: 0.5rem;
    background: rgba(241, 245, 249, 0.9);
    border: 1px solid var(--app-border);
    display: inline-block;
  }

  /* AI 요약: 피처 카드 2열 느낌 */
  .stTabs div[data-testid="stAlert"] {
    box-shadow: var(--app-shadow);
  }

  /* Expander: 콘텐츠 섹션 카드 */
  div[data-testid="stExpander"] {
    border: 1px solid var(--app-border) !important;
    border-radius: 0.85rem !important;
    background: var(--app-panel-bg) !important;
    box-shadow: var(--app-shadow) !important;
    margin-bottom: 0.65rem !important;
    overflow: hidden;
  }
  div[data-testid="stExpander"] details {
    border: none !important;
    background: transparent !important;
  }
  div[data-testid="stExpander"] summary {
    font-weight: 600 !important;
    color: var(--app-heading) !important;
    padding: 0.65rem 0.85rem !important;
  }
  .streamlit-expander {
    border: 1px solid var(--app-border) !important;
    border-radius: 0.85rem !important;
    background: var(--app-panel-bg) !important;
    box-shadow: var(--app-shadow) !important;
    margin-bottom: 0.65rem !important;
  }

  /* 차트(막대): 차분한 카드 프레임 */
  .stTabs [data-testid="stVegaLiteChart"],
  .stTabs [data-testid="stArrowVegaLiteChart"] {
    padding: 0.85rem 0.5rem 0.35rem;
    margin-top: 0.25rem;
    border-radius: 0.75rem;
    background: var(--app-surface);
    border: 1px solid var(--app-border);
    box-shadow: var(--app-shadow);
  }

  /* 슬라이더·숫자 입력 */
  .stTabs .stSlider label,
  .stTabs [data-testid="stNumberInput"] label {
    font-weight: 500 !important;
    color: var(--app-text) !important;
  }

  /* 카드형 bordered 블록 */
  div[data-testid="stVerticalBlockBorderWrapper"] {
    border-radius: 1rem !important;
    border-color: var(--app-border) !important;
    background: rgba(255, 255, 255, 0.72) !important;
    box-shadow: var(--app-shadow) !important;
    transition: border-color 0.15s ease, box-shadow 0.15s ease;
  }
  .stTabs div[data-testid="stVerticalBlockBorderWrapper"]:hover {
    border-color: #cbd5e1 !important;
    box-shadow: 0 4px 14px rgba(15, 23, 42, 0.07) !important;
  }

  /* 지표 카드 */
  [data-testid="stMetric"] {
    background: var(--app-surface) !important;
    padding: 1rem 1.1rem !important;
    border-radius: 0.85rem !important;
    border: 1px solid var(--app-border) !important;
    box-shadow: var(--app-shadow) !important;
  }

  /* 알림 · 인포 박스 */
  div[data-testid="stAlert"] {
    border-radius: 0.85rem !important;
    border: 1px solid var(--app-border) !important;
  }

  /* 사이드바 */
  section[data-testid="stSidebar"] {
    background: linear-gradient(200deg, #ffffff 0%, #f8fafc 48%, #f1f5f9 100%) !important;
    border-right: 1px solid var(--app-border) !important;
  }
  section[data-testid="stSidebar"] .stTextInput input,
  section[data-testid="stSidebar"] .stTextArea textarea {
    border-radius: 0.65rem !important;
    border-color: var(--app-border) !important;
    background: rgba(255, 255, 255, 0.95) !important;
  }

  section[data-testid="stSidebar"] [data-testid="stBaseButton-primary"] button {
    background: linear-gradient(180deg, #14b8a6 0%, var(--app-accent) 100%) !important;
    color: #ffffff !important;
    border: none !important;
    font-weight: 600 !important;
    padding: 0.62rem 1rem !important;
    border-radius: 0.65rem !important;
    box-shadow: 0 1px 3px rgba(15, 118, 110, 0.28);
  }
  section[data-testid="stSidebar"] [data-testid="stBaseButton-primary"] button:hover {
    background: linear-gradient(180deg, var(--app-accent) 0%, var(--app-accent-deep) 100%) !important;
    box-shadow: 0 4px 12px rgba(15, 118, 110, 0.28);
  }
  section[data-testid="stSidebar"] [data-testid="stBaseButton-secondary"] button {
    background: rgba(255, 255, 255, 0.9) !important;
    color: var(--app-text) !important;
    border: 1px solid var(--app-border) !important;
    font-weight: 500 !important;
    border-radius: 0.65rem !important;
    box-shadow: none !important;
  }
  section[data-testid="stSidebar"] [data-testid="stBaseButton-secondary"] button:hover {
    background: var(--app-surface) !important;
    border-color: #cbd5e1 !important;
  }

  /* 메인 영역 기본 버튼 톤 맞춤 */
  .main [data-testid="stBaseButton-primary"] button {
    border-radius: 0.65rem !important;
    font-weight: 600 !important;
    background: linear-gradient(180deg, #14b8a6 0%, var(--app-accent) 100%) !important;
    border: none !important;
  }
  .main [data-testid="stBaseButton-secondary"] button {
    border-radius: 0.65rem !important;
    font-weight: 500 !important;
  }

  div[data-testid="stDecoration"] {
    border-radius: 0.65rem;
  }
"""
st.html(f"<style>{_APP_CSS}</style>")

st.markdown(
    '<div class="app-hero">'
    '<span class="app-hero-badge">뉴스 · 트렌드 · 맞춤 요약</span>'
    '<h1 class="app-hero-title">IT 뉴스 트렌드</h1>'
    "<p class=\"app-hero-lead\">관심사에 맞춰 뉴스를 모아 요약하고, 이어서 배울 강의까지 한 흐름으로 제안합니다.</p>"
    "</div>",
    unsafe_allow_html=True,
)

if not (os.getenv("OPENAI_API_KEY") or "").strip():
    st.error(
        "**OPENAI_API_KEY**가 필요합니다. 로컬은 `.env`, **Streamlit Cloud**는 "
        "**Manage app → Secrets**에 `OPENAI_API_KEY = \"sk-...\"` 형식으로 넣은 뒤 "
        "**Reboot** 하거나 페이지를 새로고침 하세요."
    )
    st.stop()


def _ai_text(msg: object) -> str:
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


_H2 = re.compile(r"^##\s+(.+)$", re.MULTILINE)


def _split_ai_summary_sections(md: str) -> list[tuple[str, str]]:
    """마크다운을 `## 제목` 기준으로 분리 (AI 요약 가독성용)."""
    md = (md or "").strip()
    if not md:
        return []
    matches = list(_H2.finditer(md))
    if not matches:
        return [("전체", md)]
    out: list[tuple[str, str]] = []
    preamble = ""
    if matches[0].start() > 0:
        preamble = md[: matches[0].start()].strip()
    for i, m in enumerate(matches):
        title = m.group(1).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(md)
        body = md[start:end].strip()
        if i == 0 and preamble:
            body = f"{preamble}\n\n{body}".strip()
        out.append((title, body))
    return out


def _lead_text(text: str, *, max_chars: int = 480) -> str:
    text = (text or "").strip()
    if not text:
        return ""
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    for sep in ("\n\n", "다.", ". ", "。", "\n"):
        idx = cut.rfind(sep)
        if idx > max_chars // 2:
            return cut[: idx + len(sep)].strip() + "…"
    return cut.strip() + "…"


def _render_readable_markdown_chunks(md: str, *, batch: int = 2) -> None:
    """문단을 묶어 짧은 블록 단위로 표시 (가독성). HTML div 안 마크다운은 깨질 수 있어 container로 구분."""
    md = (md or "").strip()
    if not md:
        st.caption("내용 없음")
        return
    parts = [p.strip() for p in re.split(r"\n\s*\n", md) if p.strip()]
    if not parts:
        return
    for i in range(0, len(parts), max(1, batch)):
        chunk = "\n\n".join(parts[i : i + batch])
        with st.container(border=True):
            st.markdown(chunk)


def _render_ai_analysis_tab(ai_text: str) -> None:
    """요약 박스 → 접는 상세(expander) → 전체 보기. 긴 글 부담을 줄임."""
    st.markdown(
        '<div class="app-mkt-intro">'
        '<span class="app-mkt-intro__eyebrow">AI 요약</span>'
        '<span class="app-mkt-intro__hint">트렌드·핵심 이슈·기사 맥락·맞춤 조언을 순서대로 확인하세요. 아래 펼침에서 섹션별 전문을 읽을 수 있습니다.</span>'
        "</div>",
        unsafe_allow_html=True,
    )
    st.subheader("AI 분석")
    sections = _split_ai_summary_sections(ai_text)

    if len(sections) > 1:
        trend_body = next((b for t, b in sections if "트렌드" in t), "")
        advice_body = next((b for t, b in sections if "조언" in t), "")
        st.markdown("**먼저 읽기** — 핵심만 짧게 정리했습니다.")
        c_ex1, c_ex2 = st.columns(2, gap="medium")
        with c_ex1:
            st.markdown("###### 요즘 IT 흐름")
            if trend_body:
                st.info(_lead_text(trend_body, max_chars=520))
            else:
                st.caption("트렌드 섹션이 비어 있습니다.")
        with c_ex2:
            st.markdown("###### 내게 맞는 방향")
            if advice_body:
                st.success(_lead_text(advice_body, max_chars=520))
            else:
                st.caption("조언 섹션이 비어 있습니다.")
        st.divider()

    st.caption("아래에서 섹션별 상세를 펼칠 수 있습니다. (긴 글은 문단 사이에 구분선이 있습니다.)")
    if len(sections) > 1:
        for ttl, body in sections:
            is_issues = "핵심" in ttl or "이슈" in ttl
            is_articles = "기사" in ttl
            expanded = bool(is_issues or is_articles)
            with st.expander(ttl, expanded=expanded):
                _render_readable_markdown_chunks(body if body else "_내용 없음_")
    else:
        with st.container(border=True):
            _render_readable_markdown_chunks(sections[0][1] if sections else ai_text)

    st.divider()
    with st.expander("전체 텍스트 한 번에 보기", expanded=False):
        _render_readable_markdown_chunks(ai_text)

    st.download_button(
        "분석 전체 저장 (마크다운)",
        ai_text,
        file_name="it-news-ai-analysis.md",
        mime="text/markdown",
        key="dl_ai_md",
    )


@st.cache_data(ttl=86400, show_spinner=False)
def _inflearn_thumb_cached(course_url: str) -> str | None:
    return fetch_og_image_url(course_url)


def _articles_filtered_sorted(
    articles: list,
    scores: dict | None,
    *,
    min_score: float = 0.0,
) -> list:
    scores = scores or {}
    srt = sorted(
        articles,
        key=lambda x: float(scores.get((x.get("link") or "").strip(), 0.0)),
        reverse=True,
    )
    out: list = []
    for a in srt:
        link = (a.get("link") or "").strip()
        if float(scores.get(link, 0.0)) < min_score:
            continue
        out.append(a)
    return out


def _sync_news_reader_link(filtered: list) -> None:
    """필터에 맞는 목록 안에서만 선택이 유지되도록 링크 동기화."""
    links = [(a.get("link") or "").strip() for a in filtered if (a.get("link") or "").strip()]
    cur = (st.session_state.get(_SIK_NEWS_LINK) or "").strip()
    if cur not in links and links:
        st.session_state[_SIK_NEWS_LINK] = links[0]
    elif not links:
        st.session_state[_SIK_NEWS_LINK] = ""


def _link_button_key(link: str, prefix: str) -> str:
    h = hashlib.md5(link.encode("utf-8"), usedforsecurity=False).hexdigest()[:16]
    return f"{prefix}_{h}"


def _render_article_reader(article: dict) -> None:
    """선택한 기사 제목과 원문 링크만 안내한다."""
    link = (article.get("link") or "").strip()
    title = (article.get("title") or "").strip()
    st.markdown(f"### {title or '제목 없음'}")
    if link:
        st.link_button("원문 열기", link, use_container_width=True, type="primary")
        st.caption(link)
    else:
        st.warning("이 항목에는 URL이 없습니다.")


def _render_article_cards_selectable(
    filtered: list,
    scores: dict | None,
    *,
    limit: int = 6,
    selected_link: str,
) -> None:
    scores = scores or {}
    n = 0
    for a in filtered:
        link = (a.get("link") or "").strip()
        sc = float(scores.get(link, 0.0))
        title = (a.get("title") or "제목 없음").strip()
        snip = (a.get("summary") or "").replace("\n", " ").strip()
        if len(snip) > 160:
            snip = snip[:159] + "…"
        is_on = bool(link and link == (selected_link or "").strip())
        with st.container(border=True):
            h1, h2 = st.columns((4.5, 1.2))
            with h1:
                st.markdown(f"**{n + 1}.** {title}")
                st.caption(snip or "요약 없음")
            with h2:
                st.metric("관련도", f"{sc:.2f}")
            label = "선택됨 · 이어서 읽기" if is_on else "이 기사 읽기"
            if st.button(
                label,
                key=_link_button_key(link or f"_empty_{n}", "npick"),
                type="primary" if is_on else "secondary",
                use_container_width=True,
                disabled=not link,
            ):
                st.session_state[_SIK_NEWS_LINK] = link
                st.rerun()
        n += 1
        if n >= limit:
            break
    if n == 0:
        st.caption("조건에 맞는 기사가 없습니다. 최소 관련도를 낮춰 보세요.")


def _score_chart_df(articles: list, scores: dict | None, *, top_n: int = 12) -> pd.DataFrame:
    scores = scores or {}
    srt = sorted(
        articles,
        key=lambda x: scores.get(x.get("link") or "", 0.0),
        reverse=True,
    )[:top_n]
    labels: list[str] = []
    vals: list[float] = []
    for a in srt:
        t = (a.get("title") or "")[:36]
        if len((a.get("title") or "")) > 36:
            t += "…"
        labels.append(t or "(제목 없음)")
        vals.append(float(scores.get(a.get("link") or "", 0.0)))
    return pd.DataFrame({"기사": labels, "관련도": vals}).set_index("기사")


@st.cache_resource
def _workflow():
    llm = ChatOpenAI(model="gpt-5-mini")
    return build_news_workflow(llm)


with st.sidebar:
    st.markdown("### 관심사")
    role = st.text_input("역할", placeholder="예: 백엔드 엔지니어")
    stack = st.text_input("기술 스택", placeholder="예: Python, FastAPI, PostgreSQL")
    topics = st.text_area("관심 토픽", placeholder="예: LLM, 에이전트, MLOps", height=110)
    st.divider()
    run = st.button(
        "뉴스 요약 실행",
        type="primary",
        use_container_width=True,
    )
    if st.button("저장된 결과 지우기", use_container_width=True, help="화면에 고정된 요약을 비웁니다."):
        st.session_state.pop(_SIK_FULL, None)
        st.session_state.pop(_SIK_PROFILE, None)
        st.session_state.pop(_SIK_NEWS_LINK, None)
        st.rerun()

_user_profile = {"role": role.strip(), "stack": stack.strip(), "topics": topics.strip()}
_profile_ok = bool(role.strip() or stack.strip() or topics.strip())

if run and not _profile_ok:
    st.warning("역할·스택·토픽 중 최소 한 가지는 입력해 주세요.")
    st.stop()

full: dict | None = None
_did_fresh_run = False

if run and _profile_ok:
    graph = _workflow()
    initial = {
        "messages": [],
        "user_profile": _user_profile,
        "feed_urls": list(config.DEFAULT_FEEDS),
        "articles": [],
        "article_scores": {},
        "max_match_score": 0.0,
        "scope_expanded": False,
        "branch_events": [],
        "tool_trace": [],
    }
    status = st.status("진행 중…", expanded=False)
    try:
        for snap in graph.stream(initial, stream_mode="values"):
            full = snap
    except Exception as e:
        status.update(label="오류", state="error")
        st.exception(e)
        st.stop()
    status.update(label="완료", state="complete")
    if full:
        st.session_state[_SIK_FULL] = full
        st.session_state[_SIK_PROFILE] = dict(_user_profile)
    _did_fresh_run = True
elif _SIK_FULL in st.session_state:
    full = st.session_state[_SIK_FULL]
else:
    st.info("왼쪽에서 관심사를 입력한 뒤 **뉴스 요약 실행**을 누르세요.")
    st.stop()

if _did_fresh_run:
    st.toast("분석을 마쳤습니다.", icon="✅")

if not full:
    st.error("결과가 없습니다.")
    st.stop()

_saved = st.session_state.get(_SIK_PROFILE)
if _saved and _saved != _user_profile:
    st.caption("관심사 입력이 저장된 분석과 다릅니다. 새 값으로 반영하려면 **뉴스 요약 실행**을 다시 누르세요.")

articles = list(full.get("articles") or [])
scores = full.get("article_scores") or {}
expanded = bool(full.get("scope_expanded"))
tl = react_tool_timeline(full.get("messages"))

col_main, col_side = st.columns((3.2, 1), gap="large")

with col_side:
    st.subheader("지표")
    st.metric("조회 기사", len(articles))
    st.metric("최고 관련도", f"{float(full.get('max_match_score') or 0):.3f}")
    st.metric("피드 수", len(full.get("feed_urls") or []))
    st.metric("도구 호출", len(tl))
    st.caption(
        "추가 피드로 한 번 더 수집했습니다." if expanded else "첫 수집으로 진행했습니다."
    )

with col_main:
    branch_list = full.get("branch_events") or []
    if branch_list:
        st.subheader("안내")
        for ev in branch_list:
            st.info(ev.get("detail_ko", str(ev)))

    _topics_first = (topics.strip().split(",")[0] if topics.strip() else "프로그래밍").strip()[:40]
    _search_inf = inflearn_search_url(_topics_first)

    msgs = list(full.get("messages") or [])
    last_ai = next(
        (m for m in reversed(msgs) if isinstance(m, AIMessage) and m.content),
        None,
    )
    ai_text = str(last_ai.content) if last_ai and last_ai.content else ""

    tab_news, tab_inflearn, tab_ai = st.tabs(
        ["기사 & 관련도", "인프런 학습", "AI 요약"]
    )

    with tab_news:
        st.markdown(
            '<div class="app-mkt-intro">'
            '<span class="app-mkt-intro__eyebrow">기사 · 관련도</span>'
            '<span class="app-mkt-intro__hint"><strong>이 기사 읽기</strong>로 선택하면 아래에서 <strong>원문 링크</strong>로 바로 이동할 수 있습니다. 슬라이더로 목록을 좁힐 수 있습니다.</span>'
            "</div>",
            unsafe_allow_html=True,
        )
        max_sc = max((float(v) for v in scores.values()), default=0.0)
        slider_hi = max(float(max_sc), 0.06)
        f1, f2 = st.columns([1.2, 1.2])
        with f1:
            min_sc = st.slider(
                "최소 관련도",
                0.0,
                slider_hi,
                0.0,
                step=0.005,
                help="이 값보다 낮은 기사는 카드·읽기·차트에서 제외합니다.",
            )
        with f2:
            chart_top = st.number_input(
                "차트 상위 N건",
                min_value=5,
                max_value=20,
                value=12,
                step=1,
            )

        filtered = _articles_filtered_sorted(articles, scores, min_score=min_sc)
        _sync_news_reader_link(filtered)
        sel_link = (st.session_state.get(_SIK_NEWS_LINK) or "").strip()
        sel_article = next(
            (a for a in articles if (a.get("link") or "").strip() == sel_link),
            None,
        )

        st.subheader("상위 기사")
        _render_article_cards_selectable(
            filtered,
            scores,
            limit=6,
            selected_link=sel_link,
        )

        st.subheader("기사 읽기")
        st.markdown(
            '<p class="app-mkt-reader-ribbon">선택한 기사 · 원문 페이지로 이동</p>',
            unsafe_allow_html=True,
        )
        if sel_article and sel_link:
            with st.container(border=True):
                _render_article_reader(sel_article)
        else:
            st.info("위에서 기사를 선택하면 여기에 원문 링크가 표시됩니다.")

        st.subheader("관련도 차트")
        chart_df = _score_chart_df(filtered, scores, top_n=int(chart_top))
        if chart_df.empty:
            st.caption("표시할 데이터가 없습니다.")
        else:
            st.bar_chart(chart_df, horizontal=True)

        b1, b2 = st.columns(2)
        with b1:
            srt_links = sorted(
                articles,
                key=lambda x: float(scores.get(x.get("link") or "", 0.0)),
                reverse=True,
            )
            if srt_links:
                top_link = (srt_links[0].get("link") or "").strip()
                if top_link:
                    st.link_button("1위 기사 원문", top_link)
        with b2:
            st.link_button("인프런에서 검색", _search_inf)

    with tab_inflearn:
        matches = match_curated_courses(
            full.get("user_profile") or _user_profile,
            articles,
            max_courses=5,
        )
        if matches:
            for c, _ in matches:
                if not isinstance(c, dict):
                    continue
                tit = str(c.get("title") or "")
                url = str(c.get("url") or "")
                thumb = (c.get("thumbnail") or "").strip() or (
                    _inflearn_thumb_cached(url) if url else None
                )
                c1, c2 = st.columns([0.35, 0.65])
                with c1:
                    if thumb:
                        try:
                            st.image(thumb, use_container_width=True)
                        except Exception:
                            st.caption("썸네일 로드 실패")
                    else:
                        st.caption("썸네일 없음")
                with c2:
                    st.markdown(f"**{tit}**")
                    if url:
                        st.link_button("강의 보기", url)
                    tags = c.get("tags") or []
                    if isinstance(tags, list) and tags:
                        st.caption(" · ".join(str(t) for t in tags[:8]))
                st.divider()
        else:
            st.info("프로필과 맞는 추천 강의가 없습니다. 아래 검색으로 찾아보세요.")
        st.markdown("**검색**")
        for label, surl in inflearn_search_link_pairs(
            full.get("user_profile") or _user_profile,
            max_links=3,
        ):
            st.link_button(f"«{label}» 검색", surl)
        st.link_button("인프런 강의 목록", "https://www.inflearn.com/courses")

    with tab_ai:
        if ai_text:
            _render_ai_analysis_tab(ai_text)
        else:
            st.warning("요약을 가져오지 못했습니다.")
