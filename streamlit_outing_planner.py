"""
부산 금정구 카페 나들이 ReAct 플래너 — Streamlit UI

실행 (프로젝트 루트에서, venv 활성화 후):
  streamlit run streamlit_outing_planner.py
"""

from __future__ import annotations

import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from outing_planner.core import OutingPlannerAgent

_ROOT = Path(__file__).resolve().parent
# notebooks/.env 다음 루트 .env (중복 키는 루트가 우선)
_nb = _ROOT / "notebooks" / ".env"
_rt = _ROOT / ".env"
if _nb.exists():
    load_dotenv(_nb)
if _rt.exists():
    load_dotenv(_rt, override=True)


def _api_key_configured() -> bool:
    key = os.getenv("OPENAI_API_KEY")
    return bool(key and key.strip())


@st.cache_resource
def _get_llm() -> ChatOpenAI:
    return ChatOpenAI(model="gpt-5-mini")


def _init_session() -> None:
    if "planner" not in st.session_state:
        st.session_state.planner = OutingPlannerAgent(_get_llm())
    if "ui_messages" not in st.session_state:
        st.session_state.ui_messages = []


def main() -> None:
    st.set_page_config(
        page_title="금정구 카페 나들이 플래너",
        page_icon="☕",
        layout="wide",
    )
    st.title("☕ 금정구 카페 나들이 플래너")
    st.caption("ReAct 에이전트 · 공휴일 / 선선한 날씨(금정구) / 카페 후보 도구")

    with st.sidebar:
        st.subheader("연결 상태")
        if _api_key_configured():
            st.success("OPENAI_API_KEY: 설정됨")
        else:
            st.error("OPENAI_API_KEY: 없음")
            st.markdown(
                "`notebooks/.env` 또는 프로젝트 루트 `.env`에  \n"
                "`OPENAI_API_KEY=...` 를 넣고 **앱을 새로고침** 하세요."
            )

        st.subheader("도구 요약")
        st.markdown(
            """
            - **공휴일**: `get_nearest_korean_public_holiday`
            - **날씨**: `get_cool_weather_days_geumjeong` (Open-Meteo)
            - **카페**: `find_cafes_near_geumjeong_gu` (데모 목록)
            """
        )
        if st.button("대화 초기화", type="secondary"):
            if "planner" in st.session_state:
                st.session_state.planner.reset()
            st.session_state.ui_messages = []
            st.rerun()

    if not _api_key_configured():
        st.warning(
            "API 키가 없어 응답을 생성할 수 없습니다. 위 사이드바 안내에 따라 `.env`를 만든 뒤 다시 열어 주세요."
        )

    _init_session()
    planner: OutingPlannerAgent = st.session_state.planner

    for role, text in st.session_state.ui_messages:
        with st.chat_message(role):
            st.markdown(text)

    def _run_turn(user_text: str) -> None:
        user_text = user_text.strip()
        if not user_text:
            return
        st.session_state.ui_messages.append(("user", user_text))
        try:
            reply = planner.chat(user_text)
        except Exception as e:
            reply = f"오류: {e}"
        st.session_state.ui_messages.append(("assistant", reply))

    # 방법 A: 하단 고정 채팅 (일부 브라우저/내장 뷰에서 안 보일 수 있음)
    if _api_key_configured():
        if chat_prompt := st.chat_input(
            "메시지 입력 후 Enter (예: 오늘 기준 가까운 공휴일에 맞춰 카페 코스 짜 줘)"
        ):
            _run_turn(chat_prompt)
            st.rerun()

    # 방법 B: 폼 전송 — 입력창이 안 보이거나 Enter가 먹히지 않을 때 사용
    st.divider()
    st.markdown("**메시지 보내기** — 위 입력이 안 되면 여기에 적고 **전송**을 누르세요.")
    with st.form("message_form", clear_on_submit=True):
        body = st.text_area(
            "질문",
            height=100,
            placeholder="예: 금정구에서 공휴일에 갈 만한 카페랑 선선한 날 알려줘",
            label_visibility="collapsed",
        )
        submitted = st.form_submit_button("전송", type="primary")

    if submitted and _api_key_configured():
        _run_turn(body)
        st.rerun()
    elif submitted and not _api_key_configured():
        st.error("먼저 OPENAI_API_KEY를 설정해 주세요.")


# Streamlit은 스크립트를 반복 실행하므로 보통 이 형태로 둡니다.
main()
