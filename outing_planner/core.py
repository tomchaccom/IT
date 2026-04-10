"""8번 실습과 동일: 공휴일·금정구 선선한 날씨·카페 도구 + ReAct 플래너."""

from __future__ import annotations

import json
import urllib.request
from datetime import date, timedelta
from urllib.parse import urlencode

from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

try:
    import holidays
except ImportError:
    holidays = None


@tool
def get_nearest_korean_public_holiday(reference_date: str = "") -> str:
    """대한민국 기준으로 reference_date(YYYY-MM-DD) 이후 가장 가까운 공휴일 날짜와 이름을 알려줍니다. 빈 문자열이면 오늘 날짜 기준."""
    if holidays is None:
        return "공휴일 계산에 `holidays` 패키지가 필요합니다. 터미널에서: pip install holidays"
    ref = reference_date.strip()[:10]
    if ref:
        try:
            start = date.fromisoformat(ref)
        except ValueError:
            return f"날짜 형식 오류: {reference_date!r}. YYYY-MM-DD 로 넣어 주세요."
    else:
        start = date.today()
    years = range(start.year, start.year + 3)
    kr = holidays.SouthKorea(years=years)
    for i in range(800):
        d = start + timedelta(days=i)
        if d in kr:
            names = kr[d]
            if not isinstance(names, list):
                names = [names]
            label = ", ".join(str(n) for n in names)
            return f"가장 가까운 공휴일은 {d.isoformat()} ({label}) 입니다."
    return "가까운 공휴일을 찾지 못했습니다."


@tool
def get_cool_weather_days_geumjeong(days_ahead: int = 14) -> str:
    """부산 금정구 인근(위도·경도 고정) 일기예보에서 '선선한' 날을 골라 줍니다. 기준: 일 최고기온 18~28°C (Open-Meteo API, 인터넷 필요)."""
    n = max(1, min(int(days_ahead), 16))
    lat, lon = 35.2431, 129.0882
    q = urlencode(
        {
            "latitude": lat,
            "longitude": lon,
            "daily": "temperature_2m_max,temperature_2m_min",
            "timezone": "Asia/Seoul",
            "forecast_days": n,
        }
    )
    url = f"https://api.open-meteo.com/v1/forecast?{q}"
    try:
        with urllib.request.urlopen(url, timeout=12) as resp:
            data = json.loads(resp.read().decode())
    except Exception as e:
        return f"날씨 API 오류: {e}"
    times = data.get("daily", {}).get("time", [])
    tmax = data.get("daily", {}).get("temperature_2m_max", [])
    tmin = data.get("daily", {}).get("temperature_2m_min", [])
    picked = []
    for i, ds in enumerate(times):
        if i >= len(tmax) or i >= len(tmin):
            break
        mx, mn = tmax[i], tmin[i]
        if mx is None or mn is None:
            continue
        if 18.0 <= float(mx) <= 28.0:
            picked.append(f"{ds}: 최고 {mx:.0f}°C / 최저 {mn:.0f}°C")

    head = "부산 금정구 인근 예보 중 선선한 날(최고 18~28°C):\n"
    if not picked:
        return head + "(해당 기간에 기준을 만족하는 날이 없습니다. 기간을 늘려 다시 확인해 보세요.)"
    return head + "\n".join(picked)


@tool
def find_cafes_near_geumjeong_gu(keyword: str = "") -> str:
    """부산 금정구·부산대·장산역·장전역·온천천 근처 카페 후보(데모 목록). keyword에 '뷰','베이커리','브런치' 등을 주면 비슷한 곳 위주로 골라 줍니다."""
    cafes = [
        ("카페 매미사거리", "장전동", "창가 좌석, 커피·케이크, 조용한 편"),
        ("모모스 커피 부산대", "장전동(부산대 정문)", "에스프레소·핸드드립, 학생·방문객 많음"),
        ("어느멋진날", "금정구 두구동", "감성 인테리어·디저트, 사진 찍기 좋음"),
        ("테라로사 장산", "장산동", "넓은 시트, 브런치·베이커리, 가족 단위"),
        ("아날로그 가든", "구서동", "정원 느낌 테라스·산책 후 휴식"),
        ("투썸플레이스 금정본점 인근 독립 카페거리", "서동", "동네 골목 소형 카페 여러 곳"),
        ("온천천 산책로 카페거리", "온천천 인근", "산책→카페 코스 짜기 좋음"),
    ]
    k = (keyword or "").strip().lower()
    lines = []
    for name, area, desc in cafes:
        blob = f"{name} {area} {desc}".lower()
        if not k or k in blob or any(tok in blob for tok in k.split()):
            lines.append(f"- {name} ({area}): {desc}")
    if not lines:
        lines = [f"- {c[0]} ({c[1]}): {c[2]}" for c in cafes]
    return "금정구 근처 카페 후보(데모 데이터, 실제 영업시간은 지도 앱에서 확인):\n" + "\n".join(lines)


outing_tools = [
    get_nearest_korean_public_holiday,
    get_cool_weather_days_geumjeong,
    find_cafes_near_geumjeong_gu,
]

OUTING_SYSTEM_PROMPT = """당신은 부산 금정구 근처에서 '공휴일에 맞춰' 카페 나들이 계획을 짜 주는 플래너입니다.
규칙:
- 가장 가까운 공휴일이 언제인지 알아야 하면 반드시 get_nearest_korean_public_holiday 를 사용합니다(날짜가 정해져 있으면 reference_date에 YYYY-MM-DD).
- 그날이나 그 주의 날씨가 선선한지 보려면 get_cool_weather_days_geumjeong 를 사용합니다.
- 구체적인 카페 후보는 find_cafes_near_geumjeong_gu 로만 제시합니다(직접 지어내지 마세요).
- 도구 결과를 바탕으로 한국어로 친근하게 요약하고 출발 전 확인할 점(영업시간, 예약)을 한두 줄 덧붙입니다."""


class OutingPlannerAgent:
    """공휴일·날씨·카페 도구를 쓰는 ReAct 에이전트 (히스토리 유지)."""

    def __init__(self, llm: ChatOpenAI):
        self._llm = llm
        self.agent = create_react_agent(
            llm,
            outing_tools,
            prompt=OUTING_SYSTEM_PROMPT,
        )
        self.history: list = []

    def reset(self) -> None:
        self.history = []

    def chat(self, user_input: str) -> str:
        self.history.append(HumanMessage(content=user_input))
        result = self.agent.invoke({"messages": self.history})
        ai_message = result["messages"][-1]
        self.history.append(ai_message)
        return ai_message.content
