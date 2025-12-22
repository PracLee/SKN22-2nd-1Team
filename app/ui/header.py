import streamlit as st
from pathlib import Path
import sys

# streamlit run app/app.py -> MAIN_SCRIPT = .../app/app.py
MAIN_SCRIPT = Path(sys.argv[0]).resolve()
APP_DIR = MAIN_SCRIPT.parent  # .../app

PAGE_MAP = {
    "홈": "app.py",  # 홈은 pages로 빼는 게 switch_page 안정적입니다.
    "세션 구매 확률 계산기": "pages/01_session_prob.py",
    "What-if 시뮬레이터": "pages/02_what_if.py",
    "채널 효과 분석": "pages/03_channel_effect.py",
    "고위험 이탈 탐지": "pages/04_churn_risk.py",
    "EDA 대시보드": "pages/05_eda.py",
    "모델 해석 (XAI)": "pages/06_xai.py",
    "가상 고객 페르소나": "pages/07_persona.py",
    "실험 모드 (A/B Test)": "pages/08_ab_test.py",
    "모델 성능 비교": "pages/09_model_compare.py",
    "마케팅 액션 추천": "pages/10_marketing_action.py",
}

ITEMS = [
    {"tab": "홈", "short": "홈", "icon": "🏠"},
    {"tab": "세션 구매 확률 계산기", "short": "세션 구매확률", "icon": "🧮"},
    {"tab": "What-if 시뮬레이터", "short": "What-if", "icon": "🧪"},
    {"tab": "채널 효과 분석", "short": "채널 효과", "icon": "📣"},
    {"tab": "고위험 이탈 탐지", "short": "고위험 이탈", "icon": "🚨"},
    {"tab": "EDA 대시보드", "short": "EDA", "icon": "📊"},
    {"tab": "모델 해석 (XAI)", "short": "모델 해석", "icon": "🧠"},
    {"tab": "가상 고객 페르소나", "short": "고객 페르소나", "icon": "👤"},
    {"tab": "실험 모드 (A/B Test)", "short": "A/B Test", "icon": "🧩"},
    {"tab": "모델 성능 비교", "short": "모델 비교", "icon": "⚖️"},
    {"tab": "마케팅 액션 추천", "short": "마케팅 액션", "icon": "🎯"},
]


def _inject_nav_css():
    if st.session_state.get("_nav_css_done"):
        return
    st.session_state["_nav_css_done"] = True

    st.markdown(
        """
<style>

/* 네비게이션 버튼(전체 st.button)에 적용됩니다. 페이지 내 다른 버튼도 동일 톤이면 오히려 통일감이 생깁니다. */
div[data-testid="stButton"] > button {
  height: 64px;
  border-radius: 16px;
  border: 1px solid rgba(255,255,255,0.14);
  background: rgba(255,255,255,0.03);
  font-weight: 650;
  letter-spacing: -0.2px;
  white-space: pre-line;       /* \n 줄바꿈 반영 */
  line-height: 1.15;
  padding: 10px 12px;
  transition: transform .08s ease, border-color .08s ease, box-shadow .08s ease;
}

div[data-testid="stButton"] > button:hover {
  border-color: rgba(255,255,255,0.28);
  transform: translateY(-1px);
  box-shadow: 0 10px 28px rgba(0,0,0,0.28);
}

/* primary 버튼(활성 탭)은 좀 더 강조 */
div[data-testid="stButton"] > button[kind="primary"] {
  border-color: rgba(255,255,255,0.35);
  box-shadow: 0 10px 32px rgba(0,0,0,0.35);
}

/* 헤더 아래 구분선 */
.nav-divider {
  margin-top: 10px;
  margin-bottom: 0px;
  height: 1px;
  background: linear-gradient(90deg, transparent, rgba(255,255,255,0.18), transparent);
}
</style>
        """,
        unsafe_allow_html=True,
    )


def _go(tab: str):
    rel = Path(PAGE_MAP[tab]).as_posix()
    target = (APP_DIR / rel).resolve()

    if not target.exists():
        st.error(f"페이지 파일을 찾을 수 없습니다: {target}")
        st.write("APP_DIR:", str(APP_DIR))
        st.write("rel:", rel)
        return

    st.session_state.active_tab = tab
    st.switch_page(rel)


def render_header(per_row: int = 6):
    _inject_nav_css()

    if "active_tab" not in st.session_state:
        st.session_state.active_tab = "홈"

    # 네비게이션(2줄 타일)
    for r in range(0, len(ITEMS), per_row):
        row = ITEMS[r : r + per_row]
        cols = st.columns(len(row), gap="small")
        for col, it in zip(cols, row):
            with col:
                is_active = st.session_state.active_tab == it["tab"]
                btn_type = "primary" if is_active else "secondary"

                # 아이콘/텍스트 2줄 고정
                label = f"{it['icon']}\n{it['short']}"

                if st.button(
                    label,
                    key=f"nav_{it['tab']}",
                    help=it["tab"],
                    use_container_width=True,
                    type=btn_type,
                ):
                    _go(it["tab"])

    st.markdown('<div class="nav-divider"></div>', unsafe_allow_html=True)
