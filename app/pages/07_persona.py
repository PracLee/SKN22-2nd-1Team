# app/pages/02_persona_generator.py

from __future__ import annotations

import streamlit as st
import pandas as pd

from service.session_probability_service import (
    SessionProbabilityService,
    SessionPredictionResult,
)
from ui.header import render_header

# ======================================
# 공통: 서비스 / 스타일 초기화
# ======================================

render_header()

st.set_page_config(
    page_title="가상 고객 페르소나 생성기",
    page_icon="🧑‍💻",
    layout="wide",
)

# 카드 스타일 (세션 페이지랑 맞춰 사용)
st.markdown(
    """
    <style>
    .persona-card {
        padding: 1.5rem;
        border-radius: 1rem;
        color: white;
        margin-top: 1rem;
    }
    .high-prob {
        background: linear-gradient(135deg, #16a34a, #22c55e);
    }
    .medium-prob {
        background: linear-gradient(135deg, #eab308, #facc15);
        color: #1f2933;
    }
    .low-prob {
        background: linear-gradient(135deg, #b91c1c, #ef4444);
    }
    .sub-text {
        font-size: 0.9rem;
        opacity: 0.9;
    }
    .persona-tag {
        display: inline-block;
        padding: 0.25rem 0.6rem;
        border-radius: 999px;
        font-size: 0.75rem;
        margin-right: 0.25rem;
        background-color: rgba(148, 163, 184, 0.2);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_session_probability_service() -> SessionProbabilityService:
    """
    세션 구매확률 서비스 (모델/어댑터 캐시)
    """
    return SessionProbabilityService(global_avg_purchase_prob=0.15)


service = get_session_probability_service()

# ======================================
# 페이지 타이틀
# ======================================

st.title("🧬 가상 고객 페르소나 생성기")
st.caption(
    "간단한 옵션만 선택하면, 해당 유형의 고객 세션을 자동으로 생성하고 "
    "모델이 예측한 구매 확률과 설명을 보여줍니다."
)

# 레이아웃
left, right = st.columns([0.9, 1.1])


# ======================================
# 1. 사이드바(또는 좌측)에서 페르소나 옵션 선택
# ======================================

with left:
    st.subheader("1️⃣ 페르소나 조건 선택")

    st.markdown("##### 방문 유형")
    visitor_type_label = st.radio(
        "어떤 유형의 방문자인가요?",
        options=["신규 방문자", "재방문 고객"],
        horizontal=True,
    )
    visitor_type = "New_Visitor" if visitor_type_label == "신규 방문자" else "Returning_Visitor"

    st.markdown("##### 세션 의도")
    intent_label = st.radio(
        "이번 세션의 주된 목적은?",
        options=["정보 수집형", "구매 직전"],
        horizontal=True,
    )

    st.markdown("##### 요일 / 분위기")
    weekend_label = st.radio(
        "언제 방문한 세션인가요?",
        options=["평일", "주말"],
        horizontal=True,
    )
    weekend = weekend_label == "주말"

    st.markdown("##### 사용할 모델 기준")
    model_strategy_label = st.selectbox(
        "어떤 성능 기준으로 선택된 모델을 사용할까요?",
        options=[
            "ROC-AUC 기준 베스트",
            "PR-AUC 기준 베스트",
        ],
        index=0,
    )
    strategy_map = {
        "ROC-AUC 기준 베스트": "roc_auc",
        "PR-AUC 기준 베스트": "pr_auc",
    }
    selected_strategy = strategy_map[model_strategy_label]

    st.markdown("---")

    generate_btn = st.button("✨ 페르소나 세션 생성 & 구매 확률 예측", type="primary")


# ======================================
# 2. 페르소나 → 세션 데이터 생성 로직
# ======================================

def generate_persona_session(
    visitor_type: str,
    intent_label: str,
    weekend: bool,
) -> tuple[pd.DataFrame, str, str]:
    """
    선택한 옵션(방문 유형, 의도, 주말 여부)에 따라
    UCI Online Shoppers 스타일의 세션 feature를 구성.

    return:
        - session_df: 1 row DataFrame
        - persona_name: "재방문 · 구매 직전 · 주말 형" 같은 짧은 이름
        - narrative: 자연어 설명
    """

    # 기본값 (약간의 "평균적인" 세션 느낌)
    base = dict(
        row_id=0,
        Administrative=1,
        Administrative_Duration=40.0,
        Informational=1,
        Informational_Duration=60.0,
        ProductRelated=10,
        ProductRelated_Duration=300.0,
        BounceRates=0.3,
        ExitRates=0.3,
        PageValues=10.0,
        SpecialDay=0.0,
        Month="Nov",            # 데모용으로 11월 고정
        OperatingSystems=1,     # 데스크톱
        Browser=1,              # 주요 브라우저
        Region=1,               # 기본 Region
        TrafficType=2,          # 기본 유입 채널 코드
        VisitorType=visitor_type,
        Weekend=weekend,
    )

    # 의도에 따라 행동 패턴 세팅
    if intent_label == "정보 수집형":
        base.update(
            dict(
                Informational=3,
                Informational_Duration=4 * 60.0,    # 4분 정도 정보 페이지 탐색
                ProductRelated=8,
                ProductRelated_Duration=5 * 60.0,
                PageValues=3.0,
                BounceRates=0.45,
                ExitRates=0.4,
                SpecialDay=0.0,
            )
        )
    elif intent_label == "구매 직전":
        base.update(
            dict(
                Administrative=2,
                Administrative_Duration=2 * 60.0,   # 로그인/주문 확인 등
                Informational=1,
                Informational_Duration=30.0,
                ProductRelated=25,
                ProductRelated_Duration=15 * 60.0,  # 상품 상세를 오래 봄
                PageValues=80.0,                    # 장바구니/결제 페이지 진입
                BounceRates=0.05,
                ExitRates=0.15,
                SpecialDay=0.4,                     # 이벤트/기념일 근처
            )
        )

    # 방문 유형에 따른 미세 조정
    if visitor_type == "Returning_Visitor":
        base["TrafficType"] = 2       # 예: 직접/북마크 유입
        base["Region"] = 1
        base["BounceRates"] = min(base["BounceRates"], 0.25)
    else:  # New_Visitor
        base["TrafficType"] = 1       # 예: 광고/검색 유입
        base["Region"] = 3
        base["BounceRates"] = max(base["BounceRates"], 0.35)

    # 평일/주말에 따른 미세 조정
    if weekend:
        base["Weekend"] = True
        base["ProductRelated_Duration"] *= 1.2    # 더 오래 머무는 경향
        base["Informational_Duration"] *= 1.1
    else:
        base["Weekend"] = False
        base["ProductRelated_Duration"] *= 0.9
        base["Informational_Duration"] *= 0.9

    # 페르소나 이름 & 설명 생성
    vt_kor = "신규 방문자" if visitor_type == "New_Visitor" else "재방문 고객"
    weekend_kor = "주말" if weekend else "평일"

    persona_name = f"{vt_kor} · {intent_label} · {weekend_kor}형"

    intent_desc = (
        "상품을 비교·탐색하면서 정보를 폭넓게 수집하는 고객"
        if intent_label == "정보 수집형"
        else "이미 구매 결정을 거의 마무리하고 결제 단계에 가까운 고객"
    )
    time_desc = (
        "여유 있는 주말에 천천히 둘러보는 패턴"
        if weekend
        else "평일 짧은 시간에 빠르게 살펴보는 패턴"
    )

    narrative = (
        f"이 페르소나는 **{vt_kor}** 이며, **{intent_desc}** 입니다. "
        f"또한 **{weekend_kor} 방문 세션**으로, {time_desc}을 가정합니다."
    )

    session_df = pd.DataFrame({k: [v] for k, v in base.items()})
    return session_df, persona_name, narrative


def risk_band_to_css_class(risk_band: str) -> str:
    if risk_band == "high":
        return "high-prob"
    elif risk_band == "medium":
        return "medium-prob"
    return "low-prob"


# ======================================
# 3. 결과 영역
# ======================================

with right:
    st.subheader("2️⃣ 생성된 페르소나 & 예측 결과")

    if generate_btn:
        # 1) 페르소나 기반 세션 생성
        persona_df, persona_name, narrative = generate_persona_session(
            visitor_type=visitor_type,
            intent_label=intent_label,
            weekend=weekend,
        )

        # 2) 모델 예측
        try:
            result: SessionPredictionResult = service.predict_session(
                persona_df,
                strategy=selected_strategy,
            )
        except Exception as e:
            st.error(f"예측 중 오류가 발생했습니다: {e}")
            st.stop()

        css_class = risk_band_to_css_class(result.risk_band)

        # 3) 상단 카드 (요약)
        st.markdown(
            f"""
            <div class="persona-card {css_class}">
                <div style="font-size:0.9rem; margin-bottom:0.25rem;">
                    <span class="persona-tag">{visitor_type_label}</span>
                    <span class="persona-tag">{intent_label}</span>
                    <span class="persona-tag">{weekend_label}</span>
                </div>
                <h3>🧬 {persona_name}</h3>
                <p class="sub-text" style="margin-top:0.5rem;">{narrative}</p>
                <hr style="border: none; border-top: 1px solid rgba(248,250,252,0.25); margin: 0.75rem 0;" />
                <p style="font-size:1.1rem; font-weight:600; margin-bottom:0.25rem;">
                    🧮 모델 예측 구매 확률: {result.probability * 100:.1f}%
                </p>
                <p class="sub-text">{result.status_label}</p>
                <p class="sub-text">{result.compare_text}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # 4) 설명 영역
        with st.expander("🔍 모델 관점에서 본 이 페르소나 (설명 보기)", expanded=True):
            st.markdown("**행동 특성에 대한 해석**")
            for r in result.reasons:
                st.markdown(f"- {r}")
            st.markdown("---")
            st.markdown(f"**평균 대비 요약:** {result.average_text}")

        # 5) 실제로 모델에 들어간 feature 확인용
        with st.expander("📁 생성된 세션 feature (디버깅/교육용)", expanded=False):
            st.dataframe(persona_df)
    else:
        st.info(
            "좌측에서 방문 유형 · 세션 의도 · 요일을 선택하고 "
            "**'✨ 페르소나 세션 생성 & 구매 확률 예측'** 버튼을 눌러보세요."
        )
