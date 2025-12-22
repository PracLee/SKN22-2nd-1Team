import sys, os 

# pages → app
APP_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

import streamlit as st
from ui.header import render_header

render_header()
st.set_page_config(page_title="ab_test", layout="wide")

import pandas as pd
import joblib
import altair as alt
import numpy as np

# -------------------------------
# 데이터 / 모델 경로
# -------------------------------
BASE_DIR = os.path.abspath(os.path.join(APP_DIR, ".."))
TRAIN_PATH = os.path.join(BASE_DIR, "data/processed/train.csv")
TEST_PATH = os.path.join(BASE_DIR, "data/processed/test.csv")
MODEL_PATH = os.path.join(BASE_DIR, "app/artifacts/best_balancedrf_pipeline.joblib")

# -------------------------------
# 데이터 / 모델 로드
# -------------------------------
X_train = pd.read_csv(TRAIN_PATH)
X_test = pd.read_csv(TEST_PATH)

model_dict = joblib.load(MODEL_PATH)
pipeline = model_dict["pipeline"]
best_threshold = model_dict["best_threshold"]

# -------------------------------
# 샘플 고정 (Streamlit rerun 방지)
# -------------------------------
if "base_idx" not in st.session_state:
    st.session_state["base_idx"] = int(np.random.choice(X_test.index, size=1)[0])

base_idx = st.session_state["base_idx"]
X_base = X_test.loc[[base_idx]]  # 1행 DataFrame 고정

# -------------------------------
# 페이지 타이틀
# -------------------------------
st.title("Scenario 비교 시뮬레이터")

indicator_desc = {
    "BounceRates": "단일 페이지 방문 후 이탈한 세션의 비율",
    "ExitRates": "해당 페이지에서 세션이 종료된 비율",
    "PageValues": "페이지 방문이 전환에 기여한 기여도 (0~100 기준)",
    "ProductRelated_Duration": "상품 관련 페이지 체류 시간(초)",
}

direction_hint = {
    "BounceRates": -1,
    "ExitRates": -1,
    "PageValues": 1,
    "ProductRelated_Duration": 1,
}

cols = list(indicator_desc.keys())
scenario_a = {}
scenario_b = {}

pagevalues_max = X_train["PageValues"].max()

# 좌우 컬럼 생성
col_left, col_right = st.columns(2)

# -------------------------------
# 좌측: Scenario 입력
# -------------------------------
with col_left:
    st.subheader("Scenario A")
    for col in cols:
        if col == "PageValues":
            min_val, max_val = 0.0, 100.0
            default_val = float(X_base[col].iloc[0]) / pagevalues_max * 100
            st.markdown(f"**{col}** — {indicator_desc[col]}")
            slider_val = st.slider("", min_val, max_val, default_val, step=1.0, key=f"A_{col}")
            scenario_a[col] = float(slider_val) / 100 * pagevalues_max
        else:
            min_val, max_val = float(X_train[col].min()), float(X_train[col].max())
            st.markdown(f"**{col}** — {indicator_desc[col]}")
            slider_val = st.slider("", min_val, max_val, float(X_base[col].iloc[0]), key=f"A_{col}")
            scenario_a[col] = float(slider_val)

    st.subheader("Scenario B")
    for col in cols:
        if col == "PageValues":
            min_val, max_val = 0.0, 100.0
            default_val = float(X_base[col].iloc[0]) / pagevalues_max * 100
            st.markdown(f"**{col}** — {indicator_desc[col]}")
            slider_val = st.slider("", min_val, max_val, default_val, step=1.0, key=f"B_{col}")
            scenario_b[col] = float(slider_val) / 100 * pagevalues_max
        else:
            min_val, max_val = float(X_train[col].min()), float(X_train[col].max())
            st.markdown(f"**{col}** — {indicator_desc[col]}")
            slider_val = st.slider("", min_val, max_val, float(X_base[col].iloc[0]), key=f"B_{col}")
            scenario_b[col] = float(slider_val)

# -------------------------------
# 우측: 결과 출력
# -------------------------------
with col_right:
    # 독립적 DataFrame 생성 (A/B 슬라이더 독립성 보장)
    X_a = pd.DataFrame([X_base.iloc[0].copy()])
    X_b = pd.DataFrame([X_base.iloc[0].copy()])
    for col in cols:
        X_a[col] = scenario_a[col]
        X_b[col] = scenario_b[col]

    prob_a = pipeline.predict_proba(X_a)[:, 1][0]
    prob_b = pipeline.predict_proba(X_b)[:, 1][0]

    decision_a = prob_a >= best_threshold
    decision_b = prob_b >= best_threshold

    st.write(f"Scenario A 구매 확률: {prob_a:.2%}")
    st.write(f"Scenario B 구매 확률: {prob_b:.2%}")

    data_scenario = pd.DataFrame({
        "category": ["Scenario A", "Scenario B", "결정 기준값"],
        "value": [prob_a, prob_b, best_threshold],
        "color": ["red", "green", "blue"],
    })

    chart_scenario = (
        alt.Chart(data_scenario)
        .mark_bar()
        .encode(
            x=alt.X("category:N", axis=alt.Axis(labelAngle=0)),
            y=alt.Y("value:Q", scale=alt.Scale(domain=[0.0, 1.0])),
            color=alt.Color("color:N", scale=None),
            order=alt.Order('category', sort='ascending')  # '결정 기준값' 항상 우측
        )
    )

    st.altair_chart(chart_scenario, use_container_width=True)

    diff = prob_b - prob_a
    direction_score = 0
    for col in cols:
        delta = scenario_b[col] - scenario_a[col]
        direction_score += direction_hint[col] * (1 if delta > 0 else -1 if delta < 0 else 0)

    interpretation = (
        "Scenario B가 과거 구매 패턴에 더 가까운 행동 조합"
        if diff > 0 and direction_score > 0
        else "Scenario A가 과거 구매 패턴에 더 가까운 행동 조합"
        if diff < 0 and direction_score < 0
        else "지표 변화 방향이 혼재되어 해석 신뢰도 낮음"
    )

    st.write(f"구매 확률 차이: {diff:.2%}p")
    st.write(f"기준값(Threshold) 기준 비교: A={'구매' if decision_a else '비구매'}, B={'구매' if decision_b else '비구매'}")
    st.write(f"해석 결과: {interpretation}")