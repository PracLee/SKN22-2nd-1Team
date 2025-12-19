# app/pages/02_what_if.py
import streamlit as st
from ui.header import render_header

render_header()

st.set_page_config(page_title="ab_test", layout="wide")

#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import pandas as pd
import joblib
import altair as alt
import numpy as np
import os

# -------------------------------
# 0. 경로 고정
# -------------------------------

# 현재 파일 위치: app/pages/08_ab_test.py
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))  # pages → app → 2nd
TRAIN_PATH = os.path.join(BASE_DIR, "data/processed/train.csv")
TEST_PATH = os.path.join(BASE_DIR, "data/processed/test.csv")
MODEL_PATH = os.path.join(BASE_DIR, "app/artifacts/best_balancedrf_pipeline.joblib")





# -------------------------------
# 1. 데이터 / 모델 로드
# -------------------------------
X_train = pd.read_csv(TRAIN_PATH)
X_test = pd.read_csv(TEST_PATH)

model_dict = joblib.load(MODEL_PATH)
pipeline = model_dict["pipeline"]
best_threshold = model_dict["best_threshold"]

# 무작위 샘플 선택
sample_idx = np.random.choice(X_test.index, size=5, replace=False)
X_sample = X_test.loc[sample_idx]

# -------------------------------
# 2. 페이지 타이틀
# -------------------------------
st.title("Scenario 비교 시뮬레이터")

indicator_desc = {
    "BounceRates": "단일 페이지 방문 후 이탈한 세션의 비율",
    "ExitRates": "해당 페이지에서 세션이 종료된 비율",
    "PageValues": "페이지 방문이 전환에 기여한 기여도 (0~100 기준)",
    "ProductRelated_Duration": "상품 관련 페이지 체류 시간(초)  ",
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

# PageValues 최대값 기준 100% 환산
pagevalues_max = X_train["PageValues"].max()

st.subheader("Scenario A")
for col in cols:
    if col == "PageValues":
        min_val = 0.0
        max_val = 100.0
        default_val = float(X_sample[col].iloc[0]) / pagevalues_max * 100
        st.markdown(f"**{col}** — {indicator_desc[col]}")
        scenario_a[col] = st.slider(
            label="",
            min_value=min_val,
            max_value=max_val,
            value=default_val,
            step=1.0,
            key=f"A_{col}",
        )
        scenario_a[col] = scenario_a[col] / 100 * pagevalues_max
    else:
        min_val = float(X_train[col].min())
        max_val = float(X_train[col].max())
        st.markdown(f"**{col}** — {indicator_desc[col]}")
        scenario_a[col] = st.slider(
            label="",
            min_value=min_val,
            max_value=max_val,
            value=float(X_sample[col].iloc[0]),
            key=f"A_{col}",
        )

st.subheader("Scenario B")
for col in cols:
    if col == "PageValues":
        min_val = 0.0
        max_val = 100.0
        default_val = float(X_sample[col].iloc[0]) / pagevalues_max * 100
        st.markdown(f"**{col}** — {indicator_desc[col]}")
        scenario_b[col] = st.slider(
            label="",
            min_value=min_val,
            max_value=max_val,
            value=default_val,
            step=1.0,
            key=f"B_{col}",
        )
        scenario_b[col] = scenario_b[col] / 100 * pagevalues_max
    else:
        min_val = float(X_train[col].min())
        max_val = float(X_train[col].max())
        st.markdown(f"**{col}** — {indicator_desc[col]}")
        scenario_b[col] = st.slider(
            label="",
            min_value=min_val,
            max_value=max_val,
            value=float(X_sample[col].iloc[0]),
            key=f"B_{col}",
        )

X_a = X_sample.copy()
X_b = X_sample.copy()
for col in cols:
    X_a[col] = scenario_a[col]
    X_b[col] = scenario_b[col]

prob_a = pipeline.predict_proba(X_a)[:, 1][0]
prob_b = pipeline.predict_proba(X_b)[:, 1][0]

decision_a = prob_a >= best_threshold
decision_b = prob_b >= best_threshold

st.write(f"Scenario A 구매 확률: {prob_a:.2%}")
st.write(f"Scenario B 구매 확률: {prob_b:.2%}")

data_scenario = pd.DataFrame(
    {
        "category": ["Scenario A", "Scenario B", "판단기준"],
        "value": [prob_a, prob_b, best_threshold],
        "color": ["red", "green", "blue"],  # 색상 지정
    }
)

chart_scenario = (
    alt.Chart(data_scenario)
    .mark_bar()
    .encode(
        x=alt.X("category:N", axis=alt.Axis(labelAngle=0)),
        y="value:Q",
        color=alt.Color("color:N", scale=None)  # 지정한 색상 그대로 적용
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
st.write(f"Threshold 기준 비교: A={'구매' if decision_a else '비구매'}, B={'구매' if decision_b else '비구매'}")
st.write(f"해석 결과: {interpretation}")

