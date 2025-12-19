# app/pages/02_what_if.py
import streamlit as st
from ui.header import render_header

render_header()

st.set_page_config(page_title="What-if 시뮬레이터", layout="wide")

#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import pandas as pd
import joblib
import altair as alt
import numpy as np

# -------------------------------
# 0. 경로 고정
# -------------------------------
import os

# 현재 파일 위치: app/pages/02_what_if.py
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
# 2. 탭 가시성 강화 (CSS)
# -------------------------------
st.markdown(
    """
    <style>
    div[data-baseweb="tab"] > button {
        font-size: 20px;
        font-weight: 600;
        padding: 12px 16px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------------
# 3. 페이지 타이틀
# -------------------------------
st.title("What-if 시뮬레이터")

indicator_desc = {
    "BounceRates": "단일 페이지 방문 후 이탈한 세션의 비율",
    "ExitRates": "해당 페이지에서 세션이 종료된 비율",
    "PageValues": "페이지 방문이 전환에 기여한 기여도 (100% 기준으로 환산됨)",
    "ProductRelated_Duration": "상품 관련 페이지 체류 시간",
}

# 가중치
feature_weights = {
    "BounceRates": 1.0,
    "ExitRates": 1.0,
    "PageValues": 1.0,
    "ProductRelated_Duration": 1.0,
}

direction_hint = {
    "BounceRates": -1,
    "ExitRates": -1,
    "PageValues": 1,
    "ProductRelated_Duration": 1,
}

# ===============================
# PageValues 최대값 기준 100% 환산
# ===============================
pagevalues_max = X_train["PageValues"].max()

# ===============================
# What-if 시뮬레이터
# ===============================
st.caption("드래그를 통해 개별 행동 지표를 조정하여 구매 확률 변화 확인")

slider_values = {}
target_cols = list(indicator_desc.keys())

for col in target_cols:
    min_val = float(X_train[col].min())
    max_val = float(X_train[col].max())
    default_val = float(X_sample[col].iloc[0])
    if min_val == max_val:
        max_val = min_val + 1e-6

    if col == "PageValues":
        # PageValues는 0~100%로 변환
        min_val_pct = 0.0
        max_val_pct = 100.0
        default_val_pct = default_val / pagevalues_max * 100
        st.markdown(f"**{col}**  \n{indicator_desc[col]} (0% ~ 100%)")
        slider_values[col] = st.slider(
            label="",
            min_value=min_val_pct,
            max_value=max_val_pct,
            value=default_val_pct,
            key=f"slider_{col}",
        )
        # 실제 모델 입력값으로 환산
        slider_values[col] = slider_values[col] / 100 * pagevalues_max
    else:
        unit_label = " (초)" if "Duration" in col else ""
        st.markdown(f"**{col}**  \n{indicator_desc[col]}{unit_label}")
        slider_values[col] = st.slider(
            label="",
            min_value=min_val,
            max_value=max_val,
            value=default_val,
            key=f"slider_{col}",
        )

X_input = X_sample.copy()
for col in target_cols:
    X_input[col] = slider_values[col] * feature_weights[col]

prob = pipeline.predict_proba(X_input)[:, 1][0]
decision = "구매 판단 영역" if prob >= best_threshold else "비구매 판단 영역"

st.write(f"예측 구매 확률: {prob:.2%}")
st.write(f"판단 기준({best_threshold:.2%}) 대비 결과: {decision}")

data_prob = pd.DataFrame(
    {
        "category": ["구매확률", "판단기준"],
        "value": [prob, best_threshold],
        "color": ["red", "blue"],  # 색상 지정
    }
)

chart_prob = (
    alt.Chart(data_prob)
    .mark_bar()
    .encode(
        x=alt.X("category:N", axis=alt.Axis(labelAngle=0)),
        y="value:Q",
        color=alt.Color("color:N", scale=None)  # 지정한 색상 그대로 적용
    )
)

st.altair_chart(chart_prob, use_container_width=True)

# Threshold 대비 차이 계산
threshold_diff = prob - best_threshold
threshold_pct = threshold_diff / best_threshold * 100

if prob >= best_threshold:
    st.write(f"현재 행동 조합은 Threshold보다 {threshold_pct:.2f}% 높아 구매 가능성이 충분함")
else:
    st.write(f"현재 행동 조합은 Threshold보다 {abs(threshold_pct):.2f}% 낮아 구매 가능성 부족")
