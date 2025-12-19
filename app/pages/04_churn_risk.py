import streamlit as st
import pandas as pd
import sys
import os
import plotly.graph_objects as go  # 시각화를 위한 추가
from ui.header import render_header

# --- [STEP 1] 경로 설정 ---
current_dir = os.path.dirname(os.path.abspath(__file__))
app_dir = os.path.abspath(os.path.join(current_dir, ".."))
project_root = os.path.abspath(os.path.join(app_dir, ".."))

if project_root not in sys.path:
    sys.path.insert(0, project_root)
if app_dir not in sys.path:
    sys.path.insert(0, app_dir)

# --- [STEP 2] 모듈 임포트 ---
from service.CustomerCareCenter import PurchaseIntentService
from adapters.model_loader import JoblibArtifactLoader
from adapters.purchase_intent_pr_auc_adapter import PurchaseIntentPRAUCModelAdapter

# --- [STEP 3] 데이터 및 서비스 로드 ---
@st.cache_resource
def init_service():
    model_path = "artifacts/best_pr_auc_balancedrf.joblib"
    adapter = PurchaseIntentPRAUCModelAdapter(model_path) 
    return PurchaseIntentService(adapter), adapter

@st.cache_data
def load_data():
    return pd.read_csv("data/processed/test.csv")

service, adapter = init_service()
df = load_data()

# --- [STEP 4] UI 레이아웃 설정 ---
render_header()
st.title("🛡️ 고객 이탈 방지 및 마케팅 전략 가이드")

# 좌우 레이아웃 분할 (비율 4:6)
left_col, right_col = st.columns([4, 6])

with left_col:
    st.subheader("📝 세션 정보 입력")
    # 분석할 고객 세션을 상위 10개만 슬라이싱하여 표시하도록 수정
    idx = st.selectbox("분석할 고객 세션 선택", df.index[:10], key="session_select")
    row = df.loc[idx]
    
    st.write("---")
    st.write("**📍 주요 행동 지표**")
    st.write(f"- 페이지 가치: `{row.get('PageValues', 0):.2f}`")
    st.write(f"- 이탈률: `{row.get('BounceRates', 0)*100:.1f}%`")
    st.write(f"- 체류 시간: `{row.get('ProductRelated_Duration', 0):.0f}초`")
    
    # 예측 수행 준비
    X_one = pd.DataFrame([row.drop("Revenue", errors="ignore")])
    proba = adapter.predict_proba(X_one).iloc[0]
    risk = service.classify_risk(proba)
    action = service.recommend_action(row.to_dict(), proba)

with right_col:
    st.subheader("📊 분석 결과 및 시각화")
    
    # [그래프 표현] Plotly 게이지 차트 생성
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = proba * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "구매 전환 확률 (%)", 'font': {'size': 20}},
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 1},
            'bar': {'color': "#1f77b4"},
            'steps': [
                {'range': [0, 20], 'color': "#ff4b4b"},  # HIGH RISK 영역
                {'range': [20, 60], 'color': "#ffa500"}, # OPPORTUNITY 영역
                {'range': [60, 100], 'color': "#28a745"} # LIKELY BUYER 영역
            ],
            'threshold': {
                'line': {'color': "white", 'width': 4},
                'thickness': 0.75,
                'value': proba * 100
            }
        }
    ))
    fig.update_layout(height=350, margin=dict(l=20, r=20, t=50, b=20))
    st.plotly_chart(fig, use_container_width=True)

    # 상태 및 액션 카드
    if risk == "HIGH_RISK":
        st.error(f"🚨 **상태: 고위험 이탈군** (확률: {proba*100:.1f}%)")
    elif risk == "OPPORTUNITY":
        st.warning(f"⚠️ **상태: 전환 기회군** (확률: {proba*100:.1f}%)")
    else:
        st.success(f"✅ **상태: 구매 유력군** (확률: {proba*100:.1f}%)")

    st.info(f"💡 **추천 마케팅 액션:**\n\n{action}")

# 하단 추가 정보 (선택 사항)
with st.expander("🔍 상세 세션 데이터 보기"):
    st.dataframe(pd.DataFrame([row]))