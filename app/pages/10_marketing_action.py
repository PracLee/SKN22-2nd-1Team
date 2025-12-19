import streamlit as st
import pandas as pd
import sys
import os
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
try:
    from service.CustomerCareCenter import PurchaseIntentService 
    from adapters.model_loader import JoblibArtifactLoader
    from adapters.purchase_intent_pr_auc_adapter import PurchaseIntentPRAUCModelAdapter
except ImportError:
    from app.service.CustomerCareCenter import PurchaseIntentService
    from app.adapters.model_loader import JoblibArtifactLoader
    from app.adapters.purchase_intent_pr_auc_adapter import PurchaseIntentPRAUCModelAdapter

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
df_full = load_data()

# --- [수정 포인트] 세션 값을 10개로 제한하여 정리 ---
# 전체 데이터 중 분석이 용이하도록 상위 10개만 슬라이싱합니다.
df = df_full.head(10)

# --- [STEP 4] UI 레이아웃 구성 ---
render_header()
st.title("🎯 마케팅 전략 가이드 시뮬레이터")
st.info("💡 분석 효율을 위해 상위 10개의 주요 타겟 세션을 요약하여 제공합니다.")

# 레이아웃 분할
left_col, right_col = st.columns([4, 6])

with left_col:
    st.subheader("📥 타겟 세션 리스트 (TOP 10)")
    
    # 세션 인덱스를 1~10번 형식으로 표시하여 가독성을 높임
    display_labels = [f"세션 분석 대상 #{i+1} (ID: {idx})" for i, idx in enumerate(df.index)]
    selected_label = st.selectbox("분석할 세션 선택", display_labels, key="action_select")
    
    # 선택된 라벨로부터 실제 인덱스 추출
    selected_idx = df.index[display_labels.index(selected_label)]
    row = df.loc[selected_idx]
    
    # 예측 데이터 계산
    X_one = pd.DataFrame([row.drop("Revenue", errors="ignore")])
    proba = adapter.predict_proba(X_one).iloc[0]
    risk = service.classify_risk(proba)
    action = service.recommend_action(row.to_dict(), proba)
    
    st.write("---")
    st.write("**📌 선택된 세션 행동 지표**")
    st.metric("페이지 가치 (Value)", f"{row.get('PageValues', 0):.2f}")
    st.metric("이탈률 (Bounce)", f"{row.get('BounceRates', 0)*100:.1f}%")
    st.metric("체류 시간 (Duration)", f"{row.get('ProductRelated_Duration', 0):.1f}s")

with right_col:
    st.subheader("👤 고객 페르소나 진단")
    
    # 중앙 정렬 컨테이너
    with st.container():
        if risk == "HIGH_RISK":
            img_url = "https://cdn-icons-png.flaticon.com/512/9245/9245580.png"
            status_text = "이탈 위험 높음: 케어가 시급합니다."
            color = "#FF4B4B"
        elif risk == "OPPORTUNITY":
            img_url = "https://cdn-icons-png.flaticon.com/512/9245/9245548.png"
            status_text = "망설임: 혜택이 필요한 시점입니다."
            color = "#FFAA00"
        else:
            img_url = "https://cdn-icons-png.flaticon.com/512/9245/9245524.png"
            status_text = "구매 유력: 긍정적 흐름 유지 중입니다."
            color = "#00A65A"

        st.markdown(
            f"""
            <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; padding: 20px; border-radius: 15px; background-color: rgba(255, 255, 255, 0.05); border: 2px solid {color};">
                <img src="{img_url}" width="150" style="margin-bottom: 15px;">
                <h3 style="color: {color}; margin: 0;">{status_text}</h3>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.write("") 
    st.write(f"**실시간 구매 전환 확률: {proba*100:.1f}%**")
    st.progress(float(proba))
    
    st.markdown(f"#### 📌 추천 마케팅 전략\n> **{action}**")

# 하단 데이터 상세 보기
with st.expander("🔍 선택된 세션 상세 로그 확인"):
    st.table(pd.DataFrame([row]).T)