import sys
from pathlib import Path

import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

# set_page_config는 가장 먼저 호출
st.set_page_config(page_title="EDA", layout="wide")

from ui.header import render_header
from adapters.PurchaseIntentModelAdapter import PurchaseIntentModelAdapter

render_header()

st.title("🔍 EDA (탐색적 데이터 분석)")
st.markdown("---")



# app/pages/05... -> app/
APP_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = APP_DIR / "artifacts"

# 기본 데이터 로딩을 위한 어댑터 (기본 모델 경로 사용)
default_model_path = ARTIFACTS_DIR / "best_balancedrf_pipeline.joblib"

@st.cache_resource
def get_adapter(path: str) -> PurchaseIntentModelAdapter:
    return PurchaseIntentModelAdapter(path)

# 데이터 로드용 어댑터 (Selection 전)
adapter = get_adapter(str(default_model_path))

@st.cache_data
def load_data_from_adapter():
    """Adapter를 통해 학습 데이터를 로드합니다."""
    try:
        return adapter.get_training_data()
    except Exception as e:
        st.error(f"❌ 데이터 로드 실패: {e}")
        return None

df = load_data_from_adapter()

if df is not None:
    # ----------------------------------------------------
    # 1. 변수 간 상관관계 히트맵 (Training Data Original)
    # ----------------------------------------------------
    st.header("1. 변수 간 상관관계 히트맵")
    
    # 수치형 컬럼만 선택
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    # Revenue 포함
    if 'Revenue' not in numeric_cols and 'Revenue' in df.columns:
        numeric_cols.append('Revenue')
        
    corr_matrix = df[numeric_cols].corr()

    fig_corr, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', linewidths=0.5, ax=ax)
    st.pyplot(fig_corr)
    
    st.markdown("---")

    # ----------------------------------------------------
    # 2. 주요 변수 분포 비교 (Model Selection & Prediction)
    # ----------------------------------------------------
    st.header("2. 주요 변수 분포 비교")

    # (New Position) 모델 선택
    model_option = st.radio(
        "⚙️ 사용할 모델을 선택하세요:",
        ("ROC-AUC 기준 베스트 모델 사용", "PR-AUC 기준 베스트 모델 사용"),
        horizontal=True
    )

    if model_option == "ROC-AUC 기준 베스트 모델 사용":
        model_filename = "best_balancedrf_pipeline.joblib"
    else:
        model_filename = "best_pr_auc_balancedrf.joblib"

    model_path = ARTIFACTS_DIR / model_filename
    
    # 선택된 모델로 어댑터 다시 가져오기
    # (get_adapter는 캐시되므로 같은 경로면 재사용됨)
    selected_adapter = get_adapter(str(model_path))

    # 선택된 모델 정보 표시
    try:
        threshold = selected_adapter.get_threshold()
        st.info(f"✅ **선택된 모델:** {model_option} | **Threshold:** {threshold:.4f} | **File:** `{model_filename}`")
    except Exception as e:
        st.warning(f"모델 정보를 불러오는 중 오류 발생: {e}")

    # 모델 예측 수행 (df에 컬럼 추가)
    with st.spinner("모델 예측 중..."):
        try:
            preds = selected_adapter.predict(df) 
            df['Predicted_Revenue'] = preds
        except Exception as e:
            st.error(f"예측 수행 중 오류 발생: {e}")
            # 에러 발생 시 Predicted_Revenue 없이 진행

    col_sel1, col_sel2 = st.columns(2)
    with col_sel1:
        # target_col 선택 (Revenue류 제외)
        target_col = st.selectbox(
            "분석할 변수를 선택하세요:",
            [c for c in numeric_cols if c not in ['Revenue', 'Predicted_Revenue']]
        )
    with col_sel2:
        # 그룹 기준 선택
        # 예측 실패 시 옵션 조정
        group_options = ["Revenue (실제값)"]
        if 'Predicted_Revenue' in df.columns:
            group_options.append("Predicted_Revenue (예측값)")
            
        group_col = st.radio(
            "그룹 기준 선택:",
            group_options,
            horizontal=True
        )
    
    # 선택된 그룹 컬럼명 매핑
    group_key = 'Revenue' if group_col.startswith("Revenue") else 'Predicted_Revenue'

    fig_dist = px.box(
        df, 
        x=group_key, 
        y=target_col, 
        color=group_key, 
        title=f"{target_col} Distribution by {group_key}",
        color_discrete_map={True: '#2ecc71', False: '#e74c3c', 1: '#2ecc71', 0: '#e74c3c'},
        points="outliers"
    )
    st.plotly_chart(fig_dist, use_container_width=True)
