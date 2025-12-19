from __future__ import annotations
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import sys
from pathlib import Path

from ui.header import render_header


render_header()
st.set_page_config(page_title="EDA", layout="wide")
from service.session_probability_service import (
    SessionProbabilityService,
    SessionPredictionResult,
)

# --------------------------------------------------------------------------------
# 0. 경로 설정
# --------------------------------------------------------------------------------
# 현재 파일 위치: app/pages/05_eda.py
# 이 파일의 상위 상위(app) 폴더를 sys.path에 추가해야 "adapters" 패키지를 찾을 수 있음
app_path = Path(__file__).resolve().parent.parent
if str(app_path) not in sys.path:
    sys.path.append(str(app_path))

root_path = app_path.parent

# --------------------------------------------------------------------------------
# 1. 페이지 설정 및 데이터 로드
# --------------------------------------------------------------------------------

@st.cache_resource
def get_session_probability_service() -> SessionProbabilityService:
    """
    - 모델/어댑터는 여기서 한 번만 로드 (Streamlit 캐싱)
    - Global 평균 값은 추후 실제 데이터 기준으로 수정 가능
    """
    return SessionProbabilityService(global_avg_purchase_prob=0.15)

service = get_session_probability_service()

@st.cache_data
def load_data_from_service():
    """Service를 통해 학습 데이터를 로드합니다."""
    try:
        return service.get_training_data()
    except Exception as e:
        st.error(f"❌ 데이터 로드 실패: {e}")
        return None

df = load_data_from_service()

if df is not None:
    st.title("🔍 EDA (탐색적 데이터 분석)")
    st.markdown("---")

    st.header("1. 변수 간 상관관계 히트맵")
    
    # 수치형 컬럼만 선택
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    # 의미 없는 식별자성 컬럼 제외 (필요 시)
    if 'Revenue' not in numeric_cols:
        numeric_cols.append('Revenue')
        
    corr_matrix = df[numeric_cols].corr()

    fig_corr, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', linewidths=0.5, ax=ax)
    st.pyplot(fig_corr)
    
    st.markdown("---")

    st.header("2. 주요 변수 분포 비교 (Revenue T/F)")
    
    target_col = st.selectbox(
        "분석할 변수를 선택하세요:",
        [c for c in numeric_cols if c != 'Revenue']
    )
    
    # Box Plot으로 변경된 코드 적용
    fig_dist = px.box(
        df, 
        x="Revenue", 
        y=target_col, 
        color="Revenue", 
        title=f"{target_col} Distribution by Revenue",
        color_discrete_map={True: '#2ecc71', False: '#e74c3c'},
        points="outliers"
    )
    st.plotly_chart(fig_dist, use_container_width=True)
