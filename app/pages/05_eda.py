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
    # ----------------------------------------------------
    # 2. 주요 변수 분포 비교
    # ----------------------------------------------------
    st.header("2. 주요 변수 분포 비교")

    # target_col 선택 (Revenue류 제외) 리스트 생성 및 정렬
    selectable_cols = [c for c in numeric_cols if c != 'Revenue']

    # PageValues를 최상단으로 이동 (관심도 높은 변수)
    if 'PageValues' in selectable_cols:
        selectable_cols.remove('PageValues')
        selectable_cols.insert(0, 'PageValues')

    # row_id를 최하단으로 이동 (단순 식별자)
    if 'row_id' in selectable_cols:
        selectable_cols.remove('row_id')
        selectable_cols.append('row_id')

    target_col = st.selectbox(
        "분석할 변수를 선택하세요:",
        selectable_cols
    )
    
    # 그룹 기준은 실제값(Revenue)으로 고정
    group_key = 'Revenue'

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
