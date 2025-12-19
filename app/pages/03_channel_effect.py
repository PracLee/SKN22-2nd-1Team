import sys
from pathlib import Path

import streamlit as st
import pandas as pd
import plotly.express as px

# set_page_config는 가장 먼저 호출
st.set_page_config(page_title="chennel_effect", layout="wide")

from ui.header import render_header
from adapters.PurchaseIntentModelAdapter import PurchaseIntentModelAdapter

render_header()

st.title("1. 유입 채널(TrafficType) 및 지역(Region)별 효율 분석")
st.markdown("---")

# --------------------------------------------------------------------------------
# (New Position) 분석 기준 선택 (Global)
# --------------------------------------------------------------------------------
metric_choice = st.radio(
    "📊 분석 기준 선택:",
    ("Actual Data (실제 데이터)", "Model Prediction (모델 예측 결과)"),
    horizontal=True
)

# app/pages/03... -> app/
APP_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = APP_DIR / "artifacts"

# 기본 데이터 로딩용 (Selection과 무관하게 데이터는 불변)
default_model_path = ARTIFACTS_DIR / "best_balancedrf_pipeline.joblib"

@st.cache_resource
def get_adapter(path: str) -> PurchaseIntentModelAdapter:
    return PurchaseIntentModelAdapter(path)

# 데이터 로드용 어댑터
loading_adapter = get_adapter(str(default_model_path))

@st.cache_data
def load_data_from_adapter():
    """Adapter를 통해 학습 데이터를 로드합니다."""
    try:
        return loading_adapter.get_training_data()
    except Exception as e:
        # Fallback if method missing
        return None  

df = load_data_from_adapter()

if df is not None:
    st.info("💡 **전환율(Conversion Rate)**: 해당 채널/지역 방문자 중 실제로 구매(Revenue)한 비율")

    # ----------------------------------------------------
    # (New Layout) 모델 선택 & 그래프 스타일 - Side by Side
    # ----------------------------------------------------
    col_ctrl1, col_ctrl2 = st.columns(2)
    
    with col_ctrl1:
        model_option = st.radio(
            "⚙️ 사용할 모델을 선택하세요:",
            ("ROC-AUC 기준 베스트 모델 사용", "PR-AUC 기준 베스트 모델 사용"),
            horizontal=True,
            disabled=metric_choice.startswith("Actual")
        )

    with col_ctrl2:
        plot_type = st.radio(
            "📈 그래프 스타일 선택:", 
            ["Bar Chart (막대)", "Area Chart (영역)"], 
            horizontal=True
        )

    if model_option == "ROC-AUC 기준 베스트 모델 사용":
        model_filename = "best_balancedrf_pipeline.joblib"
    else:
        model_filename = "best_pr_auc_balancedrf.joblib"

    model_path = ARTIFACTS_DIR / model_filename
    
    # 예측용 어댑터 (선택된 모델)
    prediction_adapter = get_adapter(str(model_path))
    
    # 모델 정보 표시
    try:
        threshold = prediction_adapter.get_threshold()
        st.caption(f"ℹ️ **Selected Model Threshold:** {threshold:.4f} ({model_filename})")
    except:
        pass

    # 모델 예측 수행
    with st.spinner("모델 예측 중..."):
        try:
            preds = prediction_adapter.predict(df) 
            df['Predicted_Revenue'] = preds
        except Exception as e:
            st.error(f"예측 실패: {e}")

    # 선택에 따른 타겟 컬럼 설정
    target_metric = 'Revenue' if metric_choice.startswith("Actual") else 'Predicted_Revenue'
    metric_label = '실제 구매 전환율 (%)' if target_metric == 'Revenue' else '모델 예측 전환율 (%)'
    
    # 예측값 선택했는데 데이터 없으면 처리
    if target_metric == 'Predicted_Revenue' and 'Predicted_Revenue' not in df.columns:
        st.warning("⚠️ 예측 데이터 생성 실패로 인해 실제 데이터로 대체합니다.")
        target_metric = 'Revenue'

    col1, col2 = st.columns(2)

    def create_dynamic_plot(data, x_col, y_col, 
                            chart_type, 
                            color_scale='Blues', 
                            x_label=None, y_label=None):
        """선택된 차트 타입에 따라 Plotly Figure 생성"""
        common_args = {
            'data_frame': data,
            'x': x_col,
            'y': y_col,
            'labels': {y_col: y_label, x_col: x_label}
        }
        
        if "Bar" in chart_type:
            fig = px.bar(**common_args, color=y_col, color_continuous_scale=color_scale, text_auto='.1f')
        elif "Area" in chart_type:
            fig = px.area(**common_args)
        else:
            fig = px.bar(**common_args)
        
        return fig

    # TrafficType
    with col1:
        st.subheader("🚦 Traffic Type 별 효율")
        # target_metric(실제/예측)에 따라 평균 계산
        traffic_eff = df.groupby('TrafficType')[target_metric].mean().reset_index()
        traffic_eff[target_metric] = traffic_eff[target_metric] * 100
        traffic_eff = traffic_eff.sort_values(by=target_metric, ascending=False)
        
        traffic_eff['TrafficType'] = traffic_eff['TrafficType'].astype(str)

        fig_traffic = create_dynamic_plot(
            traffic_eff, 'TrafficType', target_metric, 
            plot_type, 
            color_scale='Blues',
            x_label='Traffic Type ID', y_label=metric_label
        )
        fig_traffic.update_layout(xaxis_type='category')
        st.plotly_chart(fig_traffic, use_container_width=True)

    # Region
    with col2:
        st.subheader("🌍 지역(Region) 별 효율")
        region_eff = df.groupby('Region')[target_metric].mean().reset_index()
        region_eff[target_metric] = region_eff[target_metric] * 100
        region_eff = region_eff.sort_values(by=target_metric, ascending=False)
        
        region_eff['Region'] = region_eff['Region'].astype(str)

        fig_region = create_dynamic_plot(
            region_eff, 'Region', target_metric, 
            plot_type, 
            color_scale='Greens',
            x_label='Region ID', y_label=metric_label
        )
        fig_region.update_layout(xaxis_type='category')
        st.plotly_chart(fig_region, use_container_width=True)
