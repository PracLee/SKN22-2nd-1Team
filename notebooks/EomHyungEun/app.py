import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

# --------------------------------------------------------------------------------
# 1. 페이지 기본 설정 및 데이터 로드
# --------------------------------------------------------------------------------
st.set_page_config(
    page_title="쇼핑몰 데이터 분석 대시보드",
    page_icon="🛍️",
    layout="wide"
)

@st.cache_data
def load_data():
    # 데이터 로드 (같은 경로에 train.csv가 있어야 함)
    try:
        df = pd.read_csv('train.csv')
        return df
    except FileNotFoundError:
        st.error("❌ 'train.csv' 파일을 찾을 수 없습니다. 같은 폴더에 파일을 넣어주세요.")
        return None

df = load_data()

if df is not None:
    # --------------------------------------------------------------------------------
    # 2. 사이드바 및 제목
    # --------------------------------------------------------------------------------
    st.title("📊 온라인 쇼핑몰 고객 행동 분석")
    st.markdown("---")

    # 탭 구성
    tab1, tab2 = st.tabs(["📢 채널 및 지역 효과 분석", "🔍 EDA (탐색적 데이터 분석)"])

    # --------------------------------------------------------------------------------
    # [TAB 1] 채널 효과 분석 (TrafficType / Region)
    # --------------------------------------------------------------------------------
    with tab1:
        st.header("1. 유입 채널(TrafficType) 및 지역(Region)별 효율 분석")
        st.info("💡 **전환율(Conversion Rate)**: 해당 채널/지역 방문자 중 실제로 구매(Revenue)한 비율")

        col1, col2 = st.columns(2)

        # 1-1. TrafficType 별 전환율 분석
        with col1:
            st.subheader("🚦 Traffic Type 별 구매 전환율")
            
            # 데이터 가공
            traffic_eff = df.groupby('TrafficType')['Revenue'].mean().reset_index()
            traffic_eff['Revenue'] = traffic_eff['Revenue'] * 100  # 백분율 변환
            traffic_eff = traffic_eff.sort_values(by='Revenue', ascending=False)
            
            # 그래프 (Bar Chart)
            fig_traffic = px.bar(
                traffic_eff, 
                x='TrafficType', 
                y='Revenue',
                color='Revenue',
                labels={'Revenue': '구매 전환율 (%)', 'TrafficType': 'Traffic Type ID'},
                color_continuous_scale='Blues',
                text_auto='.1f'
            )
            fig_traffic.update_layout(xaxis_type='category') # X축을 카테고리로 인식
            st.plotly_chart(fig_traffic, use_container_width=True)
            
            st.markdown("""
            **해석 가이드:**
            - 그래프가 높을수록 **구매 확률이 높은 알짜배기 유입 경로**입니다.
            - 전환율이 낮지만 방문자가 많은 채널은 '인지도 확대'용일 수 있습니다.
            """)

        # 1-2. Region 별 전환율 분석
        with col2:
            st.subheader("🌍 지역(Region) 별 구매 전환율")
            
            # 데이터 가공
            region_eff = df.groupby('Region')['Revenue'].mean().reset_index()
            region_eff['Revenue'] = region_eff['Revenue'] * 100
            region_eff = region_eff.sort_values(by='Revenue', ascending=False)
            
            # 그래프 (Bar Chart)
            fig_region = px.bar(
                region_eff, 
                x='Region', 
                y='Revenue',
                color='Revenue',
                labels={'Revenue': '구매 전환율 (%)', 'Region': 'Region ID'},
                color_continuous_scale='Greens',
                text_auto='.1f'
            )
            fig_region.update_layout(xaxis_type='category')
            st.plotly_chart(fig_region, use_container_width=True)

            st.markdown("""
            **해석 가이드:**
            - 특정 지역의 전환율이 유독 낮다면, 해당 지역의 **배송비, 언어, 마케팅 메시지** 등을 점검해야 합니다.
            """)
        
        # 1-3. 상세 데이터 보기 (옵션)
        with st.expander("🔢 상세 데이터 표 보기"):
            st.dataframe(df.groupby(['TrafficType', 'Region'])['Revenue'].mean().unstack().fillna(0).style.background_gradient(cmap='YlOrRd'))


    # --------------------------------------------------------------------------------
    # [TAB 2] EDA 대시보드 (상관관계 & 분포)
    # --------------------------------------------------------------------------------
    with tab2:
        st.header("2. 데이터 탐색 (EDA)")
        
        # 2-1. 상관관계 히트맵
        st.subheader("🔥 변수 간 상관관계 히트맵")
        st.markdown("수치형 변수들 사이의 관계를 파악하여, **매출(Revenue)과 가장 관련 깊은 변수**를 찾습니다.")
        
        # 수치형 컬럼만 선택
        numerical_cols = ['Administrative', 'Administrative_Duration', 'Informational', 
                          'Informational_Duration', 'ProductRelated', 'ProductRelated_Duration', 
                          'BounceRates', 'ExitRates', 'PageValues', 'SpecialDay', 'Revenue']
        
        corr_matrix = df[numerical_cols].corr()

        # 히트맵 그리기
        fig_corr, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', linewidths=0.5, ax=ax)
        st.pyplot(fig_corr)
        
        st.warning("⚡ **PageValues**가 Revenue와 가장 강한 양의 상관관계를 보인다면, 이 지표 관리가 핵심입니다.")

        st.markdown("---")

        # 2-2. 분포도 시각화 (Interactive)
        st.subheader("📈 주요 변수 분포 비교 (구매 vs 비구매)")
        
        col_dist1, col_dist2 = st.columns([1, 3])
        
        with col_dist1:
            # 사용자 선택 박스
            target_col = st.selectbox(
                "분석할 변수를 선택하세요:",
                ['PageValues', 'ExitRates', 'BounceRates', 'ProductRelated_Duration', 'Administrative_Duration']
            )
            st.markdown(f"**선택된 변수:** `{target_col}`")
            st.markdown("구매한 그룹(True)과 구매하지 않은 그룹(False)의 차이를 확인하세요.")

        with col_dist2:
            # 히스토그램 & 박스플롯 (Plotly)
            fig_dist = px.histogram(
                df, 
                x=target_col, 
                color="Revenue", 
                marginal="box", # 상단에 박스플롯 추가
                barmode="overlay", 
                title=f"{target_col} Distribution by Revenue",
                color_discrete_map={True: '#2ecc71', False: '#e74c3c'}, # 초록(구매), 빨강(비구매)
                opacity=0.7
            )
            st.plotly_chart(fig_dist, use_container_width=True)
