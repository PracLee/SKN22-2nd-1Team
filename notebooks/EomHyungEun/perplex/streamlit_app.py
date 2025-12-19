# ============================================================================
# Online Shoppers Intent Prediction - Streamlit App
# 10가지 기능: 계산기, 시뮬레이터, 채널분석, 이탈탐지, EDA, Feature Importance, 페르소나, 비교, 모델비교, 액션추천
# ============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 1. 초기 설정 및 데이터 로드
# ============================================================================

st.set_page_config(
    page_title="온라인 쇼핑 구매 예측 시스템",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS 커스터마이징
st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .high-prob {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
    }
    .med-prob {
        background-color: #fff3cd;
        border-left: 5px solid #ffc107;
    }
    .low-prob {
        background-color: #f8d7da;
        border-left: 5px solid #dc3545;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models_and_data():
    """모델, 메타데이터, 데이터 로드"""
    with open('models_trained.pkl', 'rb') as f:
        models = pickle.load(f)
    
    with open('metadata.json', 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    
    with open('label_encoders.pkl', 'rb') as f:
        label_encoders = pickle.load(f)
    
    df = pd.read_csv('data_clean.csv')
    
    return models, metadata, scaler, label_encoders, df

models, metadata, scaler, label_encoders, df = load_models_and_data()

# ============================================================================
# 2. 유틸리티 함수
# ============================================================================

def predict_purchase(input_dict, model_name='GB', scaler_obj=None):
    """
    세션 데이터로 구매 확률 예측
    """
    model_info = models[model_name]
    model = model_info['model']
    
    feature_cols = metadata['feature_columns']
    input_array = np.array([input_dict.get(col, 0) for col in feature_cols]).reshape(1, -1)
    
    if model_name == 'LR':
        input_array = scaler_obj.transform(input_array)
    
    proba = model.predict_proba(input_array)[0, 1]
    return proba

def get_probability_color(prob):
    """확률에 따른 색상 반환"""
    if prob >= 0.25:
        return "🟢", "high-prob", "#28a745"
    elif prob >= 0.15:
        return "🟡", "med-prob", "#ffc107"
    else:
        return "🔴", "low-prob", "#dc3545"

def generate_insight_text(input_dict, prediction_prob):
    """예측 결과 기반 인사이트 텍스트 생성"""
    insights = []
    
    page_values = input_dict.get('PageValues', 0)
    pagevalues_log = input_dict.get('PageValues_log', 0)
    bounce = input_dict.get('BounceRates', 0)
    exit_rate = input_dict.get('ExitRates', 0)
    product_related = input_dict.get('ProductRelated', 0)
    product_duration = input_dict.get('ProductRelated_Duration', 0)
    visitor_type = input_dict.get('VisitorType', 'Unknown')
    
    # 강점 분석
    if pagevalues_log > np.percentile(df['PageValues_log'], 75):
        insights.append("✅ 높은 페이지 가치 - 상품에 대한 관심 신호 강함")
    
    if bounce < 0.01:
        insights.append("✅ 매우 낮은 이탈률 - 랜딩 페이지 적응 우수")
    
    if exit_rate < 0.025:
        insights.append("✅ 낮은 이탈률 - 사용자 여정 진행률 높음")
    
    if product_related > np.percentile(df['ProductRelated'], 75):
        insights.append("✅ 높은 상품 페이지 탐색 - 구매 의도 명확")
    
    if visitor_type == 'New_Visitor':
        insights.append("✅ 신규 방문자 - 높은 전환 잠재력 (신규 방문자의 평균 구매율 24.9%)")
    
    # 약점 분석
    if bounce > 0.1:
        insights.append("⚠️ 높은 이탈률 - 랜딩 페이지 개선 필요")
    
    if exit_rate > 0.1:
        insights.append("⚠️ 높은 이탈률 - 체크아웃 프로세스 단순화 추천")
    
    if pagevalues_log < 1:
        insights.append("⚠️ 낮은 페이지 가치 - 가치 페이지 도달 실패")
    
    if product_related < 5:
        insights.append("⚠️ 상품 탐색 부족 - 추천 상품 노출 강화 필요")
    
    return insights

def generate_action_recommendation(input_dict, prediction_prob):
    """액션 추천 생성"""
    page_values = input_dict.get('PageValues', 0)
    pagevalues_log = input_dict.get('PageValues_log', 0)
    bounce = input_dict.get('BounceRates', 0)
    exit_rate = input_dict.get('ExitRates', 0)
    product_duration = input_dict.get('ProductRelated_Duration', 0)
    
    if prediction_prob >= 0.25:
        return "🎯 높은 구매 가능성: 즉시 구매 유도 (제한 시간 제안, 결제 버튼 강조)"
    
    if prediction_prob < 0.15 and pagevalues_log > np.percentile(df['PageValues_log'], 75):
        return "💰 상품 관심 높음 + 낮은 구매율: 할인 쿠폰 또는 리뷰/평점 강조 권장"
    
    if bounce > 0.1 or exit_rate > 0.1:
        return "🔧 높은 이탈률 감지: 랜딩 페이지/체크아웃 단순화, 신뢰 요소(보증, 리뷰) 추가"
    
    if product_duration < np.percentile(df['ProductRelated_Duration'], 25):
        return "📱 짧은 체류 시간: 연관 상품 추천, 특가/번들 상품 노출"
    
    return "🔄 일반적인 트래픽: 개인화된 상품 추천, 이메일 팔로우업 전략 추천"

# ============================================================================
# 3. 사이드바 구성
# ============================================================================

with st.sidebar:
    st.title("🛒 온라인 쇼핑 구매 예측")
    st.markdown("---")
    
    # 페이지 선택
    page = st.radio(
        "📑 메뉴 선택",
        [
            "1️⃣ 구매 확률 계산기",
            "2️⃣ What-If 시뮬레이터",
            "3️⃣ 채널 효과 분석",
            "4️⃣ 이탈 세션 탐지",
            "5️⃣ EDA 대시보드",
            "6️⃣ Feature Importance",
            "7️⃣ 고객 페르소나",
            "8️⃣ 시나리오 비교",
            "9️⃣ 모델 성능 비교",
            "🔟 액션 추천 카드"
        ]
    )
    
    st.markdown("---")
    
    # 모델 선택
    default_model_idx = 2  # GB
    selected_model = st.selectbox(
        "🤖 모델 선택",
        ["LR (Logistic Regression)", "RF (Random Forest)", "GB (Gradient Boosting)"],
        index=default_model_idx,
        help="예측에 사용할 모델을 선택하세요"
    )
    model_name = selected_model.split()[0]
    
    st.markdown("---")
    st.markdown("**📊 프로젝트 정보**")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("총 세션", f"{metadata['data_summary']['total_sessions']:,}")
    with col2:
        st.metric("구매율", f"{metadata['class_distribution']['Purchase_ratio']:.1%}")

# ============================================================================
# 4. 페이지별 콘텐츠
# ============================================================================

if page == "1️⃣ 구매 확률 계산기":
    st.header("🎯 세션 구매 확률 계산기")
    st.markdown("한 고객의 세션 정보를 입력하면 구매 확률을 예측합니다.")
    
    col1, col2 = st.columns(2)
    
    # 입력 폼 (좌측)
    with col1:
        st.subheader("📝 세션 정보 입력")
        
        # 페이지 방문 수
        st.markdown("**🌐 페이지 방문 수**")
        admin = st.number_input("Administrative 페이지", min_value=0, max_value=30, value=2)
        info = st.number_input("Informational 페이지", min_value=0, max_value=30, value=0)
        product = st.number_input("ProductRelated 페이지", min_value=0, max_value=100, value=20)
        
        # 체류 시간 (초)
        st.markdown("**⏱️ 체류 시간 (초)**")
        admin_dur = st.number_input("Administrative 체류", min_value=0.0, value=50.0, step=10.0)
        info_dur = st.number_input("Informational 체류", min_value=0.0, value=0.0, step=10.0)
        product_dur = st.number_input("ProductRelated 체류", min_value=0.0, value=500.0, step=50.0)
        
        # 행동 지표
        st.markdown("**📈 행동 지표**")
        bounce = st.slider("Bounce Rate (0-0.2)", 0.0, 0.2, 0.02, 0.001)
        exit_rate = st.slider("Exit Rate (0-0.2)", 0.0, 0.2, 0.05, 0.001)
        page_value = st.number_input("Page Values", min_value=0.0, value=10.0, step=5.0)
        
        # 범주형 변수
        st.markdown("**📋 기타 정보**")
        month = st.selectbox("월", metadata['le_month_classes'], index=4)  # May
        visitor_type = st.selectbox("방문자 유형", metadata['le_visitor_classes'], index=0)  # New
        weekend = st.checkbox("주말 방문", value=False)
        special_day = st.number_input("특별한 날까지의 거리", min_value=0.0, max_value=1.0, value=0.0, step=0.1)
        traffic_type = st.number_input("Traffic Type (1-20)", min_value=1, max_value=20, value=2)
        os = st.number_input("Operating System (1-8)", min_value=1, max_value=8, value=2)
        browser = st.number_input("Browser (1-13)", min_value=1, max_value=13, value=2)
        region = st.number_input("Region (1-9)", min_value=1, max_value=9, value=1)
    
    # 결과 표시 (우측)
    with col2:
        st.subheader("📊 예측 결과")
        
        # 인코딩
        month_encoded = np.where(np.array(metadata['le_month_classes']) == month)[0][0]
        visitor_encoded = np.where(np.array(metadata['le_visitor_classes']) == visitor_type)[0][0]
        weekend_int = 1 if weekend else 0
        
        # 로그 변환
        admin_dur_log = np.log1p(admin_dur)
        info_dur_log = np.log1p(info_dur)
        product_dur_log = np.log1p(product_dur)
        page_value_log = np.log1p(page_value)
        
        # 입력 딕셔너리
        input_data = {
            'Administrative': admin,
            'Administrative_Duration': admin_dur,
            'Informational': info,
            'Informational_Duration': info_dur,
            'ProductRelated': product,
            'ProductRelated_Duration': product_dur,
            'BounceRates': bounce,
            'ExitRates': exit_rate,
            'PageValues': page_value,
            'SpecialDay': special_day,
            'Month_encoded': month_encoded,
            'OperatingSystems': os,
            'Browser': browser,
            'Region': region,
            'TrafficType': traffic_type,
            'VisitorType_encoded': visitor_encoded,
            'Weekend_int': weekend_int,
            'ProductRelated_Duration_log': product_dur_log,
            'PageValues_log': page_value_log,
            'Administrative_Duration_log': admin_dur_log,
            'Informational_Duration_log': info_dur_log
        }
        
        # 예측
        if st.button("🔮 구매 확률 예측", key="predict_main", use_container_width=True):
            prediction = predict_purchase(input_data, model_name, scaler)
            avg_prob = metadata['average_purchase_proba'][model_name]
            
            emoji, css_class, color = get_probability_color(prediction)
            
            st.markdown(f"<div class='{css_class}' style='padding: 1.5rem; border-radius: 10px; margin: 1rem 0;'>", unsafe_allow_html=True)
            st.markdown(f"## {emoji} 구매 확률")
            st.markdown(f"# {prediction:.1%}", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
            # 평균 대비
            diff = prediction - avg_prob
            diff_pct = (diff / avg_prob) * 100 if avg_prob > 0 else 0
            
            if diff > 0:
                st.success(f"📈 평균보다 {diff_pct:+.1f}% 높음 (평균: {avg_prob:.1%})")
            else:
                st.warning(f"📉 평균보다 {diff_pct:+.1f}% 낮음 (평균: {avg_prob:.1%})")
            
            # 인사이트
            st.markdown("---")
            st.subheader("💡 인사이트")
            insights = generate_insight_text(input_data, prediction)
            if insights:
                for insight in insights:
                    st.markdown(f"- {insight}")
            else:
                st.info("인사이트를 생성할 정보가 부족합니다.")
            
            # 액션 추천
            st.markdown("---")
            st.subheader("🎯 마케팅 액션")
            action = generate_action_recommendation(input_data, prediction)
            st.info(action)
            
            # 모델별 예측 비교
            st.markdown("---")
            st.subheader("🤖 모델별 예측")
            model_results = {}
            for m in ['LR', 'RF', 'GB']:
                prob = predict_purchase(input_data, m, scaler)
                model_results[m] = prob
            
            col_lr, col_rf, col_gb = st.columns(3)
            with col_lr:
                st.metric("LR", f"{model_results['LR']:.1%}")
            with col_rf:
                st.metric("RF", f"{model_results['RF']:.1%}")
            with col_gb:
                st.metric("GB", f"{model_results['GB']:.1%}")

# ============================================================================

elif page == "2️⃣ What-If 시뮬레이터":
    st.header("🎮 What-If 시뮬레이터")
    st.markdown("슬라이더를 조작하여 예측 결과의 변화를 실시간으로 확인하세요.")
    
    # 기본값 설정 (평균값)
    st.subheader("📌 기본 세션 설정")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        admin_base = st.number_input("Administrative (기본)", min_value=0, max_value=30, value=2, key="admin_base")
        bounce_base = st.number_input("Bounce Rate (기본)", min_value=0.0, max_value=0.2, value=0.02, step=0.001, key="bounce_base")
    
    with col2:
        product_base = st.number_input("ProductRelated (기본)", min_value=0, max_value=100, value=20, key="product_base")
        exit_base = st.number_input("Exit Rate (기본)", min_value=0.0, max_value=0.2, value=0.05, step=0.001, key="exit_base")
    
    with col3:
        product_dur_base = st.number_input("ProductRelated Duration (기본)", min_value=0.0, value=500.0, step=50.0, key="product_dur_base")
        page_value_base = st.number_input("Page Values (기본)", min_value=0.0, value=10.0, step=5.0, key="page_value_base")
    
    # 기본값으로 예측
    input_base = {
        'Administrative': admin_base,
        'Administrative_Duration': 50.0,
        'Informational': 0,
        'Informational_Duration': 0.0,
        'ProductRelated': product_base,
        'ProductRelated_Duration': product_dur_base,
        'BounceRates': bounce_base,
        'ExitRates': exit_base,
        'PageValues': page_value_base,
        'SpecialDay': 0.0,
        'Month_encoded': 4,
        'OperatingSystems': 2,
        'Browser': 2,
        'Region': 1,
        'TrafficType': 2,
        'VisitorType_encoded': 0,
        'Weekend_int': 0,
        'ProductRelated_Duration_log': np.log1p(product_dur_base),
        'PageValues_log': np.log1p(page_value_base),
        'Administrative_Duration_log': np.log1p(50.0),
        'Informational_Duration_log': np.log1p(0.0)
    }
    
    prob_base = predict_purchase(input_base, model_name, scaler)
    
    st.markdown("---")
    st.subheader("🎚️ 시뮬레이션 슬라이더")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**행동 지표 조정**")
        bounce_sim = st.slider("Bounce Rate", 0.0, 0.2, bounce_base, 0.001, key="bounce_sim")
        exit_sim = st.slider("Exit Rate", 0.0, 0.2, exit_base, 0.001, key="exit_sim")
        page_value_sim = st.slider("Page Values", 0.0, 100.0, page_value_base, 5.0, key="page_value_sim")
    
    with col2:
        st.markdown("**페이지 탐색 조정**")
        product_sim = st.slider("ProductRelated 페이지", 0, 100, product_base, 5, key="product_sim")
        product_dur_sim = st.slider("ProductRelated Duration", 0.0, 2000.0, product_dur_base, 100.0, key="product_dur_sim")
        admin_sim = st.slider("Administrative 페이지", 0, 30, admin_base, 1, key="admin_sim")
    
    # 시뮬레이션 입력
    input_sim = input_base.copy()
    input_sim.update({
        'Administrative': admin_sim,
        'ProductRelated': product_sim,
        'ProductRelated_Duration': product_dur_sim,
        'BounceRates': bounce_sim,
        'ExitRates': exit_sim,
        'PageValues': page_value_sim,
        'ProductRelated_Duration_log': np.log1p(product_dur_sim),
        'PageValues_log': np.log1p(page_value_sim)
    })
    
    prob_sim = predict_purchase(input_sim, model_name, scaler)
    
    # 결과 비교
    st.markdown("---")
    st.subheader("📊 시뮬레이션 결과")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("기본 구매 확률", f"{prob_base:.1%}")
    with col2:
        st.metric("시뮬레이션 확률", f"{prob_sim:.1%}")
    with col3:
        change = prob_sim - prob_base
        st.metric("변화", f"{change:+.1%}", delta=f"{change:+.1%p}")
    
    # 그래프
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=['기본 상태', '시뮬레이션'],
        y=[prob_base * 100, prob_sim * 100],
        marker_color=['#ffc107', '#28a745'],
        text=[f"{prob_base:.1%}", f"{prob_sim:.1%}"],
        textposition='outside',
        name='구매 확률'
    ))
    fig.update_layout(
        title="구매 확률 변화",
        yaxis_title="구매 확률 (%)",
        height=400,
        showlegend=False,
        hovermode='x'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # 인사이트
    st.markdown("---")
    if prob_sim > prob_base:
        increase = ((prob_sim - prob_base) / prob_base) * 100
        st.success(f"✅ {increase:.1f}% 구매 확률 상승!")
        st.info(f"💡 '{change:+.1%p}' 변화로 구매 확률이 향상되었습니다. 이 전략을 실행해보세요.")
    else:
        decrease = ((prob_base - prob_sim) / prob_base) * 100
        st.warning(f"❌ {decrease:.1f}% 구매 확률 감소!")
        st.info("💡 현재 설정이 구매 의도를 감소시키고 있습니다. 다른 조합을 시도해보세요.")

# ============================================================================

elif page == "3️⃣ 채널 효과 분석":
    st.header("📊 채널 효과 분석 대시보드")
    st.markdown("TrafficType, Region, Browser별 구매 성과를 분석합니다.")
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.subheader("🔧 필터 설정")
        category = st.radio("분석 기준", ["TrafficType", "Region", "Browser", "OperatingSystems"])
        metric = st.selectbox("지표", ["구매율 (%)", "세션 수", "평균 구매 확률 (%)"])
    
    with col2:
        st.subheader(f"🎯 {category} 분석")
        
        # 데이터 집계
        if metric == "구매율 (%)":
            agg_data = df.groupby(category)['Revenue'].agg(['sum', 'count'])
            agg_data.columns = ['Purchase', 'Total']
            agg_data['Rate'] = (agg_data['Purchase'] / agg_data['Total'] * 100).round(2)
            agg_data = agg_data.sort_values('Rate', ascending=False).head(10)
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=agg_data.index,
                y=agg_data['Rate'],
                marker_color='#667eea',
                text=agg_data['Rate'].apply(lambda x: f"{x:.1f}%"),
                textposition='outside'
            ))
            fig.update_layout(
                title=f"{category}별 구매율 (상위 10)",
                xaxis_title=category,
                yaxis_title="구매율 (%)",
                height=400,
                hovermode='x'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        elif metric == "세션 수":
            agg_data = df[category].value_counts().head(10)
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=agg_data.index,
                y=agg_data.values,
                marker_color='#764ba2',
                text=agg_data.values,
                textposition='outside'
            ))
            fig.update_layout(
                title=f"{category}별 세션 수 (상위 10)",
                xaxis_title=category,
                yaxis_title="세션 수",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        else:  # 평균 구매 확률
            group_data = df.groupby(category)['Revenue'].mean() * 100
            group_data = group_data.sort_values(ascending=False).head(10)
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=group_data.index,
                y=group_data.values,
                marker_color='#28a745',
                text=group_data.values.apply(lambda x: f"{x:.1f}%"),
                textposition='outside'
            ))
            fig.update_layout(
                title=f"{category}별 평균 구매율 (상위 10)",
                xaxis_title=category,
                yaxis_title="평균 구매율 (%)",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # 세부 테이블
        st.markdown("---")
        st.markdown("**📋 세부 데이터**")
        
        detail_data = df.groupby(category).agg({
            'Revenue': ['sum', 'count', 'mean']
        }).round(3)
        detail_data.columns = ['구매', '총세션', '구매율']
        detail_data['구매율'] = (detail_data['구매율'] * 100).round(2)
        detail_data = detail_data.sort_values('구매율', ascending=False).head(15)
        
        st.dataframe(detail_data, use_container_width=True)

# ============================================================================

elif page == "4️⃣ 이탈 세션 탐지":
    st.header("⚠️ 고위험 이탈 세션 탐지기")
    st.markdown("세션 정보를 입력하여 이탈 위험도를 진단하고 액션 추천을 받으세요.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📝 세션 정보")
        bounce_risk = st.slider("Bounce Rate", 0.0, 0.2, 0.05)
        exit_risk = st.slider("Exit Rate", 0.0, 0.2, 0.08)
        page_value_risk = st.number_input("Page Values", min_value=0.0, value=5.0)
        product_dur_risk = st.number_input("ProductRelated Duration", min_value=0.0, value=300.0)
        product_pages = st.number_input("ProductRelated Pages", min_value=0, value=10)
    
    with col2:
        st.subheader("🎯 위험도 진단")
        
        input_risk = {
            'Administrative': 2,
            'Administrative_Duration': 50.0,
            'Informational': 0,
            'Informational_Duration': 0.0,
            'ProductRelated': product_pages,
            'ProductRelated_Duration': product_dur_risk,
            'BounceRates': bounce_risk,
            'ExitRates': exit_risk,
            'PageValues': page_value_risk,
            'SpecialDay': 0.0,
            'Month_encoded': 4,
            'OperatingSystems': 2,
            'Browser': 2,
            'Region': 1,
            'TrafficType': 2,
            'VisitorType_encoded': 0,
            'Weekend_int': 0,
            'ProductRelated_Duration_log': np.log1p(product_dur_risk),
            'PageValues_log': np.log1p(page_value_risk),
            'Administrative_Duration_log': np.log1p(50.0),
            'Informational_Duration_log': np.log1p(0.0)
        }
        
        prob_risk = predict_purchase(input_risk, model_name, scaler)
        churn_risk = 1 - prob_risk
        
        # 위험도 게이지
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=churn_risk * 100,
            title="이탈 위험도",
            domain={'x': [0, 1], 'y': [0, 1]},
            delta={'reference': 50},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#dc3545"},
                'steps': [
                    {'range': [0, 33], 'color': "#d4edda"},
                    {'range': [33, 66], 'color': "#fff3cd"},
                    {'range': [66, 100], 'color': "#f8d7da"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 70
                }
            }
        ))
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # 위험도 평가
        if churn_risk < 0.3:
            st.success("✅ 낮은 위험 - 구매 가능성 높음")
        elif churn_risk < 0.7:
            st.warning("⚠️ 중간 위험 - 적극적 개입 필요")
        else:
            st.error("🔴 높은 위험 - 즉시 액션 필요")
    
    # 액션 추천
    st.markdown("---")
    st.subheader("🎯 맞춤형 액션 추천")
    
    recommendations = []
    
    if bounce_risk > 0.1:
        recommendations.append({
            '문제': "높은 이탈률 (>10%)",
            '원인': "랜딩 페이지 매칭 불일치",
            '액션': "페이지 로딩 속도 개선, 핵심 정보 강조"
        })
    
    if exit_risk > 0.1:
        recommendations.append({
            '문제': "높은 이탈률 (>10%)",
            '원인': "체크아웃 프로세스 복잡",
            '액션': "1-Click 결제, 게스트 체크아웃 추가"
        })
    
    if page_value_risk < 1:
        recommendations.append({
            '문제': "낮은 페이지 가치",
            '원인': "가치 페이지 도달 실패",
            '액션': "상품 추천 알고리즘 강화, 번들 상품 제안"
        })
    
    if product_dur_risk < 200:
        recommendations.append({
            '문제': "짧은 체류 시간",
            '원인': "상품 탐색 부족",
            '액션': "유사 상품 추천, 비교 기능 강화"
        })
    
    if recommendations:
        for i, rec in enumerate(recommendations, 1):
            st.markdown(f"**{i}. {rec['문제']}**")
            st.write(f"- 원인: {rec['원인']}")
            st.write(f"- 액션: {rec['액션']}")
            st.markdown("---")
    else:
        st.info("현재 세션은 특별한 위험 신호가 없습니다.")

# ============================================================================

elif page == "5️⃣ EDA 대시보드":
    st.header("📊 탐색적 데이터 분석 (EDA) 대시보드")
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 Overview", "👥 Behavior", "🎯 Features", "🔗 Correlation", "🤖 Model"])
    
    with tab1:
        st.subheader("데이터셋 요약")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("총 세션", f"{metadata['data_summary']['total_sessions']:,}")
        with col2:
            st.metric("구매 세션", f"{metadata['class_distribution']['Purchase']:,}")
        with col3:
            st.metric("구매율", f"{metadata['class_distribution']['Purchase_ratio']:.1%}")
        with col4:
            st.metric("피처 수", f"{len(feature_cols_final)}")
        
        st.markdown("---")
        
        # Revenue 분포
        revenue_counts = df['Revenue'].value_counts()
        fig = go.Figure(data=[
            go.Pie(
                labels=['비구매', '구매'],
                values=[revenue_counts[False], revenue_counts[True]],
                marker=dict(colors=['#dc3545', '#28a745']),
                textinfo='label+percent',
                textposition='inside',
                hovertemplate='%{label}: %{value} (%{percent})'
            )
        ])
        fig.update_layout(title="Revenue 분포", height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("사용자 행동 분석")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Month별 구매율
            month_data = df.groupby('Month')['Revenue'].agg(['sum', 'count'])
            month_data['rate'] = (month_data['sum'] / month_data['count'] * 100).round(2)
            month_order = ['Feb', 'Mar', 'May', 'June', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            month_data = month_data.reindex([m for m in month_order if m in month_data.index])
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=month_data.index,
                y=month_data['rate'],
                mode='lines+markers',
                marker=dict(size=10, color='#667eea'),
                line=dict(width=2),
                fill='tozeroy',
                name='구매율'
            ))
            fig.update_layout(
                title="월별 구매 전환율",
                xaxis_title="월",
                yaxis_title="구매율 (%)",
                height=400,
                hovermode='x'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # VisitorType별 구매율
            visitor_data = df.groupby('VisitorType')['Revenue'].agg(['sum', 'count'])
            visitor_data['rate'] = (visitor_data['sum'] / visitor_data['count'] * 100).round(2)
            visitor_data = visitor_data.sort_values('rate', ascending=False)
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=visitor_data.index,
                y=visitor_data['rate'],
                marker_color=['#28a745', '#ffc107', '#dc3545'],
                text=visitor_data['rate'].apply(lambda x: f"{x:.1f}%"),
                textposition='outside'
            ))
            fig.update_layout(
                title="방문자 유형별 구매율",
                xaxis_title="방문자 유형",
                yaxis_title="구매율 (%)",
                height=400,
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Weekend vs Weekday
        weekend_data = df.groupby('Weekend')['Revenue'].agg(['sum', 'count'])
        weekend_data['rate'] = (weekend_data['sum'] / weekend_data['count'] * 100).round(2)
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=['평일', '주말'],
            y=weekend_data['rate'].values,
            marker_color=['#667eea', '#764ba2'],
            text=weekend_data['rate'].values.astype(str) + '%',
            textposition='outside'
        ))
        fig.update_layout(
            title="평일/주말 구매율",
            yaxis_title="구매율 (%)",
            height=300,
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("피처 분포 분석")
        
        feature_to_plot = st.selectbox(
            "분석할 피처 선택",
            ['PageValues', 'BounceRates', 'ExitRates', 'ProductRelated_Duration']
        )
        
        # Histogram with Revenue overlay
        fig = go.Figure()
        
        for revenue in [False, True]:
            label = '구매' if revenue else '비구매'
            color = '#28a745' if revenue else '#dc3545'
            fig.add_trace(go.Histogram(
                x=df[df['Revenue'] == revenue][feature_to_plot],
                name=label,
                marker_color=color,
                opacity=0.7,
                nbinsx=30
            ))
        
        fig.update_layout(
            title=f"{feature_to_plot} 분포 (Revenue별)",
            xaxis_title=feature_to_plot,
            yaxis_title="빈도",
            barmode='overlay',
            height=400,
            hovermode='x'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.subheader("상관관계 분석")
        
        # 상관계수 데이터 로드 (메타데이터에서)
        perf = metadata['model_performance']
        
        st.markdown("**수치형 변수와 Revenue의 상관계수**")
        
        # 간단한 상관계수 표시
        key_features = ['PageValues_log', 'ExitRates', 'ProductRelated', 'BounceRates', 'ProductRelated_Duration_log']
        correlations = []
        
        for feat in key_features:
            if feat in df.columns:
                corr = df[feat].corr(df['Revenue'].astype(int))
                correlations.append({'피처': feat, '상관계수': round(corr, 4)})
        
        corr_df_display = pd.DataFrame(correlations).sort_values('상관계수', ascending=False)
        st.dataframe(corr_df_display, use_container_width=True)
    
    with tab5:
        st.subheader("모델 성능 비교")
        
        perf = pd.DataFrame(metadata['model_performance'])
        st.dataframe(perf, use_container_width=True)
        
        # 성능 시각화
        fig = go.Figure()
        
        for metric in ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']:
            fig.add_trace(go.Bar(
                x=perf['Model'],
                y=perf[metric],
                name=metric
            ))
        
        fig.update_layout(
            title="모델별 성능 메트릭",
            xaxis_title="모델",
            yaxis_title="점수",
            height=400,
            barmode='group'
        )
        st.plotly_chart(fig, use_container_width=True)

# ============================================================================

elif page == "6️⃣ Feature Importance":
    st.header("🎯 Feature Importance & 피처 분석")
    
    if model_name in ['RF', 'GB']:
        st.subheader(f"{model_name} 모델의 피처 중요도")
        
        feature_imp = models[model_name]['feature_importance']
        feature_imp_sorted = dict(sorted(feature_imp.items(), key=lambda x: x[1], reverse=True))
        
        # Top 15 피처
        top_features = dict(list(feature_imp_sorted.items())[:15])
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=list(top_features.keys()),
            x=list(top_features.values()),
            orientation='h',
            marker_color='#667eea',
            text=[f"{v:.4f}" for v in top_features.values()],
            textposition='outside'
        ))
        fig.update_layout(
            title="Feature Importance (Top 15)",
            xaxis_title="중요도",
            height=500,
            yaxis={'categoryorder': 'total ascending'}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # 피처별 상세 분석
        st.markdown("---")
        st.subheader("🔍 피처별 상세 분석")
        
        selected_feature = st.selectbox("분석할 피처 선택", list(top_features.keys()))
        
        if selected_feature in df.columns:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"**{selected_feature} 분포 (비구매)**")
                fig1 = go.Figure()
                fig1.add_trace(go.Histogram(
                    x=df[df['Revenue'] == False][selected_feature],
                    name='비구매',
                    marker_color='#dc3545'
                ))
                fig1.update_layout(height=300, showlegend=False)
                st.plotly_chart(fig1, use_container_width=True)
            
            with col2:
                st.markdown(f"**{selected_feature} 분포 (구매)**")
                fig2 = go.Figure()
                fig2.add_trace(go.Histogram(
                    x=df[df['Revenue'] == True][selected_feature],
                    name='구매',
                    marker_color='#28a745'
                ))
                fig2.update_layout(height=300, showlegend=False)
                st.plotly_chart(fig2, use_container_width=True)
    
    else:
        st.info("Logistic Regression은 Feature Importance 분석이 제한적입니다.")
        st.markdown("Coefficients를 대신 표시합니다.")

# ============================================================================

elif page == "7️⃣ 고객 페르소나":
    st.header("👤 가상 고객 페르소나 생성기")
    st.markdown("사전 정의된 페르소나 중 하나를 선택하면 해당 세션의 구매 확률을 예측합니다.")
    
    personas = {
        "🆕 신규 고객 (정보 탐색)": {
            'description': "첫 방문, 상품 정보만 탐색, 결제 의도 낮음",
            'params': {
                'admin': 1, 'info': 2, 'product': 5,
                'admin_dur': 20, 'info_dur': 100, 'product_dur': 100,
                'bounce': 0.08, 'exit': 0.12, 'page_value': 0,
                'visitor': 'New_Visitor', 'month': 'May'
            }
        },
        "🔄 재방문자 (상품 비교)": {
            'description': "재방문, 상품 페이지 깊게 탐색, 비교 중",
            'params': {
                'admin': 2, 'info': 1, 'product': 35,
                'admin_dur': 50, 'info_dur': 20, 'product_dur': 800,
                'bounce': 0.01, 'exit': 0.03, 'page_value': 20,
                'visitor': 'Returning_Visitor', 'month': 'Nov'
            }
        },
        "💰 구매 직전 (고의도 구매자)": {
            'description': "높은 이탈 위험, 장바구니 단계 추정",
            'params': {
                'admin': 3, 'info': 0, 'product': 25,
                'admin_dur': 80, 'info_dur': 0, 'product_dur': 1200,
                'bounce': 0.005, 'exit': 0.02, 'page_value': 50,
                'visitor': 'Returning_Visitor', 'month': 'Nov'
            }
        },
        "❌ 높은 이탈 위험": {
            'description': "높은 이탈률, 랜딩 페이지 매칭 문제",
            'params': {
                'admin': 0, 'info': 0, 'product': 2,
                'admin_dur': 0, 'info_dur': 0, 'product_dur': 50,
                'bounce': 0.15, 'exit': 0.15, 'page_value': 0,
                'visitor': 'New_Visitor', 'month': 'Feb'
            }
        },
        "🎯 이상적인 고객": {
            'description': "낮은 이탈, 높은 참여, 구매 가능성 높음",
            'params': {
                'admin': 5, 'info': 1, 'product': 50,
                'admin_dur': 100, 'info_dur': 30, 'product_dur': 2000,
                'bounce': 0.001, 'exit': 0.01, 'page_value': 100,
                'visitor': 'New_Visitor', 'month': 'Nov'
            }
        }
    }
    
    persona_name = st.selectbox("페르소나 선택", list(personas.keys()))
    persona = personas[persona_name]
    
    st.markdown(f"**{persona_name}**")
    st.info(persona['description'])
    
    # 파라미터 표시
    params = persona['params']
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**세션 정보**")
        st.write(f"- Admin Pages: {params['admin']}")
        st.write(f"- Info Pages: {params['info']}")
        st.write(f"- Product Pages: {params['product']}")
    
    with col2:
        st.markdown("**행동 지표**")
        st.write(f"- Bounce Rate: {params['bounce']:.3f}")
        st.write(f"- Exit Rate: {params['exit']:.3f}")
        st.write(f"- Page Values: {params['page_value']}")
    
    st.markdown("---")
    
    if st.button("🔮 페르소나 예측", use_container_width=True):
        # 데이터 구성
        month_encoded = np.where(np.array(metadata['le_month_classes']) == params['month'])[0][0]
        visitor_encoded = np.where(np.array(metadata['le_visitor_classes']) == params['visitor'])[0][0]
        
        input_persona = {
            'Administrative': params['admin'],
            'Administrative_Duration': params['admin_dur'],
            'Informational': params['info'],
            'Informational_Duration': params['info_dur'],
            'ProductRelated': params['product'],
            'ProductRelated_Duration': params['product_dur'],
            'BounceRates': params['bounce'],
            'ExitRates': params['exit'],
            'PageValues': params['page_value'],
            'SpecialDay': 0.0,
            'Month_encoded': month_encoded,
            'OperatingSystems': 2,
            'Browser': 2,
            'Region': 1,
            'TrafficType': 2,
            'VisitorType_encoded': visitor_encoded,
            'Weekend_int': 0,
            'ProductRelated_Duration_log': np.log1p(params['product_dur']),
            'PageValues_log': np.log1p(params['page_value']),
            'Administrative_Duration_log': np.log1p(params['admin_dur']),
            'Informational_Duration_log': np.log1p(params['info_dur'])
        }
        
        prob_persona = predict_purchase(input_persona, model_name, scaler)
        
        emoji, css_class, color = get_probability_color(prob_persona)
        
        st.markdown(f"<div class='{css_class}' style='padding: 1.5rem; border-radius: 10px; margin: 1rem 0;'>", unsafe_allow_html=True)
        st.markdown(f"## {emoji} 예측 구매 확률")
        st.markdown(f"# {prob_persona:.1%}", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        insights = generate_insight_text(input_persona, prob_persona)
        if insights:
            st.markdown("**💡 특징:**")
            for insight in insights:
                st.markdown(f"- {insight}")
        
        action = generate_action_recommendation(input_persona, prob_persona)
        st.markdown(f"**🎯 추천 액션:**\n{action}")

# ============================================================================

elif page == "8️⃣ 시나리오 비교":
    st.header("⚖️ 시나리오 A vs B 비교")
    st.markdown("두 세션을 입력하여 예측 결과를 나란히 비교합니다.")
    
    col1, col2 = st.columns(2)
    
    # Scenario A
    with col1:
        st.subheader("시나리오 A")
        
        a_admin = st.number_input("Admin Pages (A)", min_value=0, max_value=30, value=2, key="a_admin")
        a_product = st.number_input("Product Pages (A)", min_value=0, max_value=100, value=15, key="a_product")
        a_bounce = st.slider("Bounce Rate (A)", 0.0, 0.2, 0.03, key="a_bounce")
        a_exit = st.slider("Exit Rate (A)", 0.0, 0.2, 0.05, key="a_exit")
        a_pagevalue = st.number_input("Page Values (A)", min_value=0.0, value=5.0, key="a_pagevalue")
        a_visitor = st.selectbox("Visitor Type (A)", metadata['le_visitor_classes'], index=0, key="a_visitor")
    
    # Scenario B
    with col2:
        st.subheader("시나리오 B")
        
        b_admin = st.number_input("Admin Pages (B)", min_value=0, max_value=30, value=3, key="b_admin")
        b_product = st.number_input("Product Pages (B)", min_value=0, max_value=100, value=25, key="b_product")
        b_bounce = st.slider("Bounce Rate (B)", 0.0, 0.2, 0.015, key="b_bounce")
        b_exit = st.slider("Exit Rate (B)", 0.0, 0.2, 0.025, key="b_exit")
        b_pagevalue = st.number_input("Page Values (B)", min_value=0.0, value=15.0, key="b_pagevalue")
        b_visitor = st.selectbox("Visitor Type (B)", metadata['le_visitor_classes'], index=2, key="b_visitor")
    
    if st.button("📊 비교 분석 시작", use_container_width=True, use_container_width=True):
        # Scenario A 예측
        a_visitor_encoded = np.where(np.array(metadata['le_visitor_classes']) == a_visitor)[0][0]
        input_a = {
            'Administrative': a_admin,
            'Administrative_Duration': 50.0,
            'Informational': 0,
            'Informational_Duration': 0.0,
            'ProductRelated': a_product,
            'ProductRelated_Duration': 400.0,
            'BounceRates': a_bounce,
            'ExitRates': a_exit,
            'PageValues': a_pagevalue,
            'SpecialDay': 0.0,
            'Month_encoded': 4,
            'OperatingSystems': 2,
            'Browser': 2,
            'Region': 1,
            'TrafficType': 2,
            'VisitorType_encoded': a_visitor_encoded,
            'Weekend_int': 0,
            'ProductRelated_Duration_log': np.log1p(400.0),
            'PageValues_log': np.log1p(a_pagevalue),
            'Administrative_Duration_log': np.log1p(50.0),
            'Informational_Duration_log': np.log1p(0.0)
        }
        
        # Scenario B 예측
        b_visitor_encoded = np.where(np.array(metadata['le_visitor_classes']) == b_visitor)[0][0]
        input_b = {
            'Administrative': b_admin,
            'Administrative_Duration': 50.0,
            'Informational': 0,
            'Informational_Duration': 0.0,
            'ProductRelated': b_product,
            'ProductRelated_Duration': 600.0,
            'BounceRates': b_bounce,
            'ExitRates': b_exit,
            'PageValues': b_pagevalue,
            'SpecialDay': 0.0,
            'Month_encoded': 4,
            'OperatingSystems': 2,
            'Browser': 2,
            'Region': 1,
            'TrafficType': 2,
            'VisitorType_encoded': b_visitor_encoded,
            'Weekend_int': 0,
            'ProductRelated_Duration_log': np.log1p(600.0),
            'PageValues_log': np.log1p(b_pagevalue),
            'Administrative_Duration_log': np.log1p(50.0),
            'Informational_Duration_log': np.log1p(0.0)
        }
        
        prob_a = predict_purchase(input_a, model_name, scaler)
        prob_b = predict_purchase(input_b, model_name, scaler)
        
        # 비교 표시
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("시나리오 A", f"{prob_a:.1%}")
        
        with col2:
            winner = "B 승리 🎉" if prob_b > prob_a else ("A 승리 🎉" if prob_a > prob_b else "동점")
            st.metric("차이", f"{abs(prob_b - prob_a):+.1%}", delta=winner)
        
        with col3:
            st.metric("시나리오 B", f"{prob_b:.1%}")
        
        # 비교 그래프
        st.markdown("---")
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=['시나리오 A', '시나리오 B'],
            y=[prob_a * 100, prob_b * 100],
            marker_color=['#667eea', '#764ba2'],
            text=[f"{prob_a:.1%}", f"{prob_b:.1%}"],
            textposition='outside'
        ))
        fig.update_layout(
            title="구매 확률 비교",
            yaxis_title="구매 확률 (%)",
            height=400,
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # 결론
        if prob_b > prob_a:
            st.success(f"✅ 시나리오 B가 더 나은 선택입니다 ({prob_b - prob_a:+.1%p} 상승)")
        elif prob_a > prob_b:
            st.success(f"✅ 시나리오 A가 더 나은 선택입니다 ({prob_a - prob_b:+.1%p} 상승)")
        else:
            st.info("두 시나리오가 동등합니다.")

# ============================================================================

elif page == "9️⃣ 모델 성능 비교":
    st.header("🤖 모델 성능 비교")
    st.markdown("LR, RF, GB 3가지 모델의 성능을 비교합니다.")
    
    # 성능 표
    st.subheader("📊 모델별 성능 메트릭")
    perf_df = pd.DataFrame(metadata['model_performance'])
    st.dataframe(perf_df, use_container_width=True)
    
    # 메트릭별 비교
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**정확도 비교 (Accuracy & ROC-AUC)**")
        fig1 = go.Figure()
        fig1.add_trace(go.Bar(
            x=perf_df['Model'],
            y=perf_df['Accuracy'],
            name='Accuracy',
            marker_color='#667eea'
        ))
        fig1.add_trace(go.Bar(
            x=perf_df['Model'],
            y=perf_df['ROC-AUC'],
            name='ROC-AUC',
            marker_color='#764ba2'
        ))
        fig1.update_layout(
            height=400,
            barmode='group',
            hovermode='x'
        )
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        st.markdown("**정밀도 vs 재현율 (Precision vs Recall)**")
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            x=perf_df['Model'],
            y=perf_df['Precision'],
            name='Precision',
            marker_color='#28a745'
        ))
        fig2.add_trace(go.Bar(
            x=perf_df['Model'],
            y=perf_df['Recall'],
            name='Recall',
            marker_color='#ffc107'
        ))
        fig2.update_layout(
            height=400,
            barmode='group',
            hovermode='x'
        )
        st.plotly_chart(fig2, use_container_width=True)
    
    # 모델 추천
    st.markdown("---")
    st.subheader("🏆 모델 추천")
    
    best_accuracy = perf_df.loc[perf_df['Accuracy'].idxmax()]
    best_auc = perf_df.loc[perf_df['ROC-AUC'].idxmax()]
    best_f1 = perf_df.loc[perf_df['F1-Score'].idxmax()]
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info(f"**정확도 최고:**\n{best_accuracy['Model']}\n({best_accuracy['Accuracy']:.4f})")
    
    with col2:
        st.warning(f"**ROC-AUC 최고:**\n{best_auc['Model']}\n({best_auc['ROC-AUC']:.4f})")
    
    with col3:
        st.success(f"**F1-Score 최고:**\n{best_f1['Model']}\n({best_f1['F1-Score']:.4f})")
    
    st.markdown("---")
    st.markdown("""
    **💡 모델 선택 가이드:**
    - **LR (Logistic Regression)**: 해석 가능성이 중요할 때, 속도가 중요할 때
    - **RF (Random Forest)**: 안정성과 성능의 균형
    - **GB (Gradient Boosting)**: 최고 성능 필요, ROC-AUC 최고 (추천)
    """)

# ============================================================================

elif page == "🔟 액션 추천 카드":
    st.header("💡 마케팅 액션 추천")
    st.markdown("세션 분석을 통해 맞춤형 마케팅 액션을 추천합니다.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📝 세션 정보")
        
        scenario = st.radio(
            "시나리오 선택",
            [
                "Case A: 상품 관심 높음 + 낮은 구매 확률",
                "Case B: 높은 이탈률",
                "Case C: 구매 직전",
                "Case D: 커스텀 입력"
            ]
        )
        
        if scenario == "Case A: 상품 관심 높음 + 낮은 구매 확률":
            input_action = {
                'Administrative': 3,
                'Administrative_Duration': 80.0,
                'Informational': 0,
                'Informational_Duration': 0.0,
                'ProductRelated': 30,
                'ProductRelated_Duration': 1000.0,
                'BounceRates': 0.01,
                'ExitRates': 0.02,
                'PageValues': 3.0,
                'SpecialDay': 0.0,
                'Month_encoded': 4,
                'OperatingSystems': 2,
                'Browser': 2,
                'Region': 1,
                'TrafficType': 2,
                'VisitorType_encoded': 0,
                'Weekend_int': 0,
                'ProductRelated_Duration_log': np.log1p(1000.0),
                'PageValues_log': np.log1p(3.0),
                'Administrative_Duration_log': np.log1p(80.0),
                'Informational_Duration_log': np.log1p(0.0)
            }
        
        elif scenario == "Case B: 높은 이탈률":
            input_action = {
                'Administrative': 1,
                'Administrative_Duration': 20.0,
                'Informational': 0,
                'Informational_Duration': 0.0,
                'ProductRelated': 3,
                'ProductRelated_Duration': 50.0,
                'BounceRates': 0.15,
                'ExitRates': 0.15,
                'PageValues': 0.0,
                'SpecialDay': 0.0,
                'Month_encoded': 2,
                'OperatingSystems': 2,
                'Browser': 2,
                'Region': 1,
                'TrafficType': 2,
                'VisitorType_encoded': 0,
                'Weekend_int': 0,
                'ProductRelated_Duration_log': np.log1p(50.0),
                'PageValues_log': np.log1p(0.0),
                'Administrative_Duration_log': np.log1p(20.0),
                'Informational_Duration_log': np.log1p(0.0)
            }
        
        elif scenario == "Case C: 구매 직전":
            input_action = {
                'Administrative': 4,
                'Administrative_Duration': 100.0,
                'Informational': 0,
                'Informational_Duration': 0.0,
                'ProductRelated': 40,
                'ProductRelated_Duration': 1500.0,
                'BounceRates': 0.001,
                'ExitRates': 0.01,
                'PageValues': 80.0,
                'SpecialDay': 0.0,
                'Month_encoded': 7,
                'OperatingSystems': 2,
                'Browser': 2,
                'Region': 1,
                'TrafficType': 2,
                'VisitorType_encoded': 2,
                'Weekend_int': 0,
                'ProductRelated_Duration_log': np.log1p(1500.0),
                'PageValues_log': np.log1p(80.0),
                'Administrative_Duration_log': np.log1p(100.0),
                'Informational_Duration_log': np.log1p(0.0)
            }
        
        else:  # Custom
            input_action = {
                'Administrative': st.number_input("Admin Pages", min_value=0, max_value=30, value=2),
                'Administrative_Duration': st.number_input("Admin Duration", min_value=0.0, value=50.0),
                'Informational': st.number_input("Info Pages", min_value=0, max_value=30, value=0),
                'Informational_Duration': st.number_input("Info Duration", min_value=0.0, value=0.0),
                'ProductRelated': st.number_input("Product Pages", min_value=0, max_value=100, value=20),
                'ProductRelated_Duration': st.number_input("Product Duration", min_value=0.0, value=500.0),
                'BounceRates': st.number_input("Bounce Rate", min_value=0.0, max_value=0.2, value=0.05),
                'ExitRates': st.number_input("Exit Rate", min_value=0.0, max_value=0.2, value=0.08),
                'PageValues': st.number_input("Page Values", min_value=0.0, value=10.0),
                'SpecialDay': 0.0,
                'Month_encoded': 4,
                'OperatingSystems': 2,
                'Browser': 2,
                'Region': 1,
                'TrafficType': 2,
                'VisitorType_encoded': 0,
                'Weekend_int': 0,
                'ProductRelated_Duration_log': np.log1p(st.number_input("Product Duration", min_value=0.0, value=500.0)),
                'PageValues_log': np.log1p(st.number_input("Page Values", min_value=0.0, value=10.0)),
                'Administrative_Duration_log': np.log1p(st.number_input("Admin Duration", min_value=0.0, value=50.0)),
                'Informational_Duration_log': np.log1p(st.number_input("Info Duration", min_value=0.0, value=0.0))
            }
    
    with col2:
        st.subheader("🎯 추천 액션")
        
        prob_action = predict_purchase(input_action, model_name, scaler)
        
        st.markdown(f"**예측 구매 확률:** {prob_action:.1%}")
        
        # 액션 카드들
        action_rec = generate_action_recommendation(input_action, prob_action)
        st.warning(action_rec)
        
        st.markdown("---")
        
        # 세부 액션 제안
        st.markdown("**💼 실행 액션 (우선순위)**")
        
        page_values = input_action.get('PageValues', 0)
        bounce = input_action.get('BounceRates', 0)
        exit_rate = input_action.get('ExitRates', 0)
        product_dur = input_action.get('ProductRelated_Duration', 0)
        
        actions_priority = []
        
        if prob_action < 0.15:
            actions_priority.append({
                'priority': 1,
                'action': '🎯 긴급 개입 필요',
                'details': ['가격 할인/쿠폰 제공', '리뷰/신뢰 요소 강조', '제한 시간 프로모션']
            })
        
        if bounce > 0.1 or exit_rate > 0.1:
            actions_priority.append({
                'priority': 2,
                'action': '🔧 페이지 최적화',
                'details': ['로딩 속도 개선', '모바일 최적화', '결제 프로세스 단순화']
            })
        
        if page_values < 1:
            actions_priority.append({
                'priority': 3,
                'action': '📱 가치 페이지 유도',
                'details': ['상품 추천 강화', '번들/세트 상품 제안', '리뷰 페이지 노출']
            })
        
        if product_dur < 300:
            actions_priority.append({
                'priority': 4,
                'action': '🔍 정보 강화',
                'details': ['상품 비교 기능', '상세 이미지/영상', '고객 리뷰 추가']
            })
        
        if actions_priority:
            for ap in actions_priority:
                st.markdown(f"**{ap['priority']}. {ap['action']}**")
                for detail in ap['details']:
                    st.markdown(f"   - {detail}")
        else:
            st.success("✅ 현재 상태로도 충분한 전환 신호가 있습니다!")

# ============================================================================
# 푸터
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #999; font-size: 12px;'>
    <p>🛒 Online Shoppers Purchasing Intention Prediction System</p>
    <p>데이터 기반 e-commerce 전환율 최적화 도구</p>
    <p>Models: Logistic Regression | Random Forest | Gradient Boosting</p>
</div>
""", unsafe_allow_html=True)
