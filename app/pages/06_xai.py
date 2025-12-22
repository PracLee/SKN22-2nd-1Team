# app/pages/06_xai.py
import streamlit as st
from ui.header import render_header
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import os
import platform

render_header()
st.set_page_config(page_title="XAI", layout="wide")

# --- 폰트 및 마이너스 설정 ---
def setup_font():
    os_name = platform.system()
    if os_name == 'Windows':
        plt.rc('font', family='Malgun Gothic')
    elif os_name == 'Darwin':
        plt.rc('font', family='AppleGothic')
    else:
        plt.rc('font', family='NanumGothic') 
    plt.rcParams['axes.unicode_minus'] = False

setup_font()

# --- 컬럼 한글 매핑 정의 ---
col_mapping = {
    'Administrative': '관리 페이지 조회 수',
    'Administrative_Duration': '관리 페이지 체류 시간',
    'Informational': '정보 페이지 조회 수',
    'Informational_Duration': '정보 페이지 체류 시간',
    'ProductRelated': '제품 관련 페이지 조회 수',
    'ProductRelated_Duration': '제품 관련 페이지 체류 시간',
    'BounceRates': '이탈률',
    'ExitRates': '종료율',
    'PageValues': '페이지 가치',
    'SpecialDay': '기념일',
    'Month': '월',
    'OperatingSystems': '운영체제',
    'Browser': '브라우저',
    'Region': '지역',
    'TrafficType': '트래픽 유형',
    'VisitorType_New_Visitor': '방문자 유형_신규',
    'VisitorType_Returning_Visitor': '방문자 유형_재방문',
    'Weekend': '주말 여부',
    'Revenue': '구매 여부',
    'row_id': '행 인덱스',
    'Month_Nov': '11월',
    'Month_May': '5월',
    'Month_Dec': '12월',
    'Month_Mar': '3월',
    'Month_Sep': '9월',
}

# --- 데이터 및 모델 로드 ---
@st.cache_resource
def load_resources():
    # 경로 설정
    curr_path = Path(__file__).resolve()
    app_root = curr_path.parent.parent
    project_root = app_root.parent
    
    # 데이터 로드
    data_path = project_root / "data" / "processed" / "test.csv"
    if not data_path.exists():
        st.error(f"데이터 파일을 찾을 수 없습니다: {data_path}")
        st.stop()
    df = pd.read_csv(data_path)
    
    # 모델 로드
    main_model_path = app_root / "artifacts" / "best_pr_auc_balancedrf.joblib"
    if not main_model_path.exists():
        st.error(f"모델 파일을 찾을 수 없습니다: {main_model_path}")
        st.stop()
        
    artifact = joblib.load(main_model_path)
    pipeline = artifact["base_pipeline"] if isinstance(artifact, dict) else artifact
    
    return pipeline, df

# 전역 변수로 초기화
model = None
feature_names_kor = []  # 한글 피처 이름 리스트
preprocessor = None

try:
    pipeline, df = load_resources()
    preprocessor = pipeline.named_steps['preprocess']
    model = pipeline.named_steps['model']

    # 1. 원본 피처 이름 추출 (영어)
    raw_feature_names = preprocessor.get_feature_names_out()
    # 'num__', 'cat__' 등의 접두사 제거
    feature_names_en = [name.split('__')[-1] for name in raw_feature_names]
    
    # 2. 한글 매핑 적용 (One-Hot Encoding 처리 포함)
    for name in feature_names_en:
        # 1차 시도: 딕셔너리에 정확히 일치하는 키가 있는지 확인 (수치형 변수 등)
        if name in col_mapping:
            feature_names_kor.append(col_mapping[name])
        else:
            # 2차 시도: One-Hot Encoding된 변수 처리 (예: Month_Feb -> 월_Feb)
            mapped_name = name
            for en_key, ko_val in col_mapping.items():
                # 변수명이 매핑 키로 시작하면 (예: Month로 시작하면)
                if name.startswith(en_key):
                    mapped_name = name.replace(en_key, ko_val)
                    break
            feature_names_kor.append(mapped_name)
    
except Exception as e:
    st.error(f"초기화 중 오류 발생: {e}")
    st.stop()

# --- UI 시작 ---
st.title("🧠 모델 해석 (XAI)")
st.markdown("모델이 전체적으로 어떤 기준을 가지고 구매 여부를 판단하는지 분석합니다.")

tab1, tab2 = st.tabs(["🏆 Global Importance", "📊 Summary Analysis"])

with tab1:
    st.subheader("전역 변수 중요도 (Feature Importance)")
    
    # 모델의 중요도 추출
    importances = model.feature_importances_
    
    # [수정] 한글 이름 리스트(feature_names_kor) 사용
    imp_df = pd.DataFrame({'Feature': feature_names_kor, 'Importance': importances}).sort_values(by='Importance', ascending=False).head(10)
    
    # 1위 피처 이름 추출 (인사이트 문구용)
    top_feature = imp_df.iloc[0]['Feature']
    
    st.success(f"💡 **핵심 요약:** 분석 결과, 구매 결정에 있어 **'{top_feature}'** 데이터가 가장 결정적인 역할을 하고 있습니다.")
    st.markdown("모델이 학습 과정에서 중요하게 참고한 상위 10개 지표입니다.")

    # 그래프 그리기
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 6), facecolor='#0E1117')
    ax.set_facecolor('#0E1117')
    
    sns.barplot(
        data=imp_df, 
        x='Importance', 
        y='Feature', 
        palette='magma_r', 
        ax=ax,
        edgecolor='white',
        linewidth=1.2
    )
    
    ax.set_title("Global Feature Importances", color='white', fontsize=16, pad=20, fontweight='bold')
    ax.tick_params(colors='white', labelsize=11)
    
    # 값 텍스트 표시
    for i, v in enumerate(imp_df['Importance']):
        ax.text(v + 0.002, i, f'{v:.3f}', color='white', va='center', fontweight='bold')
    
    for spine in ax.spines.values():
        spine.set_visible(False)
        
    st.pyplot(fig)
    plt.close(fig)

with tab2:
    st.subheader("SHAP Summary 분석")
    
    st.info("💡 **그래프 해석법:** 점의 색상이 **빨간색(High Value)**일수록, 점이 **오른쪽**에 위치할수록 구매 확률을 높이는 요인입니다.")
    st.write("각 피처의 수치 변화가 실제 구매 예측값에 미치는 영향력을 상세 분석합니다.")
    
    # SHAP 분석용 데이터 준비
    X_sample = preprocessor.transform(df.drop(columns=['Revenue'], errors='ignore').iloc[:100])
    if hasattr(X_sample, "toarray"): X_sample = X_sample.toarray()
    
    # [수정] 데이터프레임 생성 시 columns에 한글 이름 리스트 적용
    X_df = pd.DataFrame(X_sample, columns=feature_names_kor)
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_df)
    
    # 이진 분류 SHAP 값 처리
    sv = shap_values[1] if isinstance(shap_values, list) else (shap_values[:,:,1] if len(np.shape(shap_values))==3 else shap_values)
    
    # 그래프 그리기
    plt.style.use('dark_background')
    fig_sum = plt.figure(figsize=(10, 6), facecolor='#0E1117')
    
    # feature_names 인자는 X_df의 컬럼명이 이미 한글이므로 자동 적용됨
    shap.summary_plot(sv, X_df, show=False)
    
    # 다크모드 텍스트 보정
    for text in fig_sum.findobj(match=plt.Text):
        t = text.get_text()
        if '−' in t: text.set_text(t.replace('−', '-'))
        text.set_color('white')
        
    for ax in fig_sum.get_axes():
        ax.set_facecolor('#0E1117')
        ax.tick_params(colors='white')

    st.pyplot(fig_sum)
    plt.close(fig_sum)