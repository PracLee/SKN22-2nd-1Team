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

import streamlit as st
# ... (상단 import 동일)

# --- 데이터 및 모델 로드 (경로 로직 수정) ---
@st.cache_resource
def load_resources():
    # 현재 파일 위치: app/pages/06_xai.py
    # app_root: app/
    app_root = Path(__file__).parent.parent 
    artifact_dir = app_root / "artifacts"
    
    # 데이터가 app 폴더 밖에(루트에) 있다면 .parent.parent / "data"
    # 데이터가 app 폴더 안에 있다면 app_root / "data"
    # 현재 구조에 맞춰 선택 (일단 최상단 루트에 있다고 가정)
    project_root = app_root.parent
    data_path = project_root / "data" / "processed" / "test.csv"
    
    # 1. 데이터 로드 확인
    if not data_path.exists():
        st.error(f"데이터 파일을 찾을 수 없습니다: {data_path}")
        st.stop()
    df = pd.read_csv(data_path)
    
    # 2. 모델 파일 로드 확인
    main_model_path = artifact_dir / "best_pr_auc_balancedrf.joblib"
    if not main_model_path.exists():
        st.error(f"모델 파일을 찾을 수 없습니다: {main_model_path}")
        st.stop()
        
    artifact = joblib.load(main_model_path)
    pipeline = artifact["pipeline"] if isinstance(artifact, dict) else artifact
    
    return pipeline, df

# 전역 변수로 초기화 (에러 방지)
model = None
feature_names = []

try:
    pipeline, df = load_resources()
    preprocessor = pipeline.named_steps['preprocess']
    model = pipeline.named_steps['model']

    # 피처 이름 정제
    raw_feature_names = preprocessor.get_feature_names_out()
    feature_names = [name.split('__')[-1] for name in raw_feature_names]
    
except Exception as e:
    st.error(f"초기화 중 오류 발생: {e}")
    st.stop() # 여기서 멈춰야 아래 코드에서 model 관련 에러가 안 납니다.

st.title("🧠 AI Model Explainability (XAI)")
st.markdown("모델이 어떤 기준으로 구매 여부를 판단하는지 분석합니다.")

tab1, tab2 = st.tabs(["Global Importance", "Summary Analysis"])

with tab1:
    st.subheader("🏆 피처 중요도 (Feature Importance)")
    
    # --- [수정] 인사이트를 그래프 위로 이동 ---
    importances = model.feature_importances_
    imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances}).sort_values(by='Importance', ascending=False).head(10)
    
    st.success(f"💡 **핵심 요약:** 분석 결과, 구매 결정에 있어 **'{imp_df.iloc[0]['Feature']}'** 데이터가 가장 결정적인 역할을 하고 있습니다.")
    st.markdown("모델이 학습 과정에서 중요하게 참고한 상위 10개 지표입니다.")

    # --- 그래프 그리기 시작 ---
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
    
    # 축 및 텍스트 설정
    ax.set_title("Global Feature Importances", color='white', fontsize=16, pad=20, fontweight='bold')
    ax.tick_params(colors='white', labelsize=11)
    for i, v in enumerate(imp_df['Importance']):
        ax.text(v + 0.002, i, f'{v:.3f}', color='white', va='center', fontweight='bold')
    
    for spine in ax.spines.values():
        spine.set_visible(False)
        
    st.pyplot(fig)
    plt.close(fig)

with tab2:
    st.subheader("📊 SHAP Summary 분석")
    
    # --- [수정] 해석법을 그래프 위로 이동 ---
    st.info("💡 **그래프 해석법:** 점의 색상이 **빨간색(High Value)**일수록, 점이 **오른쪽**에 위치할수록 구매 확률을 높이는 요인입니다.")
    st.write("각 피처의 수치 변화가 실제 구매 예측값에 미치는 영향력을 상세 분석합니다.")
    
    # SHAP 분석용 데이터 준비
    X_sample = preprocessor.transform(df.drop(columns=['Revenue'], errors='ignore').iloc[:100])
    if hasattr(X_sample, "toarray"): X_sample = X_sample.toarray()
    X_df = pd.DataFrame(X_sample, columns=feature_names)
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_df)
    sv = shap_values[1] if isinstance(shap_values, list) else (shap_values[:,:,1] if len(np.shape(shap_values))==3 else shap_values)
    
    # --- SHAP 그래프 그리기 ---
    plt.style.use('dark_background')
    fig_sum = plt.figure(figsize=(10, 6), facecolor='#0E1117')
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