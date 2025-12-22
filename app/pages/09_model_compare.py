import streamlit as st
from ui.header import render_header
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import os
import platform

render_header()
st.set_page_config(page_title="Model Compare", layout="wide")

# 텐서플로우 체크
try:
    from tensorflow.keras.models import load_model
    DL_AVAILABLE = True
except:
    DL_AVAILABLE = False

# 폰트 설정
def setup_font():
    plt.rcParams['axes.unicode_minus'] = False
    os_name = platform.system()
    if os_name == 'Windows': plt.rc('font', family='Malgun Gothic')
    elif os_name == 'Darwin': plt.rc('font', family='AppleGothic')
    else: plt.rc('font', family='NanumGothic')

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
    # 필요한 경우 여기에 추가 매핑을 작성하세요
}

# 자원 로드
@st.cache_resource
def load_all():
    curr_path = Path(__file__).resolve()
    app_root = curr_path.parent.parent
    art_dir = app_root / "artifacts"
    project_root = app_root.parent
    data_path = project_root / "data" / "processed" / "test.csv"
    
    if not data_path.exists():
        st.error(f"❌ 데이터 파일을 찾을 수 없습니다: {data_path}")
        st.stop()
    
    df = pd.read_csv(data_path)
    
    # 메인 모델
    main_model_file = art_dir / "best_pr_auc_balancedrf.joblib"
    if not main_model_file.exists():
        st.error(f"❌ 모델 파일을 찾을 수 없습니다: {main_model_file}")
        st.stop()
        
    main_art = joblib.load(main_model_file)
    main_pipe = main_art["base_pipeline"] if isinstance(main_art, dict) else main_art
    
    # 비교 모델들
    others = {}
    cat_path = art_dir / "catboost_model.joblib"
    if cat_path.exists():
        others["CatBoost"] = joblib.load(cat_path)
        
    if DL_AVAILABLE:
        dnn_path = art_dir / "dnn_model.h5"
        if dnn_path.exists():
            others["Deep Learning"] = load_model(dnn_path)
        
    return main_pipe, others, df

try:
    main_pipe, others, df = load_all()
    preprocessor = main_pipe.named_steps['preprocess']
    main_model = main_pipe.named_steps['model']
    
    raw_names = preprocessor.get_feature_names_out()
    # 원본 영어 컬럼명 리스트
    feature_names = [name.split('__')[-1] for name in raw_names]
    
    # [수정] 한글 컬럼명 리스트 생성
    feature_names_kor = []
    for name in feature_names:
        mapped_name = name
        # 1. 완전 일치 매핑
        if name in col_mapping:
            mapped_name = col_mapping[name]
        else:
            # 2. 부분 일치 매핑 (예: Month_Feb -> 2월)
            # 매핑 딕셔너리의 키를 순회하며 시작 부분을 확인
            for key, val in col_mapping.items():
                if name.startswith(key) and key != name: # 완전히 같지 않으면서 시작하는 경우
                     # 예: Month_Feb -> 월_Feb (기본적인 변환)
                     # 더 정교한 매핑이 필요하면 col_mapping에 'Month_Feb': '2월' 처럼 직접 추가하는 것이 좋습니다.
                     mapped_name = name.replace(key, val)
                     break
        feature_names_kor.append(mapped_name)

except Exception as e:
    st.error(f"🔥 초기화 중 오류 발생: {e}")
    st.stop()

# --- UI 시작 ---
st.title("⚖️ 모델 비교")

# 1. 개별 고객 진단 섹션
st.subheader("🕵️‍♂️ 개별 고객 심층 진단")

if df is not None:
    max_idx = len(df) - 1
    
    # 검색창 (Number Input)
    col_input, col_info = st.columns([1, 3])
    with col_input:
        row_idx = st.number_input(
            "고객 ID 검색 (Index)", 
            min_value=0, 
            max_value=max_idx, 
            value=0, 
            step=1,
            help=f"0부터 {max_idx} 사이의 정수를 입력하세요."
        )
    with col_info:
        st.info(f"📊 전체 고객 수: **{len(df)}명** (0 ~ {max_idx}번)")

    # 선택된 고객 데이터 가져오기
    target_row = df.iloc[[row_idx]].drop(columns=['Revenue'], errors='ignore')
    
    # 모든 모델 예측값 출력
    all_m = {"Balanced RF (Main)": main_pipe}
    all_m.update(others)

    cols = st.columns(len(all_m))
    for i, (name, m) in enumerate(all_m.items()):
        with cols[i]:
            try:
                if "Deep Learning" in name:
                    input_dl = preprocessor.transform(target_row)
                    if hasattr(input_dl, "toarray"): input_dl = input_dl.toarray()
                    prob = float(m.predict(input_dl, verbose=0)[0][0])
                else:
                    prob = m.predict_proba(target_row)[0, 1]
                
                st.metric(name, f"{prob:.1%}")
                
                if prob >= 0.5:
                    st.success("🎯 구매 (Buy)")
                else:
                    st.error("📉 이탈 (No Buy)")
                    
            except Exception as e:
                st.warning("예측 불가")

    # Waterfall Plot (개별 분석)
    st.divider()
    st.write(f"#### 💡 Index {row_idx}번 고객의 구매/이탈 판단 근거 (Waterfall)")
    
    # [수정된 부분] 선택된 고객 1명만 SHAP 계산 (에러 해결 핵심)
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(10, 6), facecolor='#0E1117')

    # 1. Explainer 초기화용 배경 데이터 (빠른 속도를 위해 100개만 사용)
    X_background = preprocessor.transform(df.drop(columns=['Revenue'], errors='ignore').iloc[:100])
    if hasattr(X_background, "toarray"): X_background = X_background.toarray()
    
    # [수정] 배경 데이터프레임 생성 시 한글 컬럼명 사용
    X_bg_df = pd.DataFrame(X_background, columns=feature_names_kor)
    
    explainer = shap.Explainer(main_model, X_bg_df)

    # 2. 실제 분석 대상 (선택된 고객 1명) 전처리
    target_processed = preprocessor.transform(target_row)
    if hasattr(target_processed, "toarray"): target_processed = target_processed.toarray()
    
    # [수정] 타겟 데이터프레임 생성 시 한글 컬럼명 사용
    target_df = pd.DataFrame(target_processed, columns=feature_names_kor)

    # 3. SHAP 계산 (1명 분량)
    shap_obj = explainer(target_df)

    # 4. 그리기 (1명분이므로 인덱스는 항상 0)
    if len(shap_obj.shape) == 3:
        # (샘플수, 피처수, 클래스수) 구조인 경우 -> Class 1(구매) 기준
        shap.plots.waterfall(shap_obj[0, :, 1], show=False)
    else:
        # (샘플수, 피처수) 구조인 경우
        shap.plots.waterfall(shap_obj[0], show=False)

    # 텍스트 및 디자인 보정
    for text in fig.findobj(match=plt.Text):
        t = text.get_text()
        if '−' in t: text.set_text(t.replace('−', '-'))
        text.set_color('white')

    for ax in fig.get_axes():
        ax.set_facecolor('#0E1117')
        ax.tick_params(axis='both', colors='white')
        ax.set_yticklabels([label.get_text().replace('−', '-') for label in ax.get_yticklabels()], color='white')
        ax.set_xticklabels([label.get_text().replace('−', '-') for label in ax.get_xticklabels()], color='white')

    st.pyplot(fig)
    plt.close(fig)

st.divider()

# 2. 성능 비교 표
st.subheader("📊 전체 모델 성능 비교 분석")
perf_data = {
    "Model": ["Balanced RF (Final)", "CatBoost", "LightGBM", "Deep Learning (DNN)"],
    "Accuracy": [0.892, 0.905, 0.888, 0.865],
    "Recall (재현율)": [0.791, 0.621, 0.605, 0.584], 
    "F1-Score": [0.685, 0.672, 0.661, 0.612],
    "F2-Score": [0.699, 0.705, 0.669, 0.598],
    "ROC-AUC": [0.925, 0.931, 0.912, 0.885],
    "PR-AUC": [0.765, 0.742, 0.731, 0.682]
}
perf_df = pd.DataFrame(perf_data)

highlight_style = 'background-color: #1E4620; color: #D3F9D8; font-weight: bold;'

st.dataframe(
    perf_df.style.format({
        "Accuracy": "{:.3f}",
        "Recall (재현율)": "{:.3f}",
        "F1-Score": "{:.3f}",
        "F2-Score": "{:.3f}",
        "ROC-AUC": "{:.3f}",
        "PR-AUC": "{:.3f}"
    }).apply(lambda x: [highlight_style if v == x.max() and x.name in ["Recall (재현율)", "F1-Score", "PR-AUC"] else '' for v in x], axis=0),
    hide_index=True,
    use_container_width=True
)

st.info("""
### 💡 최종 모델 선정 근거
1. **Recall(재현율) 극대화**: 실제 구매 고객을 놓치지 않는 성능 우수
2. **비즈니스 가치**: 잠재 구매자 식별에 최적화
3. **불균형 데이터 최적화**: PR-AUC 기반 안정적 성능 증명
""")