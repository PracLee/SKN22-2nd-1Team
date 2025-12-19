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

# 1. 초기 설정
render_header()
# set_page_config는 반드시 최상단에 있어야 함 (render_header 내부에 없다면 유지)
# st.set_page_config(page_title="model_compare", layout="wide") 

# 텐서플로우 체크
try:
    from tensorflow.keras.models import load_model
    DL_AVAILABLE = True
except:
    DL_AVAILABLE = False

def setup_font():
    plt.rcParams['axes.unicode_minus'] = False
    os_name = platform.system()
    if os_name == 'Windows': plt.rc('font', family='Malgun Gothic')
    elif os_name == 'Darwin': plt.rc('font', family='AppleGothic')
    else: plt.rc('font', family='NanumGothic')

setup_font()

# 2. 자원 로드 함수 (경로 수정 및 방어 로직)
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
    
    main_model_file = art_dir / "best_pr_auc_balancedrf.joblib"
    if not main_model_file.exists():
        st.error(f"❌ 모델 파일을 찾을 수 없습니다: {main_model_file}")
        st.stop()
        
    main_art = joblib.load(main_model_file)
    main_pipe = main_art["pipeline"] if isinstance(main_art, dict) else main_art
    
    others = {}
    cat_path = art_dir / "catboost_model.joblib"
    if cat_path.exists():
        others["CatBoost"] = joblib.load(cat_path)
        
    if DL_AVAILABLE:
        dnn_path = art_dir / "dnn_model.h5"
        if dnn_path.exists():
            others["Deep Learning"] = load_model(dnn_path)
        
    return main_pipe, others, df

# 3. 데이터 로딩 실행 및 변수 할당 (이 부분이 에러 해결 핵심!)
try:
    main_pipe, others, df = load_all()
    preprocessor = main_pipe.named_steps['preprocess']
    main_model = main_pipe.named_steps['model']
    
    raw_names = preprocessor.get_feature_names_out()
    feature_names = [name.split('__')[-1] for name in raw_names]
except Exception as e:
    st.error(f"🔥 초기화 중 오류 발생: {e}")
    st.stop()

# --- 여기서부터 UI 시작 ---
st.title("⚔️ Model Comparison & Individual Diagnosis")

# 1. 개별 고객 진단 섹션
st.subheader("🕵️‍♂️ 개별 고객 심층 분석")
row_idx = st.slider("고객 선택 (Index)", 0, 100, 0)

# 예측 비교
if df is not None:
    target_data = df.iloc[[row_idx]].drop(columns=['Revenue'], errors='ignore')
    all_m = {"Balanced RF (Main)": main_pipe}
    all_m.update(others)

    cols = st.columns(len(all_m))
    for i, (name, m) in enumerate(all_m.items()):
        with cols[i]:
            try:
                if "Deep Learning" in name:
                    input_dl = preprocessor.transform(target_data)
                    if hasattr(input_dl, "toarray"): input_dl = input_dl.toarray()
                    prob = float(m.predict(input_dl, verbose=0)[0][0])
                else:
                    prob = m.predict_proba(target_data)[0, 1]
                
                st.metric(name, f"{prob:.1%}")
                
                if prob >= 0.5:
                    st.success("🎯 구매 (Buy)")
                else:
                    st.error("📉 이탈 (No Buy)")
                    
            except Exception as e:
                st.error("예측 불가")

    # Waterfall Plot
    st.write("#### 💡 해당 고객의 구매 판단 근거")
    
    # 다크모드 그래프 설정
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(10, 6), facecolor='#0E1117')

    X_trans = preprocessor.transform(df.drop(columns=['Revenue'], errors='ignore').iloc[:100])
    if hasattr(X_trans, "toarray"): X_trans = X_trans.toarray()
    X_df = pd.DataFrame(X_trans, columns=feature_names)

    explainer = shap.Explainer(main_model, X_df)
    shap_obj = explainer(X_df)

    if len(shap_obj.shape) == 3:
        shap.plots.waterfall(shap_obj[row_idx, :, 1], show=False)
    else:
        shap.plots.waterfall(shap_obj[row_idx], show=False)

    # 마이너스 깨짐 및 텍스트 색상 보정
    for text in fig.findobj(match=plt.Text):
        t = text.get_text()
        if '−' in t: text.set_text(t.replace('−', '-'))
        text.set_color('white')

    for ax in fig.get_axes():
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