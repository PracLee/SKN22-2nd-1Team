# import streamlit as st
# import pandas as pd
# import sys
# import os
# import plotly.graph_objects as go  # 시각화를 위한 추가
# from ui.header import render_header

# # =========================================================
# # [STEP 0] Streamlit 기본 설정
# # - 페이지 단독 실행 시에도 공통 레이아웃 유지
# # =========================================================
# st.set_page_config(
#     page_title="Churn Risk Analysis",
#     page_icon="🚨",
#     layout="wide")

# # --- [STEP 1] 경로 설정 ---
# current_dir = os.path.dirname(os.path.abspath(__file__))
# app_dir = os.path.abspath(os.path.join(current_dir, ".."))
# project_root = os.path.abspath(os.path.join(app_dir, ".."))

# if project_root not in sys.path:
#     sys.path.insert(0, project_root)
# if app_dir not in sys.path:
#     sys.path.insert(0, app_dir)

# # --- [STEP 2] 모듈 임포트 ---
# from service.CustomerCareCenter import PurchaseIntentService
# from adapters.model_loader import JoblibArtifactLoader
# from adapters.purchase_intent_pr_auc_adapter import PurchaseIntentPRAUCModelAdapter

# # --- [STEP 3] 데이터 및 서비스 로드 ---
# @st.cache_resource
# def init_service():
#     model_path = "artifacts/best_pr_auc_balancedrf.joblib"
#     adapter = PurchaseIntentPRAUCModelAdapter(model_path)
#     return PurchaseIntentService(adapter), adapter

# @st.cache_data
# def load_data():
#     return pd.read_csv("data/processed/test.csv")

# service, adapter = init_service()
# df = load_data()

# # =========================================================
# # [추가] 개발 중 코드/메시지 변경이 반영 안 될 때를 대비한 캐시 초기화 버튼
# # - st.cache_resource 때문에 "서비스 객체"가 오래 살아남아
# #   수정한 recommend_action이 즉시 반영되지 않는 경우가 있음
# # =========================================================
# with st.sidebar:
#     if st.button("캐시 초기화(개발용)"):
#         st.cache_data.clear()
#         st.cache_resource.clear()
#         st.rerun()
# # =========================================================

# # =========================================================
# # [기존 유지] 상태명 매핑
# # =========================================================
# RISK_NAME_MAP = {
#     "HIGH_RISK": "고위험 이탈군",
#     "OPPORTUNITY": "전환 기회군",
#     "LIKELY_BUYER": "구매 유력군",
# }

# # =========================================================
# # [기존 유지] 전체 데이터에 대해 구매확률/위험등급 계산
# # =========================================================
# @st.cache_data
# def compute_scores(df_all: pd.DataFrame) -> pd.DataFrame:
#     X_all = df_all.drop(columns=["Revenue"], errors="ignore")
#     proba_series = adapter.predict_proba(X_all)

#     if hasattr(proba_series, "values"):
#         proba_values = proba_series.values
#         idx_values = df_all.index
#         proba_s = pd.Series(proba_values, index=idx_values, name="purchase_proba")
#     else:
#         proba_s = pd.Series(proba_series, index=df_all.index, name="purchase_proba")

#     risk_codes = proba_s.apply(lambda p: service.classify_risk(float(p)))

#     score_df = pd.DataFrame({
#         "purchase_proba": proba_s,
#         "risk_code": risk_codes
#     }, index=df_all.index)

#     return score_df

# # =========================================================
# # [기존 유지] "고위험 5 / 기회 3 / 유력 2"로 10개 세션 선정
# # =========================================================
# def select_10_sessions(score_df: pd.DataFrame) -> list[int]:
#     high_needed, opp_needed, likely_needed = 5, 3, 2

#     high_df = score_df[score_df["risk_code"] == "HIGH_RISK"].sort_values("purchase_proba", ascending=True)
#     opp_df = score_df[score_df["risk_code"] == "OPPORTUNITY"].sort_values("purchase_proba", ascending=False)
#     likely_df = score_df[score_df["risk_code"] == "LIKELY_BUYER"].sort_values("purchase_proba", ascending=False)

#     selected_idx = []
#     selected_idx += list(high_df.head(high_needed).index)
#     selected_idx += list(opp_df.head(opp_needed).index)
#     selected_idx += list(likely_df.head(likely_needed).index)

#     if len(selected_idx) < 10:
#         remaining = score_df.drop(index=selected_idx, errors="ignore").sort_values("purchase_proba", ascending=False)
#         need = 10 - len(selected_idx)
#         selected_idx += list(remaining.head(need).index)

#     return selected_idx[:10]

# # =========================================================
# # [기존 유지] 드롭다운 라벨 생성
# # [수정] label -> group_id(1~10) 매핑을 추가로 만든다
# # =========================================================
# score_df = compute_scores(df)
# selected_idx_list = select_10_sessions(score_df)

# df_selected = df.loc[selected_idx_list].copy()
# score_selected = score_df.loc[selected_idx_list]

# group_label_map = {}  # label -> real_idx
# group_id_map = {}     # [수정/추가] label -> group_id(1~10)

# for i, real_idx in enumerate(df_selected.index, start=1):
#     risk_code = score_selected.loc[real_idx, "risk_code"]
#     risk_name = RISK_NAME_MAP.get(risk_code, "관찰 필요")
#     label = f"그룹{i}({risk_name})"

#     group_label_map[label] = real_idx
#     group_id_map[label] = i  # [수정/추가] 핵심: UI 그룹번호를 저장

# # --- [STEP 4] UI 레이아웃 설정 ---
# render_header()
# st.title("🛡️ 고객 이탈 방지 및 마케팅 전략 가이드")

# left_col, right_col = st.columns([4, 6])

# with left_col:
#     st.subheader("📝 세션 정보 입력")

#     selected_label = st.selectbox(
#         "분석할 고객 세션 선택",
#         options=list(group_label_map.keys()),
#         key="session_select_group"
#     )

#     idx = group_label_map[selected_label]
#     selected_group_id = group_id_map[selected_label]  # [수정/추가] 선택된 그룹번호(1~10)

#     row = df.loc[idx]

#     st.write("---")
#     st.write("**📍 주요 행동 지표**")
#     st.write(f"- 페이지 가치: `{row.get('PageValues', 0):.2f}`")
#     st.write(f"- 이탈률: `{row.get('BounceRates', 0)*100:.1f}%`")
#     st.write(f"- 체류 시간: `{row.get('ProductRelated_Duration', 0):.0f}초`")

#     X_one = pd.DataFrame([row.drop("Revenue", errors="ignore")])
#     proba = float(adapter.predict_proba(X_one).iloc[0])
#     risk = service.classify_risk(proba)

#     # =========================================================
#     # [수정] group_id(1~10)를 recommend_action에 전달
#     # - 이제 그룹 선택에 따라 10종 메시지가 1:1로 바뀜
#     # =========================================================
#     action = service.recommend_action(row.to_dict(), proba, group_id=selected_group_id)

# with right_col:
#     st.subheader("📊 분석 결과 및 시각화")

#     fig = go.Figure(go.Indicator(
#         mode="gauge+number",
#         value=proba * 100,
#         domain={'x': [0, 1], 'y': [0, 1]},
#         title={'text': "구매 전환 확률 (%)", 'font': {'size': 20}},
#         gauge={
#             'axis': {'range': [None, 100], 'tickwidth': 1},
#             'bar': {'color': "#1f77b4"},
#             'steps': [
#                 {'range': [0, 20], 'color': "#ff4b4b"},
#                 {'range': [20, 60], 'color': "#ffa500"},
#                 {'range': [60, 100], 'color': "#28a745"}
#             ],
#             'threshold': {
#                 'line': {'color': "white", 'width': 4},
#                 'thickness': 0.75,
#                 'value': proba * 100
#             }
#         }
#     ))
#     fig.update_layout(height=350, margin=dict(l=20, r=20, t=50, b=20))
#     st.plotly_chart(fig, use_container_width=True)

#     if risk == "HIGH_RISK":
#         st.error(f"🚨 **상태: 고위험 이탈군** (확률: {proba*100:.1f}%)")
#     elif risk == "OPPORTUNITY":
#         st.warning(f"⚠️ **상태: 전환 기회군** (확률: {proba*100:.1f}%)")
#     else:
#         st.success(f"✅ **상태: 구매 유력군** (확률: {proba*100:.1f}%)")

#     st.info(f"💡 **추천 마케팅 액션:**\n\n{action}")

# with st.expander("ℹ️ 분석 기준 및 타겟팅 로직 안내"):
#     st.markdown("""
#     **분석 대상 선정 기준(총 10개 세션):**
#     * **전체 예측 기반 샘플링**: 테스트 데이터(`test.csv`) 전체 세션에 대해 모델이 **구매 전환 확률(purchase_proba)** 을 계산합니다.
#     * **위험 등급 분류**:
#       - **고위험 이탈군(HIGH_RISK)**: `p < 0.20`
#       - **전환 기회군(OPPORTUNITY)**: `0.20 ≤ p < 0.60`
#       - **구매 유력군(LIKELY_BUYER)**: `p ≥ 0.60`
#     * **그룹 구성 비율(데모용)**: **고위험 5 / 기회 3 / 유력 2**로 총 10개를 제공합니다.
#     * **대표 세션 선정 방식**
#       - 고위험 5개: HIGH_RISK 중 **구매확률이 가장 낮은 5개**
#       - 기회 3개: OPPORTUNITY 중 **구매확률이 가장 높은 3개**
#       - 유력 2개: LIKELY_BUYER 중 **구매확률이 가장 높은 2개**
#     * **중요**: 드롭다운의 “그룹1~10”은 위 규칙으로 뽑힌 **표본의 순번**이며,
#       이 순번(1~10)을 서비스로 전달하여 **10종 메시지를 1:1로 출력**합니다.
#     """)

# with st.expander("🔍 상세 세션 데이터 보기"):
#     st.dataframe(pd.DataFrame([row]))

# st.info("💡 데모 효율을 위해, 모델 예측 기반으로 10개 세션을 (고위험 5 / 기회 3 / 유력 2)로 구성해 제공합니다.")



import streamlit as st
import pandas as pd
import sys
import os
import plotly.graph_objects as go  # 시각화를 위한 추가
from ui.header import render_header

# =========================================================
# [STEP 0] Streamlit 기본 설정
# - 페이지 단독 실행 시에도 공통 레이아웃 유지
# =========================================================
st.set_page_config(
    page_title="Churn Risk Analysis",
    page_icon="🚨",
    layout="wide"
)

# =========================================================
# [STEP 1] 경로 설정
# =========================================================
current_dir = os.path.dirname(os.path.abspath(__file__))
app_dir = os.path.abspath(os.path.join(current_dir, ".."))
project_root = os.path.abspath(os.path.join(app_dir, ".."))

if project_root not in sys.path:
    sys.path.insert(0, project_root)
if app_dir not in sys.path:
    sys.path.insert(0, app_dir)

# =========================================================
# [STEP 2] 모듈 임포트
# =========================================================
from service.CustomerCareCenter import PurchaseIntentService
from adapters.model_loader import JoblibArtifactLoader
from adapters.purchase_intent_pr_auc_adapter import PurchaseIntentPRAUCModelAdapter

# =========================================================
# [STEP 3] 데이터 및 서비스 로드
# - ✅ PurchaseIntentService가 artifact_path를 요구하므로 반드시 전달
# - ✅ artifact도 함께 로드해서 모델 입력 스키마(feature_names_in_) 추출
# =========================================================
@st.cache_resource
def init_service():
    model_path = os.path.join(app_dir, "artifacts", "best_pr_auc_balancedrf.joblib")

    if not os.path.exists(model_path):
        st.error(f"❌ 모델 파일이 존재하지 않습니다: {model_path}")
        st.stop()

    # Adapter는 경로 기반으로 로드 (팀 어댑터 구현 기준)
    adapter = PurchaseIntentPRAUCModelAdapter(model_path)

    # Service는 artifact_path 필요 (지금 너희 CustomerCareCenter 최종 구조 기준)
    service = PurchaseIntentService(adapter=adapter, artifact_path=model_path)

    # artifact도 로드해서 pipeline/meta 확인 가능하게 보관
    loader = JoblibArtifactLoader(model_path)
    artifact = loader.load()

    return service, adapter, artifact


@st.cache_data
def load_data():
    data_path = os.path.join(project_root, "data", "processed", "test.csv")
    if not os.path.exists(data_path):
        st.error(f"❌ 데이터 파일이 존재하지 않습니다: {data_path}")
        st.stop()
    return pd.read_csv(data_path)


service, adapter, artifact = init_service()
df = load_data()

# =========================================================
# [추가] 모델 기준 feature 스키마 추출 + 입력 정렬
# - ✅ "입력 DataFrame을 모델기준으로 맞춰줘" 요구사항 반영
# - UI/문구/그래프는 그대로 유지하고 내부 입력만 정렬
# =========================================================
@st.cache_data
def get_expected_columns(df_sample: pd.DataFrame) -> list[str]:
    # 1) adapter.pipeline.feature_names_in_ 우선
    if hasattr(adapter, "pipeline") and hasattr(adapter.pipeline, "feature_names_in_"):
        return list(adapter.pipeline.feature_names_in_)

    # 2) artifact.pipeline.feature_names_in_
    if hasattr(artifact, "pipeline") and hasattr(artifact.pipeline, "feature_names_in_"):
        return list(artifact.pipeline.feature_names_in_)

    # 3) artifact.meta에 feature 리스트가 저장된 경우
    if hasattr(artifact, "meta") and isinstance(artifact.meta, dict):
        for k in ["feature_cols", "feature_columns", "columns", "X_columns"]:
            if k in artifact.meta:
                return list(artifact.meta[k])

    # 4) 최후 fallback: df에서 Revenue 제외
    return [c for c in df_sample.columns if c != "Revenue"]


EXPECTED_COLS = get_expected_columns(df)


def align_to_model_schema(row_or_df):
    """
    row(Series) 또는 df(DataFrame)를 모델이 기대하는 feature 컬럼(EXPECTED_COLS)에 맞춘다.
    - 없는 컬럼은 0으로 채움
    - 추가 컬럼은 제거
    - 컬럼 순서도 모델 기준으로 정렬
    """
    if isinstance(row_or_df, pd.Series):
        X = pd.DataFrame([row_or_df.drop("Revenue", errors="ignore")])
    else:
        X = row_or_df.drop(columns=["Revenue"], errors="ignore").copy()

    for col in EXPECTED_COLS:
        if col not in X.columns:
            X[col] = 0

    return X[EXPECTED_COLS]

# =========================================================
# [추가] 개발 중 코드/메시지 변경이 반영 안 될 때를 대비한 캐시 초기화 버튼
# =========================================================
with st.sidebar:
    if st.button("캐시 초기화(개발용)"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()

# =========================================================
# [기존 유지] 상태명 매핑
# =========================================================
RISK_NAME_MAP = {
    "HIGH_RISK": "고위험 이탈군",
    "OPPORTUNITY": "전환 기회군",
    "LIKELY_BUYER": "구매 유력군",
}

# =========================================================
# [기존 유지] 전체 데이터에 대해 구매확률/위험등급 계산
# - ✅ 입력은 align_to_model_schema로 모델 기준 정렬
# =========================================================
@st.cache_data
def compute_scores(df_all: pd.DataFrame) -> pd.DataFrame:
    X_all = align_to_model_schema(df_all)
    proba_series = adapter.predict_proba(X_all)

    if hasattr(proba_series, "values"):
        proba_values = proba_series.values
        idx_values = df_all.index
        proba_s = pd.Series(proba_values, index=idx_values, name="purchase_proba")
    else:
        proba_s = pd.Series(proba_series, index=df_all.index, name="purchase_proba")

    # ✅ service.classify_risk가 CustomerCareCenter에 추가된 상태여야 함
    risk_codes = proba_s.apply(lambda p: service.classify_risk(float(p)))

    score_df = pd.DataFrame({
        "purchase_proba": proba_s,
        "risk_code": risk_codes
    }, index=df_all.index)

    return score_df

# =========================================================
# [기존 유지] "고위험 5 / 기회 3 / 유력 2"로 10개 세션 선정
# =========================================================
def select_10_sessions(score_df: pd.DataFrame) -> list[int]:
    high_needed, opp_needed, likely_needed = 5, 3, 2

    high_df = score_df[score_df["risk_code"] == "HIGH_RISK"].sort_values("purchase_proba", ascending=True)
    opp_df = score_df[score_df["risk_code"] == "OPPORTUNITY"].sort_values("purchase_proba", ascending=False)
    likely_df = score_df[score_df["risk_code"] == "LIKELY_BUYER"].sort_values("purchase_proba", ascending=False)

    selected_idx = []
    selected_idx += list(high_df.head(high_needed).index)
    selected_idx += list(opp_df.head(opp_needed).index)
    selected_idx += list(likely_df.head(likely_needed).index)

    if len(selected_idx) < 10:
        remaining = score_df.drop(index=selected_idx, errors="ignore").sort_values("purchase_proba", ascending=False)
        need = 10 - len(selected_idx)
        selected_idx += list(remaining.head(need).index)

    return selected_idx[:10]

# =========================================================
# [기존 유지] 드롭다운 라벨 생성
# [유지] label -> group_id(1~10) 매핑
# =========================================================
score_df = compute_scores(df)
selected_idx_list = select_10_sessions(score_df)

df_selected = df.loc[selected_idx_list].copy()
score_selected = score_df.loc[selected_idx_list]

group_label_map = {}  # label -> real_idx
group_id_map = {}     # label -> group_id(1~10)

for i, real_idx in enumerate(df_selected.index, start=1):
    risk_code = score_selected.loc[real_idx, "risk_code"]
    risk_name = RISK_NAME_MAP.get(risk_code, "관찰 필요")
    label = f"그룹{i}({risk_name})"

    group_label_map[label] = real_idx
    group_id_map[label] = i  # 핵심: UI 그룹번호를 저장

# =========================================================
# [STEP 4] UI 레이아웃 설정
# =========================================================
render_header()
st.title("🛡️ 고객 이탈 방지 및 마케팅 전략 가이드")

left_col, right_col = st.columns([4, 6])

with left_col:
    st.subheader("📝 세션 정보 입력")

    selected_label = st.selectbox(
        "분석할 고객 세션 선택",
        options=list(group_label_map.keys()),
        key="session_select_group"
    )

    idx = group_label_map[selected_label]
    selected_group_id = group_id_map[selected_label]  # 선택된 그룹번호(1~10)

    row = df.loc[idx]

    st.write("---")
    st.write("**📍 주요 행동 지표**")
    st.write(f"- 페이지 가치: `{row.get('PageValues', 0):.2f}`")
    st.write(f"- 이탈률: `{row.get('BounceRates', 0)*100:.1f}%`")
    st.write(f"- 체류 시간: `{row.get('ProductRelated_Duration', 0):.0f}초`")

    # ✅ 단일 row도 모델 기준 컬럼 정렬
    X_one = align_to_model_schema(row)
    proba = float(adapter.predict_proba(X_one).iloc[0])
    risk = service.classify_risk(proba)

    # group_id(1~10)를 recommend_action에 전달
    action = service.recommend_action(row.to_dict(), proba, group_id=selected_group_id)

with right_col:
    st.subheader("📊 분석 결과 및 시각화")

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=proba * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "구매 전환 확률 (%)", 'font': {'size': 20}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1},
            'bar': {'color': "#1f77b4"},
            'steps': [
                {'range': [0, 20], 'color': "#ff4b4b"},
                {'range': [20, 60], 'color': "#ffa500"},
                {'range': [60, 100], 'color': "#28a745"}
            ],
            'threshold': {
                'line': {'color': "white", 'width': 4},
                'thickness': 0.75,
                'value': proba * 100
            }
        }
    ))
    fig.update_layout(height=350, margin=dict(l=20, r=20, t=50, b=20))
    st.plotly_chart(fig, use_container_width=True)

    if risk == "HIGH_RISK":
        st.error(f"🚨 **상태: 고위험 이탈군** (확률: {proba*100:.1f}%)")
    elif risk == "OPPORTUNITY":
        st.warning(f"⚠️ **상태: 전환 기회군** (확률: {proba*100:.1f}%)")
    else:
        st.success(f"✅ **상태: 구매 유력군** (확률: {proba*100:.1f}%)")

    st.info(f"💡 **추천 마케팅 액션:**\n\n{action}")

# =========================================================
# [유지] 분석 기준 및 타겟팅 로직 안내 (Expander)
# =========================================================
with st.expander("ℹ️ 분석 기준 및 타겟팅 로직 안내"):
    st.markdown("""
    **분석 대상 선정 기준(총 10개 세션):**
    * **전체 예측 기반 샘플링**: 테스트 데이터(`test.csv`) 전체 세션에 대해 모델이 **구매 전환 확률(purchase_proba)** 을 계산합니다.
    * **위험 등급 분류**:
      - **고위험 이탈군(HIGH_RISK)**: `p < 0.20`
      - **전환 기회군(OPPORTUNITY)**: `0.20 ≤ p < 0.60`
      - **구매 유력군(LIKELY_BUYER)**: `p ≥ 0.60`
    * **그룹 구성 비율(데모용)**: **고위험 5 / 기회 3 / 유력 2**로 총 10개를 제공합니다.
    * **대표 세션 선정 방식**
      - 고위험 5개: HIGH_RISK 중 **구매확률이 가장 낮은 5개**
      - 기회 3개: OPPORTUNITY 중 **구매확률이 가장 높은 3개**
      - 유력 2개: LIKELY_BUYER 중 **구매확률이 가장 높은 2개**
    * **중요**: 드롭다운의 “그룹1~10”은 위 규칙으로 뽑힌 **표본의 순번**이며,
      이 순번(1~10)을 서비스로 전달하여 **10종 메시지를 1:1로 출력**합니다.
    """)

# =========================================================
# [유지] 상세 세션 데이터 보기 (Expander)
# =========================================================
with st.expander("🔍 상세 세션 데이터 보기"):
    st.dataframe(pd.DataFrame([row]))

# =========================================================
# [유지] 마지막 안내 문구
# =========================================================
st.info("💡 데모 효율을 위해, 모델 예측 기반으로 10개 세션을 (고위험 5 / 기회 3 / 유력 2)로 구성해 제공합니다.")
