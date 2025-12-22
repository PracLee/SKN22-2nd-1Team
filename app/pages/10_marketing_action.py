# # import streamlit as st
# # import pandas as pd
# # import sys
# # import requests
# # from io import BytesIO
# # from pathlib import Path
# from ui.header import render_header

# # =========================================================
# # [STEP 1] 프로젝트 루트 경로 설정 (Path 기반, OS 독립)
# # =========================================================
# CURRENT_FILE = Path(__file__).resolve()
# APP_DIR = CURRENT_FILE.parents[1]
# PROJECT_ROOT = CURRENT_FILE.parents[2]

# if str(PROJECT_ROOT) not in sys.path:
#     sys.path.insert(0, str(PROJECT_ROOT))
# if str(APP_DIR) not in sys.path:
#     sys.path.insert(0, str(APP_DIR))

# # =========================================================
# # [STEP 2] 모듈 임포트
# # =========================================================
# from adapters.model_loader import JoblibArtifactLoader
# from adapters.purchase_intent_pr_auc_adapter import PurchaseIntentPRAUCModelAdapter
# from service.CustomerCareCenter import PurchaseIntentService

# # =========================================================
# # [STEP 3] 모델 로딩
# # - 🔧 수정 핵심:
# #   PurchaseIntentPRAUCModelAdapter는 "ModelArtifact"가 아니라
# #   "artifact_path(경로)"를 받도록 구현되어 있으므로,
# #   adapter에는 경로를 전달해야 함.
# # =========================================================
# @st.cache_resource
# def init_service():
#     artifact_path = PROJECT_ROOT / "artifacts" / "best_pr_auc_balancedrf.joblib"

#     # 🔧 수정: 파일 존재 여부 체크 (다른 PC에서도 원인 파악 쉬움)
#     if not artifact_path.exists():
#         st.error(f"❌ 모델 파일이 존재하지 않습니다: {artifact_path}")
#         st.stop()

#     # 🔧 수정: adapter는 "artifact 객체"가 아니라 "경로"를 받는다
#     adapter = PurchaseIntentPRAUCModelAdapter(artifact_path)

#     # service 생성
#     service = PurchaseIntentService(adapter)

#     # (선택) artifact를 꼭 반환해야 한다면 loader로 따로 로드해서 반환 가능
#     # 단, adapter가 내부에서도 로드할 수 있어 "중복 로딩"이 될 수 있음.
#     loader = JoblibArtifactLoader(artifact_path)
#     artifact = loader.load()

#     return service, adapter, artifact


# service, adapter, artifact = init_service()

# # =========================================================
# # [STEP 4] 데이터 로드 (UI용 샘플)
# # =========================================================
# @st.cache_data
# def load_data():
#     data_path = PROJECT_ROOT / "data" / "processed" / "test.csv"
#     if not data_path.exists():
#         st.error(f"❌ 데이터 파일이 존재하지 않습니다: {data_path}")
#         st.stop()
#     return pd.read_csv(data_path)

# # 기존 유지: 30개 세션 사용
# df = load_data().head(30)

# # =========================================================
# # [유지] 그룹별 Google Drive 이미지 (10개)
# # =========================================================
# GROUP_IMAGE_MAP = {
#     1: "https://drive.google.com/uc?id=1C_FatsRQqIQPnygwx3McU4NcZZiRiBFp",
#     2: "https://drive.google.com/uc?id=1KB0J8zrm7ZFC4FvAcL3Z1r831ZBZpKJR",
#     3: "https://drive.google.com/uc?id=1n7l5AhZIU46u7vD6UxBk7xWSaraVMDao",
#     4: "https://drive.google.com/uc?id=1tmKyCJ_qhVv0050H9DPNdckhgXV0QwBt",
#     5: "https://drive.google.com/uc?id=1o3XvxRP9-iN80cO8T_aVPoA04CMk94hh",
#     6: "https://drive.google.com/uc?id=1QUXQMxvR0b7Gyx-KsHidAA6kVg8nFp_Y",
#     7: "https://drive.google.com/uc?id=1yJ5An-fs3J8PlADZ4NYUp4ySiFh4Okct",
#     8: "https://drive.google.com/uc?id=1u7eZMaBMpQ2aqg5A9BRw5BP1eKu6Kw4m",
#     9: "https://drive.google.com/uc?id=1kU9k2cCKkHRNnHhLCYKy4QsIQdTYAyCc",
#     10: "https://drive.google.com/uc?id=1kZpn2fKK2yC1PImdHo2CwQ61DVWf9qSy",
# }

# @st.cache_data
# def load_image_from_drive(url: str):
#     try:
#         response = requests.get(url, timeout=10)
#         if response.status_code == 200:
#             return BytesIO(response.content)
#     except Exception:
#         return None
#     return None

# # =========================================================
# # [추가] 개발 중 캐시 초기화 버튼
# # - Streamlit 캐시 때문에 코드 수정이 반영 안 되는 상황 방지
# # =========================================================
# with st.sidebar:
#     if st.button("캐시 초기화(개발용)"):
#         st.cache_data.clear()
#         st.cache_resource.clear()
#         st.rerun()

# # =========================================================
# # [UI]
# # =========================================================
# render_header()
# st.title("🎯 마케팅 전략 가이드 시뮬레이터")
# st.info("💡 분석 효율을 위해 상위 30개의 주요 타겟 세션을 요약하여 제공합니다.")

# left_col, right_col = st.columns([4, 6])

# with left_col:
#     st.subheader("📥 타겟 세션 리스트 (TOP 30)")

#     labels = [f"세션 분석 대상 #{i+1}" for i in range(len(df))]
#     selected_label = st.selectbox("분석할 세션 선택", labels)

#     group_id = labels.index(selected_label) + 1
#     row = df.iloc[group_id - 1]

#     X_one = pd.DataFrame([row.drop("Revenue", errors="ignore")])

#     # adapter.predict_proba 사용
#     proba = float(adapter.predict_proba(X_one).iloc[0])
#     risk = service.classify_risk(proba)
#     action = service.recommend_action(row.to_dict(), proba, group_id=group_id)

#     st.write("---")
#     st.metric("페이지 가치 (Value)", f"{row.get('PageValues', 0):.2f}")
#     st.metric("이탈률 (Bounce)", f"{row.get('BounceRates', 0)*100:.1f}%")
#     st.metric("체류 시간 (Duration)", f"{row.get('ProductRelated_Duration', 0):.1f}s")

# with right_col:
#     st.subheader("👤 고객 페르소나 진단")

#     # 30개 세션 → 이미지 10개 순환 매핑
#     image_key = ((group_id - 1) % 10) + 1
#     img_bytes = load_image_from_drive(GROUP_IMAGE_MAP.get(image_key))

#     if img_bytes:
#         st.image(img_bytes, width=420)
#     else:
#         st.warning("⚠️ 페르소나 이미지를 불러오지 못했습니다.")

#     st.write(f"**실시간 구매 전환 확률: {proba*100:.1f}%**")
#     st.progress(proba)

#     if risk == "HIGH_RISK":
#         status_text = "🚨 이탈 위험 높음: 즉각적인 케어가 필요합니다."
#     elif risk == "OPPORTUNITY":
#         status_text = "⚠️ 망설이는 단계: 혜택이 전환의 열쇠입니다."
#     else:
#         status_text = "✅ 구매 유력 상태: 흐름을 방해하지 마세요."

#     st.markdown("### 📌 추천 마케팅 전략")
#     st.markdown(f"**{status_text}**")
#     st.markdown(f"> {action}")

# with st.expander("🔍 선택된 세션 상세 로그 확인"):
#     st.table(pd.DataFrame([row]).T)

# GROUP_IMAGE_MAP = {
#     1: "https://drive.google.com/uc?id=1C_FatsRQqIQPnygwx3McU4NcZZiRiBFp",
#     2: "https://drive.google.com/uc?id=1KB0J8zrm7ZFC4FvAcL3Z1r831ZBZpKJR",
#     3: "https://drive.google.com/uc?id=1n7l5AhZIU46u7vD6UxBk7xWSaraVMDao",
#     4: "https://drive.google.com/uc?id=1tmKyCJ_qhVv0050H9DPNdckhgXV0QwBt",
#     5: "https://drive.google.com/uc?id=1o3XvxRP9-iN80cO8T_aVPoA04CMk94hh",
#     6: "https://drive.google.com/uc?id=1QUXQMxvR0b7Gyx-KsHidAA6kVg8nFp_Y",
#     7: "https://drive.google.com/uc?id=1yJ5An-fs3J8PlADZ4NYUp4ySiFh4Okct",
#     8: "https://drive.google.com/uc?id=1u7eZMaBMpQ2aqg5A9BRw5BP1eKu6Kw4m",
#     9: "https://drive.google.com/uc?id=1kU9k2cCKkHRNnHhLCYKy4QsIQdTYAyCc",
#     10:"https://drive.google.com/uc?id=1kZpn2fKK2yC1PImdHo2CwQ61DVWf9qSy",
# }

# 이미지 1 "🚨 [심폐소생술 시급] 고객님이 '뒤로 가기' 버튼과 썸 타는 중입니다! 혜택 한 줄 요약이랑 베스트 리뷰로 멱살 잡고 끌어와야 해요!"
# 이미지 2. "🚪 '나 지금 나간다?'라고 온몸으로 외치는 중! 3초 안에 할인 쿠폰이나 무료배송 안 보여주면 영영 남남입니다. 빨리요!"
# 이미지 3. "🧯 관심이라는 불씨가 생기기도 전에 로그아웃 각! 랜딩 페이지에 인기 상품이랑 신뢰 팍팍 가는 인증마크로 도배해서 눈길을 뺏으세요!"
# 이미지 4.  "🪝 살짝 솔깃해 보이지만, 로딩 1초만 늦어도 떠날 분입니다. 복잡한 거 다 빼고 핵심 혜택만 코앞에 들이미세요!"
# 이미지 5. "⚠️ 이 정도면 '밀당' 고수네요. 살까 말까 고민하는 게 보입니다. '오늘만 이 가격' 콤보 한 방이면 바로 넘어옵니다!"
# 이미지 6. "👀 장바구니에 넣을까 말까 100번 고민 중! '최저가 보장'이나 '빠른 배송' 정보로 고객님의 우유부단함에 마침표를 찍어주세요!"
# 이미지 7. "🎯 대어 낚기 직전입니다! '사람들이 이 제품 칭찬을 이렇게 많이 해요'라고 사회적 증거(후기/별점)를 마구 투척하세요!"
# 이미지 8. "🔥 [결제 직전] 조금만 밀면 카드 슬래시! 한정판 쿠폰이나 '무료배송까지 얼마 안 남았어요'라는 멘트로 불을 지피세요!"
# 이미지 9. "🛒 이미 마음은 결제 완료! 괜히 팝업 띄워서 방해하지 말고, 쿠폰 자동 적용해서 레드카펫 깔아드립시다. 결제 길만 걷게 하세요!"
# 이미지 10.  "✅ [확정 전환] 이분은 숨만 쉬어도 구매하실 분입니다! 추가 영업은 사치일 뿐. 가볍게 '함께 사면 좋은 꿀템' 하나만 슥- 던져보세요."
# 이런 상황에 맞는 사람 얼굴 이미지 10개 각각 1장씩 용량은 작게 제작해줘


import streamlit as st
import pandas as pd
import sys
import requests
from io import BytesIO
from pathlib import Path

from ui.header import render_header

# =========================================================
# [STEP 0] Streamlit 기본 설정
# - (누락되기 쉬운 부분이라 명시적으로 복구)
# =========================================================
st.set_page_config(
    page_title="Marketing Action Guide",
    page_icon="🎯",
    layout="wide"
)

# =========================================================
# [STEP 1] 프로젝트 루트 경로 설정 (Path 기반, OS 독립)
# =========================================================
CURRENT_FILE = Path(__file__).resolve()
APP_DIR = CURRENT_FILE.parents[1]
PROJECT_ROOT = CURRENT_FILE.parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

# =========================================================
# [STEP 2] 모듈 임포트
# =========================================================
from adapters.model_loader import JoblibArtifactLoader
from adapters.purchase_intent_pr_auc_adapter import PurchaseIntentPRAUCModelAdapter
from service.CustomerCareCenter import PurchaseIntentService

# =========================================================
# [STEP 3] 모델 로딩
# - 🔧 수정 핵심:
#   PurchaseIntentPRAUCModelAdapter는 "ModelArtifact"가 아니라
#   "artifact_path(경로)"를 받도록 구현되어 있으므로,
#   adapter에는 경로를 전달해야 함.
# =========================================================
@st.cache_resource
def init_service():
    artifact_path = PROJECT_ROOT / "app" / "artifacts" / "best_pr_auc_balancedrf.joblib"

    # 🔧 수정: 파일 존재 여부 체크 (다른 PC에서도 원인 파악 쉬움)
    if not artifact_path.exists():
        st.error(f"❌ 모델 파일이 존재하지 않습니다: {artifact_path}")
        st.stop()

    # 🔧 수정: adapter는 "artifact 객체"가 아니라 "경로"를 받는다
    # (Path 객체도 PathLike로 동작하지만, 구현체에 따라 str이 안전할 수 있음)
    adapter = PurchaseIntentPRAUCModelAdapter(str(artifact_path))

    # service 생성
    service = PurchaseIntentService(
        adapter=adapter,
    artifact_path=str(artifact_path))

    # (선택) artifact를 꼭 반환해야 한다면 loader로 따로 로드해서 반환 가능
    # 단, adapter가 내부에서도 로드할 수 있어 "중복 로딩"이 될 수 있음.
    loader = JoblibArtifactLoader(str(artifact_path))
    artifact = loader.load()

    return service, adapter, artifact


service, adapter, artifact = init_service()

# =========================================================
# [STEP 4] 데이터 로드 (UI용 샘플)
# =========================================================
@st.cache_data
def load_data():
    data_path = PROJECT_ROOT / "data" / "processed" / "test.csv"
    if not data_path.exists():
        st.error(f"❌ 데이터 파일이 존재하지 않습니다: {data_path}")
        st.stop()
    return pd.read_csv(data_path)

# 기존 유지: 30개 세션 사용
df = load_data().head(30)

# =========================================================
# [핵심 추가] 입력 DataFrame을 모델 기준으로 정렬/보정
# - 모델이 학습한 컬럼이 df에 없으면 ColumnTransformer에서 바로 터짐
# - 따라서 "모델이 기대하는 컬럼 목록"을 추출하고, X_one을 그 스키마에 맞춘다
# =========================================================
@st.cache_data
def get_model_expected_columns() -> list[str]:
    """
    모델 파이프라인이 기대하는 입력 컬럼 목록을 추출한다.
    - sklearn 1.0+ 에서 Pipeline/Estimator에 feature_names_in_가 존재할 수 있음
    - 존재하지 않으면 artifact/meta에 저장된 컬럼 정보를 탐색
    """
    # 1) adapter가 pipeline을 들고 있는 경우 (가장 흔한 케이스)
    pipe = getattr(adapter, "pipeline", None)
    if pipe is not None and hasattr(pipe, "feature_names_in_"):
        return list(pipe.feature_names_in_)

    # 2) artifact.pipeline로 접근 가능한 경우
    art = artifact
    if hasattr(art, "pipeline") and hasattr(art.pipeline, "feature_names_in_"):
        return list(art.pipeline.feature_names_in_)

    # 3) meta에 컬럼 목록을 저장해둔 경우 (팀 규약에 따라 키명이 다를 수 있음)
    if hasattr(art, "meta") and isinstance(art.meta, dict):
        for key in ["feature_cols", "feature_columns", "columns", "X_columns", "input_columns"]:
            if key in art.meta and isinstance(art.meta[key], (list, tuple)) and len(art.meta[key]) > 0:
                return list(art.meta[key])

    # 4) 최후의 수단: 현재 df의 feature 컬럼을 사용 (Revenue 제외)
    #    (이 케이스는 "모델과 df 컬럼이 원래 동일"한 상황에서만 안전)
    fallback = [c for c in df.columns if c != "Revenue"]
    return fallback


EXPECTED_COLS = get_model_expected_columns()


def align_features_to_model_schema(row: pd.Series) -> pd.DataFrame:
    """
    단일 row를 모델이 기대하는 컬럼 스키마(EXPECTED_COLS)에 맞춰 DataFrame(1행)으로 만든다.
    - df에 없는 컬럼은 기본값으로 채움
    - df에 있지만 모델이 기대하지 않는 컬럼은 제거
    - 컬럼 순서를 EXPECTED_COLS로 정렬
    """
    # 타겟/불필요 컬럼 제거 (Revenue는 예측 입력에서 제외)
    feature_row = row.drop("Revenue", errors="ignore")

    # 1행 DataFrame 생성
    X = pd.DataFrame([feature_row.to_dict()])

    # 모델이 기대하는 컬럼만 남기고/추가
    # 없는 컬럼은 0으로 채움 (범주형이어도 0으로 두는 이유: 전처리에서 handle_unknown/빈값 처리 기대)
    # 필요 시 아래 default 값을 "Unknown" 같은 문자열로 바꾸는 것도 가능
    for col in EXPECTED_COLS:
        if col not in X.columns:
            X[col] = 0

    # 모델이 기대하지 않는 컬럼은 제거
    X = X[EXPECTED_COLS]

    return X

# =========================================================
# [유지] 그룹별 Google Drive 이미지 (10개)
# =========================================================
GROUP_IMAGE_MAP = {
    1: "https://drive.google.com/uc?id=1C_FatsRQqIQPnygwx3McU4NcZZiRiBFp",
    2: "https://drive.google.com/uc?id=1KB0J8zrm7ZFC4FvAcL3Z1r831ZBZpKJR",
    3: "https://drive.google.com/uc?id=1n7l5AhZIU46u7vD6UxBk7xWSaraVMDao",
    4: "https://drive.google.com/uc?id=1tmKyCJ_qhVv0050H9DPNdckhgXV0QwBt",
    5: "https://drive.google.com/uc?id=1o3XvxRP9-iN80cO8T_aVPoA04CMk94hh",
    6: "https://drive.google.com/uc?id=1QUXQMxvR0b7Gyx-KsHidAA6kVg8nFp_Y",
    7: "https://drive.google.com/uc?id=1yJ5An-fs3J8PlADZ4NYUp4ySiFh4Okct",
    8: "https://drive.google.com/uc?id=1u7eZMaBMpQ2aqg5A9BRw5BP1eKu6Kw4m",
    9: "https://drive.google.com/uc?id=1kU9k2cCKkHRNnHhLCYKy4QsIQdTYAyCc",
    10: "https://drive.google.com/uc?id=1kZpn2fKK2yC1PImdHo2CwQ61DVWf9qSy",
}

@st.cache_data
def load_image_from_drive(url: str):
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            return BytesIO(response.content)
    except Exception:
        return None
    return None

# =========================================================
# [추가] 개발 중 캐시 초기화 버튼
# - Streamlit 캐시 때문에 코드 수정이 반영 안 되는 상황 방지
# =========================================================
with st.sidebar:
    if st.button("캐시 초기화(개발용)"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()

# =========================================================
# [UI]
# =========================================================
render_header()
st.title("🎯 마케팅 전략 가이드 시뮬레이터")
st.info("💡 분석 효율을 위해 상위 30개의 주요 타겟 세션을 요약하여 제공합니다.")

left_col, right_col = st.columns([4, 6])

with left_col:
    st.subheader("📥 타겟 세션 리스트 (TOP 30)")

    labels = [f"세션 분석 대상 #{i+1}" for i in range(len(df))]
    selected_label = st.selectbox("분석할 세션 선택", labels)

    group_id = labels.index(selected_label) + 1
    row = df.iloc[group_id - 1]

    # =========================================================
    # [핵심 수정] 모델 기준 스키마로 X_one 정렬/보정
    # =========================================================
    X_one = align_features_to_model_schema(row)

    # adapter.predict_proba 사용
    proba = float(adapter.predict_proba(X_one).iloc[0])
    risk = service.classify_risk(proba)
    action = service.recommend_action(row.to_dict(), proba, group_id=group_id)

    st.write("---")
    st.metric("페이지 가치 (Value)", f"{row.get('PageValues', 0):.2f}")
    st.metric("이탈률 (Bounce)", f"{row.get('BounceRates', 0)*100:.1f}%")
    st.metric("체류 시간 (Duration)", f"{row.get('ProductRelated_Duration', 0):.1f}s")

with right_col:
    st.subheader("👤 고객 페르소나 진단")

    # 30개 세션 → 이미지 10개 순환 매핑
    image_key = ((group_id - 1) % 10) + 1
    img_bytes = load_image_from_drive(GROUP_IMAGE_MAP.get(image_key))

    if img_bytes:
        st.image(img_bytes, width=420)
    else:
        st.warning("⚠️ 페르소나 이미지를 불러오지 못했습니다.")

    st.write(f"**실시간 구매 전환 확률: {proba*100:.1f}%**")
    st.progress(proba)

    if risk == "HIGH_RISK":
        status_text = "🚨 이탈 위험 높음: 즉각적인 케어가 필요합니다."
    elif risk == "OPPORTUNITY":
        status_text = "⚠️ 망설이는 단계: 혜택이 전환의 열쇠입니다."
    else:
        status_text = "✅ 구매 유력 상태: 흐름을 방해하지 마세요."

    st.markdown("### 📌 추천 마케팅 전략")
    st.markdown(f"**{status_text}**")
    st.markdown(f"> {action}")

with st.expander("🔍 선택된 세션 상세 로그 확인"):
    st.table(pd.DataFrame([row]).T)
