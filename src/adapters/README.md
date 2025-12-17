
# src/adapters (외부 의존 처리)

## 역할

"바깥일" 담당:

- 데이터 로딩/저장
- 전처리 파이프라인 적용
- 모델 로딩/예측
- SHAP/Permutation Importance 등 라이브러리 사용

## 예시로 들어갈 내용

- dataset_loader.py     : CSV 로드, train/test split
- preprocess.py         : 인코딩/스케일링/컬럼 정렬(학습 때와 동일하게)
- model_store.py        : joblib 저장/로드
- predictor.py          : predict_proba 래핑(services가 쓰기 쉽게)
- explainers.py         : shap 계산/요약

## 규칙

- sklearn/xgboost/shap/pandas 의존 OK
- 대신 app이나 core로 의존이 새지 않게 여기서 막는다
