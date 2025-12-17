
# Online Shoppers Streamlit Project

목표: 세션 정보를 입력하면 `Revenue(구매)` 확률을 예측하고,
What-if / 채널분석 / 설명가능성 / 모델비교 등을 Streamlit로 제공한다.

## 구조(초간단 규칙)

- app/: 화면(UI)만. 계산 로직 금지.
- src/core/: 규칙/개념. 라이브러리 의존 최소(가능하면 순수 파이썬).
- src/services/: "기능 실행 흐름". UI가 여기만 호출.
- src/adapters/: 데이터/모델/파일/SHAP 등 외부 의존 처리.

### 한 줄 요약

UI는 요청만 → services가 흐름 실행 → core는 규칙 제공 → adapters는 IO 담당

## 실행 예시

- `streamlit run app/Home.py`
- `python scripts/train.py`
