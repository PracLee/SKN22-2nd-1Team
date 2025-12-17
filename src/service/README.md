# src/services (기능 실행 흐름)

## 역할

"한 기능이 어떻게 실행되는지"를 정리하는 곳.
UI(app)는 여기 함수/클래스만 호출한다.

## 여기서 하는 일

- 입력 검증(core 스키마 사용)
- 필요한 데이터/모델을 adapters 통해 가져오기
- 예측/집계/비교/설명 생성 흐름을 한 번에 실행
- UI가 쓰기 좋은 결과(dict/DTO)로 반환

## 파일 예시(권장)

- predict_service.py        : 아이디어 1 (구매확률)
- whatif_service.py         : 아이디어 2 (슬라이더 시뮬)
- channel_service.py        : 아이디어 3 (TrafficType/Region/Browser 분석)
- risk_service.py           : 아이디어 4 (고위험 탐지)
- explain_service.py        : 아이디어 6 (importance/shap)
- compare_models_service.py : 아이디어 9 (모델 비교)
- action_service.py         : 아이디어 10 (액션 카드)

## 규칙

- services는 Streamlit을 모르도록 유지(= UI 독립)
- 데이터/모델 접근은 adapters를 통해서만
- core 규칙을 가져다 "조립"하는 역할
