# src (재사용 가능한 제품 코드)

## 한 줄 정의

이 프로젝트의 "로직/부품"이 모여있는 곳.
Streamlit(app), scripts, tests 어디에서든 **import 해서 재사용**한다.

## 포함 내용

- core/: 규칙/개념(스키마, 임계값, 액션 룰 등) — 가능하면 순수 파이썬
- services/: 기능 실행 흐름(아이디어 1~10 유스케이스) — UI는 여기만 호출
- adapters/: 데이터/모델/파일/SHAP 등 외부 의존 처리

## 규칙

- src 안의 코드는 **직접 실행(main 실행)**을 목표로 하지 않는다.
- "어디서 실행하든" 동일하게 동작하도록, 입력/출력과 의존성을 깔끔히 유지한다.
- 모델 파일/데이터 파일 경로 같은 환경 의존은 adapters(+config) 쪽으로 모은다.

## 예시(사용하는 쪽)

- app(Streamlit)에서: `from src.services.predict_service import predict_proba`
- scripts에서: `from src.adapters.train import train_and_save`
