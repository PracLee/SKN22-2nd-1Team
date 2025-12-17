# src/core (규칙/개념)

## 역할

프로젝트의 "기준"을 담는 곳.
예: 세션 입력 스키마, 확률 구간 정의, 액션 추천 룰, 임계값 정책 등

## 예시로 들어갈 내용

- dataclass: SessionFeatures, PredictionResult, Scenario
- rule: "고위험 이탈" 기준(확률 구간/조건)
- rule: 액션 추천 매핑(예: BounceRate 높으면 X 제안)

## 규칙

- 가능한 한 순수 파이썬으로 유지
- Streamlit/pandas/sklearn 같은 무거운 의존은 여기서 최소화
- core는 파일 읽기/모델 로딩을 모른다 (그건 adapters)
