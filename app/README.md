
# app (Streamlit UI)

## 역할

- 사용자 입력 받기, 결과를 시각화하기
- 버튼/슬라이더/탭 구성
- **src/services 함수만 호출**해서 결과를 화면에 보여준다.

## 여기에 두면 좋은 것

- pages/: 아이디어별 탭 (예: 01_predict.py, 02_whatif.py ...)
- components/: 공용 카드/차트/필터 UI
- 간단한 표시용 포맷터

## 하지 말 것(중요)

- 모델 학습/로딩 로직 직접 작성 X
- 전처리/피처 엔지니어링을 UI에서 수행 X
- pandas로 무거운 집계를 UI에서 반복 수행 X (필요하면 services로)
