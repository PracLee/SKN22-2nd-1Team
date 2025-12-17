# scripts (실행 버튼 / 엔트리포인트)

## 한 줄 정의

src의 로직을 **한 번 실행**하는 "실행기" 모음.
CLI로 돌려서 데이터 생성/학습/평가/리포트 저장 같은 작업을 한다.

## src와의 차이(중요)

- src/  : 재사용 가능한 로직(부품/서비스). 여러 곳에서 import 해서 씀.
- scripts/: 그 로직을 호출하는 실행 파일(버튼). `python scripts/*.py`로 딱 실행.

즉, **scripts는 얇아야 하고**, 실제 로직은 src에 있어야 한다.

## 예시 파일(권장)

- train.py        : 학습 → artifacts 저장
- eval.py         : 평가 → metrics/리포트 저장
- build_dataset.py: data/processed 생성
- export_report.py: 대시보드용 집계 테이블 저장(선택)

## 규칙

- scripts에는 모델/전처리 구현을 길게 쓰지 않는다.
- scripts는 "파라미터 읽기 → src 호출 → 결과 저장/로그" 정도만 한다.
- 같은 기능을 Streamlit에서도 쓰려면, 로직은 반드시 src로 올린다.
