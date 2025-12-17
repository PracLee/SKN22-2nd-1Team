
### 📌 목적 (Purpose)

원본 데이터로부터 **모델 학습이 가능한 형태로 가공하는 전 과정**을 기록하고 설명한다.

### 📂 포함되어야 할 내용

#### 1. 데이터 이해 및 탐색 (EDA)

* 원본 데이터 설명
  * 컬럼 정의 (의미, 타입, 단위)
  * 타겟 변수 정의 (이탈 여부 기준)
* 기초 통계
  * 결측치 분포
  * 이상치 확인
  * 클래스 불균형 여부

#### 2. 전처리 전략 및 의사결정 근거

* 결측치 처리 방식
  * 제거 / 대체 (mean, median, mode, 특정 값 등)
  * 선택 근거
* 이상치 처리 방식
* 범주형 변수 처리
  * Label Encoding / One-Hot Encoding 선택 이유
* 수치형 변수 스케일링
  * StandardScaler / MinMaxScaler 등

#### 3. Feature Engineering

* 파생 변수 생성 내역
* 불필요한 컬럼 제거 사유
* Feature selection 여부 및 기준

#### 4. 데이터 분할

* Train / Validation / Test 비율
* Stratified split 사용 여부 및 이유
