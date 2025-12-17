
### 📌 목적 (Purpose)

전처리된 데이터를 기반으로 **모델을 학습하고 비교·평가한 결과를 체계적으로 정리**한다.

### 📂 포함되어야 할 내용

#### 1. 문제 정의

* 문제 유형: Binary Classification (가입 고객 이탈 예측)
* 평가 지표 선정 이유
  * Accuracy vs Precision / Recall / F1 / ROC-AUC

#### 2. 사용 모델 목록

* Baseline 모델
  * Logistic Regression
* Tree 기반 모델
  * RandomForest
  * XGBoost / LightGBM / CatBoost (선택 시 이유)
* (선택) 딥러닝 모델

#### 3. 학습 설정

* 주요 하이퍼파라미터
* Cross-validation 여부
* Early stopping 사용 여부

#### 4. 모델별 성능 비교

* Train / Validation 성능
* Confusion Matrix
* ROC Curve
* 과적합 여부 분석

#### 5. 최종 모델 선정 이유

* 단순 성능이 아닌 **비즈니스 관점 판단**
  * 이탈 고객을 놓치지 않는 모델
  * 해석 가능성 고려 여부
