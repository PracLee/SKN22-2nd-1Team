# 🛒 Online Purchase Insight AI (SKN22-2nd-1Team)

> **"클릭만 하고 나갈 고객인가, 구매할 고객인가?"**

실시간 세션 데이터를 분석하여 구매 확률을 예측하고, 이탈 방지 전략을 제시하는 AI 대시보드 솔루션입니다.

[![Python](https://img.shields.io/badge/Python-3.12%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.52%2B-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)

## 📌 Project Overview
이커머스 환경에서 전체 방문자 중 실제 구매자는 극소수(약 15%)에 불과합니다.

본 프로젝트는 **데이터 불균형(Class Imbalance)** 문제를 극복하고, 구매 가능성이 높은 고객을 사전에 식별하여 마케팅 효율을 극대화하는 것을 목표로 합니다.

### 🎯 Key Objectives
* **Precision Targeting:** 구매 확률이 높은 상위 고객을 정확히 선별
* **Explainability (XAI):** "왜 이 고객이 구매할 것 같은지"에 대한 명확한 근거 제시 (SHAP)
* **Actionable Insight:** 예측 결과를 바탕으로 즉각적인 비즈니스 액션(쿠폰, 넛지) 연결


## 🚀 Key Features
| 기능 | 설명 |
| :--- | :--- |
| **📊 실시간 대시보드** | 모델 성능 비교 및 예측 시뮬레이션을 위한 통합 웹 인터페이스 |
| **🔮 구매 확률 예측** | 고객 행동 데이터 입력 시 즉시 구매 확률(%) 및 등급(High/Low) 산출 |
| **🔍 심층 원인 분석** | **SHAP Waterfall Plot**을 통해 개별 고객의 구매/이탈 요인 시각화 |
| **⚖️ 모델 비교** | F1 최적화 모델과 PR-AUC 최적화 모델의 예측 결과 동시 비교 |



## 📂 Data Dictionary
본 프로젝트는 UCI Machine Learning Repository의 **Online Shoppers Purchasing Intention Dataset**을 사용합니다.

https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset

사용자의 세션 행동, 페이지 체류 시간, 이탈률 등을 분석하여 **구매 여부(`Revenue`)** 를 예측합니다.

| 구분 | 컬럼명 (Feature) | 설명 | 데이터 타입 | 비고 |
| :--- | :--- | :--- | :--- | :--- |
| **행동 데이터**<br>(Behavioral) | `Administrative` | 관리(계정 등) 페이지 방문 횟수 | Integer | |
| | `Administrative_Duration` | 관리 페이지 총 체류 시간 (초) | Float | |
| | `Informational` | 정보성 페이지 방문 횟수 | Integer | |
| | `Informational_Duration` | 정보성 페이지 총 체류 시간 (초) | Float | |
| | `ProductRelated` | **제품 상세 페이지 방문 횟수** | Integer | 구매 의도와 가장 밀접 |
| | `ProductRelated_Duration` | 제품 페이지 총 체류 시간 (초) | Float | |
| **세션 품질**<br>(Quality) | `BounceRates` | **이탈률** (해당 페이지만 보고 나간 비율) | Float | 0.0 ~ 1.0 |
| | `ExitRates` | **종료율** (해당 페이지에서 세션이 끝난 비율) | Float | 0.0 ~ 1.0 |
| | `PageValues` | **페이지 가치** (구글 애널리틱스 지표) | Float | **Feature Importance 1위**<br>(거래 기여도) |
| **고객/환경**<br>(Attributes) | `SpecialDay` | 기념일(발렌타인 등) 근접도 | Float | 0.0(멀음) ~ 1.0(당일) |
| | `Month` | 방문 월 (Month) | Categorical | Feb, Mar, May... |
| | `OperatingSystems` | 운영체제 종류 | Categorical | 정수 인코딩 상태 |
| | `Browser` | 브라우저 종류 | Categorical | 정수 인코딩 상태 |
| | `Region` | 접속 지역 | Categorical | 정수 인코딩 상태 |
| | `TrafficType` | 유입 경로 유형 | Categorical | 배너, 검색, 직접 유입 등 |
| | `VisitorType` | 방문자 유형 | Categorical | Returning, New, Other |
| | `Weekend` | 주말 방문 여부 | Boolean | True / False |
| **타겟**<br>(Target) | `Revenue` | **구매 여부** | Binary | **0: 미구매 (False)**<br>**1: 구매 (True)** |

## 📊 주요 변수 상세 설명

> 데이터 분석 시 혼동하기 쉬운 핵심 지표 3가지(BounceRates, ExitRates, PageValues)에 대한 정의입니다.

### 1. BounceRates (이탈률)

**"첫인상이 안 좋았나?"**

* **정의**: 사용자가 사이트에 들어온 후, **다른 페이지를 클릭하지 않고 바로 나간 비율**입니다.

* **쉽게 말하면**, 어떤 손님이 쇼핑몰 문을 열고 들어오자마자, 진열된 상품을 하나도 안 보고 그냥 다시 문을 닫고 나간 경우입니다. **(튕겨 나갔다)**

* **데이터적 의미**: 이 수치가 높다면 아래와 같은 상황을 의심해 볼 수 있습니다.

    * *"광고를 보고 들어왔는데 찾는 내용이 없네?"* (콘텐츠 불일치)
    * *"사이트가 너무 느리네?"* (기술적 문제)



---

### 2. ExitRates (종료율)

**"여기서 쇼핑을 마쳤네?"**

* **정의**: 특정 페이지를 본 후 **웹사이트를 나간 비율**입니다.

* **쉽게 말하면**, 손님이 매장에서 여러 물건을 구경하다가, **마지막으로 머물렀던 지점**이 어디인지를 나타냅니다.

* **데이터적 의미**:

    * **정상적인 경우**: '결제 완료' 페이지나 '회원가입 완료' 페이지의 종료율이 높은 경우.
    
    * **문제적인 경우**: '장바구니'나 '결제 도중' 페이지에서 종료율이 비정상적으로 높다면 *"결제 과정이 복잡한가?"* 라는 가설을 세울 수 있습니다.

---

### 💡 핵심 비교: 이탈률(Bounce) vs 종료율(Exit)

많은 사람들이 헷갈려 하는 두 변수의 차이는 **방문한 페이지의 수(세션의 길이)** 에 있습니다.

> **"모든 Bounce는 Exit에 포함되지만, 모든 Exit이 Bounce인 것은 아닙니다."**

* **Bounce (이탈)**: 방문 페이지가 **1개**일 때 발생합니다. (들어오자마자 나감 → 즉시 종료)
* **Exit (종료)**: 방문 페이지가 **1개 이상**일 때 발생합니다. (여러 페이지를 보다가 마지막에 나감)

| 구분 | 상황 예시 | Bounce 해당? | Exit 해당? |
| --- | --- | --- | --- |
| **A** | `페이지 A` 입장 → 바로 나감 | **O** | **O** |
| **B** | `페이지 A` 입장 → `페이지 B` 이동 → 나감 | **X** | **O** (페이지 B 기준) |

---

### 3. PageValues (페이지 가치)

**"이 페이지가 돈이 됐나?"**

* **정의**: 사용자가 구매를 완료하기 전에 방문한 페이지의 **평균적인 금전적 가치**입니다.
* **쉽게 말하면**, *"이 페이지를 본 사람들이 실제로 물건을 얼마나 샀는가?"* 를 수치화한 것입니다.
* **계산 공식**:  $$\Huge PageValues = \frac{\text{(해당 페이지를 거친 총 매출액 + 총 목표값)}}{\text{(해당 페이지의 순 조회수)}}$$

* **요소 설명**:
    * **해당 페이지를 거친 총 매출액**: 해당 페이지를 거쳐간 세션에서 발생한 총 결제 금액입니다.

    * **총 목표값**: 회원가입, 뉴스레터 구독 등 사이트가 설정한 목표를 달성했을 때 부여되는 가치입니다.

    * **순 조회수**: 한 세션에서 해당 페이지를 여러 번 새로고침 해도 1회로 카운트한, 순수 방문 횟수입니다.

* **데이터적 의미**:
    * 이 데이터셋에서 **구매 여부(Revenue)를 예측하는 데 가장 강력한 변수** 중 하나입니다.

    * 수치가 높을수록 해당 페이지가 구매 결정에 큰 기여를 했다는 뜻이므로, 마케팅이나 디자인 리소스 투입 시 최우선으로 고려해야 할 페이지입니다.



---


## 📊 Modeling & Performance
데이터 불균형 해결을 위해 **Balanced Random Forest**를 메인 모델로 선정하였으며, 비즈니스 목적에 따라 두 가지 전략으로 최적화했습니다.

### Main Model: Balanced Random Forest

데이터 불균형(Class Imbalance) 문제 해결을 위해 **Balanced Random Forest (BRF)** 알고리즘을 메인 모델로 선정하였습니다.

단, 비즈니스 목적(F1-Score 극대화 vs 순위 예측 신뢰도 확보)에 따라 최적의 성능을 낼 수 있도록 **두 가지 최적화 전략(Two-Track Strategy)** 으로 모델을 이원화하여 구축하였습니다.

- **알고리즘:** Balanced Random Forest Classifier (`imbalanced-learn` 라이브러리 활용)
- **공통 파이프라인 구성:**
    - **Step 1 (Preprocessing):** **`RobustScaler`** (이상치에 강건한 스케일링), `OneHotEncoder` (범주형 변수 변환)
    - **Step 2 (Classifier):** 다수의 Decision Tree가 Majority Class를 Under-sampling 하여 학습하고, 다수결로 최종 예측.

### **1. Type A: F1-Score 최적화 모델 (`best_balancedrf_pipeline`)**

정밀도(Precision)와 재현율(Recall)의 조화 평균인 **F1-Score를 극대화**하는 것을 목표로 합니다. 모델의 구조적 파라미터는 일반적인 설정을 유지하되, 학습 후 **'결정 임계값(Threshold)'을 튜닝**하는 후처리(Post-processing) 전략을 사용하였습니다.

- **최적화 전략 (Threshold Tuning):**
    - 기본 확률(0.5)을 사용하지 않고, Validation 과정에서 F1-Score가 최대가 되는 최적의 임계값을 산출하여 적용
    - 산출된 임계값(예: 0.74)은 메타데이터(`meta`)에 저장되어, 추론 시 동적으로 적용

<img width="680" height="380" alt="image" src="https://github.com/user-attachments/assets/c2e0b520-dd20-4459-8a21-aedaba1993a0" />

> F1 최적화 모델의 OOF F1, Test F1, Test PR-AUC, Test ROC-AUC 요약 막대그래프

- **핵심 하이퍼파라미터 (Base Settings):**
    - `n_estimators`: **100** (기본 설정, 연산 효율성 및 Baseline 성능 확보)
    - `sampling_strategy`: **'auto'** (Majority Class의 샘플 수를 Minority Class와 1:1 비율이 되도록 자동 Under-sampling)
    - `max_features`: **'sqrt'** (개별 트리의 다양성 확보)
 
<img width="1580" height="420" alt="image" src="https://github.com/user-attachments/assets/304d1030-c468-453c-83c1-90b78f51495c" />

> F1 최적화 과정에서 max_depth, min_samples_leaf, max_features에 따른 Test F1 평균 변화 그래프

<img width="680" height="380" alt="image" src="https://github.com/user-attachments/assets/7d88bcc7-819a-4cb0-95dc-4fcf46805023" />

> F1 기준 상위 20개 하이퍼파라미터 조합의 Test F1 랭킹 플롯

### **2. Type B: PR-AUC 최적화 모델 (`best_pr_auc_balancedrf`)**

이 모델은 불균형 데이터 평가에 가장 적합한 **PR-AUC (Precision-Recall Area Under Curve)** 점수를 높이는 것을 목표로 합니다.

임계값 조정보다는 사전 실험을 통해 도출된 **최적의 하이퍼파라미터 조합을 고정(Fixed)** 하여, 모델이 출력하는 확률값(Probability) 자체의 정교함을 높이는 데 주력하였습니다.

- **최적화 전략 (Fixed Hyperparameters):**
    - PR-AUC 점수가 가장 높았던 파라미터 조합을 고정하여 학습
    - 트리의 개수를 늘리고 깊이를 제한하여, 일반화 성능을 높이고 과적합(Overfitting)을 억제
 
<img width="580" height="380" alt="image" src="https://github.com/user-attachments/assets/8d5c3787-498d-40cd-b6b2-57ea751a9e66" />

> PR-AUC 최적화 모델의 교차검증(CV) PR-AUC와 테스트(Test) PR-AUC/ROC-AUC 막대그래프
  
- **핵심 하이퍼파라미터 (Optimized Settings):**
    - `n_estimators`: **300** (Type A 대비 3배 증가시켜 예측의 분산을 줄이고 안정성 확보)
    - `max_depth`: **8** (트리의 깊이를 제한하여 훈련 데이터에 대한 과적합 방지)
    - `max_features`: **0.3** (전체 피처의 30%만 무작위로 사용하여 개별 트리의 독립성 강화)
    - `sampling_strategy`: **0.5** (Majority Class를 Minority Class의 2배수(0.5 비율) 정도로 Under-sampling 하여 정보 손실 최소화)
    - `min_samples_split`: **5** (노드 분할을 위한 최소 샘플 수를 높여 보수적인 학습 유도)

<img width="600" height="1000" alt="image" src="https://github.com/user-attachments/assets/efff2741-9633-4768-a9ce-6a2892166ba9" />

> PR-AUC 탐색 과정에서의 n_estimators, max_depth, min_samples_leaf 등 주요 하이퍼파라미터 값에 따른 평균 CV PR-AUC 변화 그래프

<img width="580" height="480" alt="image" src="https://github.com/user-attachments/assets/9a4c8527-9939-45db-ae73-ddd0467b3320" />

> PR-AUC 기준 상위 20개 하이퍼파라미터 조합의 CV PR-AUC 랭킹 플롯

### 💡 Feature Importance (특성 중요도)
모델이 구매 여부를 판단할 때 가장 중요하게 고려한 상위 3가지 변수는 다음과 같습니다.

1.  **PageValues (페이지 가치):** 압도적인 중요도 1위. 사용자가 방문한 페이지가 실제 매출에 얼마나 기여했는지를 나타내며, 이 값이 높을수록 구매 확률이 급격히 상승합니다.
2.  **ExitRates (종료율):** 해당 페이지에서 세션을 종료하는 비율이 낮을수록(즉, 사이트에 더 머무를수록) 구매 가능성이 높습니다.
3.  **ProductRelated_Duration (제품 페이지 체류 시간):** 제품을 오래 탐색할수록 관심도가 높다고 판단하여 긍정적인 영향을 미칩니다.

> **XAI Insight:** SHAP 분석 결과, 단순히 체류 시간이 긴 것보다 **'목적성 있는 탐색(PageValues)'** 이 구매 예측에 훨씬 강력한 신호임이 입증되었습니다.

## 📜 Conclusion & Usage Plan

### 1. 결론 (Conclusion)
* **불균형 데이터 정복:** Balanced Random Forest와 Robust Scaling을 적용하여, 데이터 불균형 환경에서도 **Recall 0.791**이라는 높은 재현율을 달성했습니다.
* **실용성 확보:** 단순 예측을 넘어 **Streamlit 대시보드**와 **SHAP 설명력**을 결합함으로써, 현업 마케터가 즉시 활용 가능한 수준의 BI(Business Intelligence) 도구를 구축했습니다.

### 2. 활용 방안 (Usage Plan)
* **초개인화 마케팅:** 구매 확률 상위 20% 고객(High Probability)에게 **시크릿 쿠폰**을 발송하여 구매 전환율(CVR) 극대화.
* **이탈 방지 (Churn Prevention):** 장바구니에 담았으나 결제하지 않고 나갈 확률이 높은 고객(Borderline)에게 **실시간 팝업(넛지)** 제공.
* **UI/UX 개선:** `ExitRates`가 비정상적으로 높은 페이지를 식별하여, 고객 여정(Customer Journey)의 병목 구간을 개선.

### 3. 향후 계획 (Roadmap)
* **Model Stacking:** RF와 DNN의 예측값을 결합하는 메타 모델(Meta-model)을 도입하여 예측 안정성 추가 확보.
* **MLOps Pipeline:** 신규 데이터 유입 시 자동으로 모델을 재학습하고 배포하는 CI/CD 파이프라인 구축.

---

## 📂 Directory Structure

```bash
SKN22-2nd-1Team/
├── app/                 # Streamlit Web Application Root
│   ├── app.py           # 메인 대시보드 실행 파일
│   ├── adapters/        # 모델 로더 및 시각화 어댑터
│   ├── artifacts/       # 학습된 모델 및 전처리기 저장소
│   ├── pages/           # Streamlit 다중 페이지 (XAI, 모델 비교 등)
│   ├── service/         # 비즈니스 로직 및 분석 서비스 모듈
│   └── ui/              # 공통 UI 컴포넌트
│       └── header.py    # 상단 네비게이션/헤더 모듈
├── data/                # 데이터 저장소
│   ├── raw/             # 원본 데이터 (UCI Online Shoppers)
│   └── processed/       # 전처리 및 분할 완료된 데이터
├── script/              # 자동화 스크립트
└── requirements.txt     # 프로젝트 의존성 관리
```

## 🛠 Tech Stack
**Language:** Python 3.12+

**Framework:** Streamlit

**ML / DL:** Scikit-learn, TensorFlow (Keras), Imbalanced-learn

**XAI:** SHAP

**Data Processing:** Pandas, NumPy

**Visualization:** Matplotlib, Seaborn

## 👪 팀원

| **문승준** | **엄형은** | **이병재** | **최정환** | **황하령** |
| :---: | :---: | :---: | :---: | :---: |
| [![GitHub](https://img.shields.io/badge/GitHub-fleshflamer-181717?style=flat&logo=github&logoColor=white)](https://github.com/fleshflamer-commits) | [![GitHub](https://img.shields.io/badge/GitHub-DJAeun-181717?style=flat&logo=github&logoColor=white)](https://github.com/DJAeun) | [![GitHub](https://img.shields.io/badge/GitHub-PracLee-181717?style=flat&logo=github&logoColor=white)](https://github.com/PracLee) | [![GitHub](https://img.shields.io/badge/GitHub-hwany--ai-181717?style=flat&logo=github&logoColor=white)](https://github.com/hwany-ai) | [![GitHub](https://img.shields.io/badge/GitHub-harry1749-181717?style=flat&logo=github&logoColor=white)](https://github.com/harry1749) ||
