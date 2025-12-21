## 1. 문제 정의 & 데이터 개요

* **문제 정의**
  * 세션 단위로 *이번 방문에서 구매(Revenue=1)가 발생할지*를 예측하는 이진 분류 문제
  * 단순 정확도(Accuracy)가 아니라

    **“구매 고객을 얼마나 잘 찾아내는지”** 가 핵심
* **데이터**
  * UCI Online Shoppers Purchasing Intention Dataset
  * 주요 feature
    * 행동 데이터: `Administrative`, `Informational`, `ProductRelated` + 각 Duration
    * 세션 품질: `BounceRates`, `ExitRates`, `PageValues`, `SpecialDay`
    * 고객/세션 속성: `Month`, `OperatingSystems`, `Browser`, `Region`, `TrafficType`, `VisitorType`, `Weekend`
  * 클래스 불균형
    * Revenue=1(구매) 비율이 낮은 전형적인 이커머스 데이터
    * → **불균형 데이터에 강한 알고리즘 + 적절한 평가지표 선택 필요**

---

## 2. 평가 지표 설계 (ROC-AUC, PR-AUC, F1)

* **ROC-AUC**
  * TPR(재현율) – FPR(위양성률) 관계를 보는 전통적인 이진 분류 지표
  * **전체 구간에서의 분류 성능**을 넓게 파악하는 데 유리
* **PR-AUC (Average Precision)**
  * 정밀도(Precision) – 재현율(Recall) 곡선 아래 면적
  * 양성(구매) 비율이 매우 낮을 때,

    **“예측한 구매 중에서 진짜 구매가 얼마나 많은지”**를 잘 반영
* **F1 Score**
  * Precision과 Recall의 조화 평균
  * 특정 threshold에서 **양성/음성 오분류의 균형을 실제 현업에서 그대로 쓰기 좋은 지표**
* **우리의 전략**
  * 하나의 지표만 보지 않고,
    * **PR-AUC 최대화 모델** : *“구매 고객 타깃팅”*에 초점
    * **F1 Score 최대화 모델** : *“실제 운영에서 쓸 threshold까지 포함한 균형형 모델”*로 사용
  * 그래서 **최종적으로 Balanced Random Forest 기반 모델 2개를 선정하고 각각 joblib으로 저장**
    * `best_pr_auc_balancedrf.joblib`  (PR-AUC 최적화)
    * `best_balancedrf_pipeline.joblib` (F1 Score 최적화)

---

## 3. 하이퍼파라미터 탐색 전략

* **사용 알고리즘**
  * `BalancedRandomForestClassifier` 기반 파이프라인
  * 전처리(인코딩/스케일링) + 모델을 하나의 pipeline으로 묶어 joblib으로 저장
* **탐색 방법**
  * 1단계: RandomizedSearchCV로 **넓은 범위 랜덤 탐색**
  * 2단계: 유망한 구간을 중심으로 GridSearchCV / 추가 실험
  * 공통 설정
    * 교차검증: K-Fold CV
    * 불균형 데이터 대응: Balanced RF 자체의 샘플링 + `class_weight`/`sampling_strategy` 조정
* **탐색한 대표 파라미터 공간 (예시)**
  * `n_estimators` : 100, 300, 500
  * `max_depth` : None, 5, 8, 10, 15
  * `min_samples_split` : 2, 5
  * `min_samples_leaf` : 1, 2, 5
  * `max_features` : `"sqrt"`, `0.3`, `0.5`
  * `sampling_strategy` : 0.5, 0.7, 1.0
  * `replacement` : True, False
* **지표별 별도 탐색**
  * **PR-AUC용 탐색**
    * `scoring="average_precision"` 으로 설정
    * best estimator → `best_pr_auc_balancedrf.joblib`
  * **F1 Score용 탐색**
    * threshold를 튜닝하면서 F1을 최대화
    * best pipeline + threshold 조합 → `best_balancedrf_pipeline.joblib` + 최종 threshold 저장

---

## 4. Best PR-AUC 모델 (`best_pr_auc_balancedrf.joblib`)

### 4-1. 최종 하이퍼파라미터

<pre class="overflow-visible! px-0!" data-start="2381" data-end="2643"><div class="contain-inline-size rounded-2xl corner-superellipse/1.1 relative bg-token-sidebar-surface-primary"><div class="@w-xl/main:top-9 sticky top-[calc(--spacing(9)+var(--header-height))]"><div class="absolute end-0 bottom-0 flex h-9 items-center pe-2"><div class="bg-token-bg-elevated-secondary text-token-text-secondary flex items-center gap-4 rounded-sm px-2 font-sans text-xs"></div></div></div><div class="overflow-y-auto p-4" dir="ltr"><code class="whitespace-pre! language-text"><span><span>=== Using fixed best params ===
{
  "model__sampling_strategy": 0.5,
  "model__replacement": false,
  "model__n_estimators": 300,
  "model__min_samples_split": 5,
  "model__min_samples_leaf": 1,
  "model__max_features": 0.3,
  "model__max_depth": 8
}
</span></span></code></div></div></pre>

* 해석 포인트
  * **`n_estimators=300`**

    → 충분히 많은 트리로 복잡한 패턴을 포착하면서도 속도는 과하지 않게 유지
  * **`max_depth=8` / `min_samples_leaf=1` / `min_samples_split=5`**

    → 깊이를 너무 깊게 두지 않고, split 규칙을 약간 엄격하게 만들어

    과적합은 막으면서도 양성 패턴을 세밀하게 캡처
  * **`max_features=0.3`**

    → 각 split에서 전체 feature의 30%만 사용해

    트리 간 다양성을 높이고, PR-AUC 향상에 기여
  * **`sampling_strategy=0.5` + `replacement=False`**

    → 각 트리 학습 시 소수 클래스(구매)를 상대적으로 많이 포함하되,

    완전 1:1 균형까지는 아니게 조정하여 안정적인 학습 유도

### 4-2. 테스트 성능

<pre class="overflow-visible! px-0!" data-start="3140" data-end="3242"><div class="contain-inline-size rounded-2xl corner-superellipse/1.1 relative bg-token-sidebar-surface-primary"><div class="@w-xl/main:top-9 sticky top-[calc(--spacing(9)+var(--header-height))]"><div class="absolute end-0 bottom-0 flex h-9 items-center pe-2"><div class="bg-token-bg-elevated-secondary text-token-text-secondary flex items-center gap-4 rounded-sm px-2 font-sans text-xs"></div></div></div><div class="overflow-y-auto p-4" dir="ltr"><code class="whitespace-pre! language-text"><span><span>=== TEST (reference) ===
TEST PR-AUC : 0.7466332225549133
TEST ROC-AUC: 0.9302589663454292
</span></span></code></div></div></pre>

* **PR-AUC ≈ 0.747**
  * 구매(양성) 비율이 낮은 환경에서도

    **“예측한 구매 중에서 진짜 구매 비율”**이 상당히 높음을 의미
* **ROC-AUC ≈ 0.93**
  * 전체 세션에 대해 전반적인 분류 능력이 매우 양호한 수준
* 활용 시나리오
  * **마케팅 캠페인 / 타깃팅** 처럼

    “예측한 고객에게 실제로 잘 맞아야 하는 상황”에서 기본 모델로 사용

---

## 5. Best F1 모델 (`best_balancedrf_pipeline.joblib`)

> 이 모델은 **특정 threshold에서의 Precision–Recall 균형(F1 Score)을 최대로 만드는 것**을 목표로 선택했다.

### 5-1. 최종 Threshold & 성능

<pre class="overflow-visible! px-0!" data-start="3639" data-end="3844"><div class="contain-inline-size rounded-2xl corner-superellipse/1.1 relative bg-token-sidebar-surface-primary"><div class="@w-xl/main:top-9 sticky top-[calc(--spacing(9)+var(--header-height))]"><div class="absolute end-0 bottom-0 flex h-9 items-center pe-2"><div class="bg-token-bg-elevated-secondary text-token-text-secondary flex items-center gap-4 rounded-sm px-2 font-sans text-xs"></div></div></div><div class="overflow-y-auto p-4" dir="ltr"><code class="whitespace-pre! language-text"><span><span>=== TEST Evaluation (using saved threshold) ===
Threshold: 0.741237
ROC-AUC : 0.9319585271979982
PR-AUC : 0.7415823565968093
F1 : 0.6740467404674046
Confusion matrix:
[[1927  157]
 [ 108  274]]
</span></span></code></div></div></pre>

* **Threshold ≈ 0.741**

  * 예측 확률이 0.741 이상일 때만 “구매”로 판정
  * → 다소 보수적인 기준으로, **거의 살 것 같은 고객 위주로 positive 예측**
* **F1 ≈ 0.674**

  * Precision과 Recall의 균형이 잘 맞는 지점
  * 실제 운영에서 “이 threshold로 바로 배포할 수 있는” 상태의 모델
* **ROC-AUC ≈ 0.932 / PR-AUC ≈ 0.742**

  * PR-AUC 기준 최적 모델과 매우 유사한 수준의 전반적인 분류 능력 유지
* **Confusion Matrix 해석**

  |                 | 실제 0 (미구매) | 실제 1 (구매) |
  | --------------- | --------------- | ------------- |
  | 예측 0 (미구매) | 1927            | 108           |
  | 예측 1 (구매)   | 157             | 274           |


  * **TP = 274** : 실제 구매를 정확히 맞춘 세션
  * **FN = 108** : 실제 구매인데 놓친 세션
  * **FP = 157** : 실제는 안 샀는데, 산다고 예측한 세션
  * Threshold를 0.741로 잡으면서
    * FP를 너무 늘리지 않으면서
    * FN도 과하게 늘리지 않는 **실무용 균형점**으로 설정
* 활용 시나리오

  * **운영팀에서 바로 액션을 걸어야 하는 경우**

    * 예: “구매 예측 고객에게 쿠폰/알림 발송”과 같이

      실제 비용이 드는 액션에 활용할 때
  * threshold까지 포함해 튜닝해 두었기 때문에

    **모델 + threshold를 그대로 서빙에 사용할 수 있음**

---

## 6. 두 모델 비교 & 운영 전략

### 6-1. 모델 비교 요약

| 구분                  | 파일명                              | 최적화 기준                       | 주요 활용 목적                                                |
| --------------------- | ----------------------------------- | --------------------------------- | ------------------------------------------------------------- |
| **PR-AUC 모델** | `best_pr_auc_balancedrf.joblib`   | PR-AUC (0.7466)                   | 구매 고객 타깃팅, 성능 리포팅/분석용                          |
| **F1 모델**     | `best_balancedrf_pipeline.joblib` | F1 Score (0.674, threshold=0.741) | 실운영 액션(푸시/쿠폰 발송 등)에 바로 활용 가능한 실무형 모델 |

* 둘 다 **Balanced Random Forest 기반**이라
  * 불균형 데이터에 대한 강인함
  * feature importance / rule 해석이 가능해

    **기획/마케팅 관점 설명에도 유리**

### 6-2. 실제 서비스에서의 선택 로직

* **기본 대시보드 / 분석 페이지**
  * PR-AUC 기준 모델(`best_pr_auc_balancedrf.joblib`) 사용
  * 다양한 threshold 시나리오를 자유롭게 시뮬레이션
* **실제 액션이 걸리는 운영 시나리오**
  * F1 기반 모델(`best_balancedrf_pipeline.joblib`) + 저장된 threshold=0.741237 사용
  * → Precision·Recall의 균형이 검증된 상태로 바로 적용 가능

## 7. 하이퍼파라미터 튜닝까지 포함한 모델 선정 스토리

* 불균형 데이터 환경에서
  * **Balanced Random Forest**를 기본 알고리즘으로 선정
* 그 위에서
  * **PR-AUC 최적화**를 통해 “구매 고객을 잘 찾는” 모델 확보
  * **F1 최적화 + threshold 튜닝**을 통해 “실무에서 바로 쓸 수 있는” 모델 확보
* 두 모델을 각각 joblib으로 저장해
  * Streamlit 앱(세션 시뮬레이터, 페르소나 생성기)에서

    **상황에 따라 모델을 선택적으로 호출**할 수 있는 구조로 설계
