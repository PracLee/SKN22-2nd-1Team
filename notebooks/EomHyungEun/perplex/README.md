# 🛒 Online Shoppers Intent Prediction - 실행 가이드

## 📋 전체 구조

```
project/
├── 1️⃣ 데이터 & 모델 학습
│   ├── online_shoppers_intention.csv (원본 데이터)
│   └── 모델 학습 스크립트 (위에서 실행한 파이썬 코드)
│
├── 2️⃣ 학습된 모델 (자동 생성됨)
│   ├── models_trained.pkl (LR, RF, GB 모델)
│   ├── metadata.json (메타데이터 & 성능 정보)
│   ├── scaler.pkl (StandardScaler - LR용)
│   ├── label_encoders.pkl (범주형 인코더)
│   └── data_clean.csv (정제된 데이터)
│
└── 3️⃣ Streamlit 앱
    └── streamlit_app.py (메인 앱 파일)
```

---

## 🚀 실행 방법

### 1단계: 필수 라이브러리 설치

```bash
pip install streamlit pandas numpy scikit-learn plotly
```

### 2단계: 모델 학습 (첫 번째만 필요)

위에서 제공한 "파트 1-7: 모델 학습 완전 파이프라인" 코드를 실행합니다.

**결과 파일:**
- ✅ models_trained.pkl
- ✅ metadata.json
- ✅ scaler.pkl
- ✅ label_encoders.pkl
- ✅ data_clean.csv

### 3단계: Streamlit 앱 실행

```bash
streamlit run streamlit_app.py
```

**출력:**
```
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.x.x:8501
```

브라우저에서 `http://localhost:8501` 열기!

---

## 🎯 10가지 기능 상세 가이드

### 1️⃣ 세션 구매 확률 계산기

**목표:** 한 고객 세션의 구매 확률을 즉시 예측

**입력:**
- 페이지 방문 수 (Administrative, Informational, ProductRelated)
- 체류시간 (초 단위)
- 행동 지표 (Bounce Rate, Exit Rate, Page Values)
- 범주형 정보 (Month, Visitor Type, Weekend 등)

**출력:**
- 구매 확률 (%)
- 평균 대비 높음/낮음
- 자동 생성 인사이트
- 마케팅 액션 추천
- 모델별 예측 비교 (LR vs RF vs GB)

**활용 시나리오:**
- 실시간 방문자 분석
- 타겟 고객 식별

---

### 2️⃣ What-If 시뮬레이터

**목표:** "~를 개선하면 구매 확률이 얼마나 올라?"를 슬라이더로 탐색

**기능:**
- 기본값 설정 (현재 상태)
- 슬라이더로 실시간 변수 조정
- 변화 전/후 비교
- 최적의 조합 찾기

**슬라이더 대상:**
- BounceRates, ExitRates, PageValues
- ProductRelated_Duration, Admin Pages 등

**활용 시나리오:**
- "UX 개선하면 구매율 몇 %?", "프로모션 효과 시뮬레이션"
- 마케터/기획자의 의사결정 지원

---

### 3️⃣ 채널 효과 분석 대시보드

**목표:** TrafficType, Region, Browser별 구매 성과 분석

**분석 기준:**
- TrafficType (어디서 왔는가?)
- Region (어느 지역인가?)
- Browser (어떤 기기인가?)
- OperatingSystems

**지표:**
- 구매율 (%) - 가장 중요
- 세션 수 - 트래픽 규모
- 평균 구매 확률 - 모델 예측 평균

**활용 시나리오:**
- "트래픽은 많은데 구매가 적은 채널" 발견
- "높은 효율성 채널" 확장 전략

---

### 4️⃣ 고위험 이탈 세션 탐지기

**목표:** 이탈 위험도를 진단하고 액션 추천

**위험도 평가:**
- 🟢 낮음 (< 30%) - 구매 가능성 높음
- 🟡 중간 (30-70%) - 적극적 개입 필요
- 🔴 높음 (> 70%) - 즉시 액션 필요

**자동 추천 액션:**
- 높은 Bounce/Exit → 랜딩 페이지 개선
- 낮은 Page Value → 상품 추천 강화
- 짧은 체류 → 정보 노출 강화

---

### 5️⃣ EDA 대시보드 탭

**구성:**

**탭 1: Overview**
- 데이터셋 통계 (세션 수, 구매율 등)
- Revenue 분포 (파이 차트)

**탭 2: Behavior**
- 월별 구매율 추이 (라인 차트)
- VisitorType별 구매율 (바 차트)
- 평일/주말 비교

**탭 3: Features**
- PageValues, BounceRates 등 분포 (히스토그램)
- Revenue별 비교 (Overlay)

**탭 4: Correlation**
- 변수별 상관계수
- Feature Importance

**탭 5: Model**
- 모델 성능 표
- 메트릭 비교 (Accuracy, F1, ROC-AUC)

---

### 6️⃣ Feature Importance & Explainability

**목표:** 모델의 의사결정 근거를 시각화

**표시 내용:**
- Top 15 피처 중요도 (막대 그래프)
- 선택 피처의 분포 비교 (비구매 vs 구매)

**RF/GB 모델에서만 가능** (LR은 제한적)

---

### 7️⃣ 고객 페르소나 생성기

**사전 정의된 페르소나 5가지:**

1. **🆕 신규 고객 (정보 탐색)**
   - 첫 방문, 상품 탐색만 함
   - 구매 확률: 낮음

2. **🔄 재방문자 (상품 비교)**
   - 재방문, 상품 페이지 깊은 탐색
   - 구매 확률: 중간~높음

3. **💰 구매 직전 (고의도 구매자)**
   - 높은 페이지 가치, 낮은 이탈
   - 구매 확률: 매우 높음

4. **❌ 높은 이탈 위험**
   - 높은 Bounce/Exit Rate
   - 구매 확률: 매우 낮음

5. **🎯 이상적인 고객**
   - 낮은 이탈, 높은 참여
   - 구매 확률: 최고

---

### 8️⃣ 시나리오 A vs B 비교

**목표:** 두 세션을 나란히 비교하여 어느 것이 더 나은지 판단

**예시:**
- "Returning vs New Visitor 중 누가 구매할까?"
- "Weekend vs Weekday 중 어느 쪽이 더 사고 싶어할까?"

**비교 결과:**
- 구매 확률 수치
- 백분포인트 차이
- 승자 표시

---

### 9️⃣ 모델 성능 비교 탭

**표시 내용:**
- 모델별 성능 표 (Accuracy, Precision, Recall, F1, ROC-AUC)
- 메트릭별 그래프

**현재 성능 (Test Set):**
- **LR**: Accuracy 87.6%, ROC-AUC 91.1%
- **RF**: Accuracy 89.3%, ROC-AUC 92.8%
- **GB**: Accuracy 90.0%, ROC-AUC 93.1% ⭐ (최고)

---

### 🔟 마케팅 액션 추천 카드

**목표:** 예측 결과 기반 실행 가능한 마케팅 액션 제시

**제시 내용:**
1. 즉시 개입 필요 여부
2. 우선순위별 액션
3. 각 액션별 세부 실행 사항

**예시 액션:**
- 가격 할인/쿠폰
- 페이지 최적화
- 상품 추천 강화
- 신뢰 요소 추가

---

## 📊 모델 성능 요약

| 모델 | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|------|----------|-----------|--------|----------|---------|
| **LR** | 0.8763 | 0.5766 | 0.7880 | 0.6659 | 0.9111 |
| **RF** | 0.8931 | 0.6397 | 0.7251 | 0.6798 | 0.9278 |
| **GB** | 0.8992 | 0.6977 | 0.6283 | 0.6612 | 0.9313 ⭐ |

**추천:** GB (Gradient Boosting) - 최고의 ROC-AUC 및 균형잡힌 성능

---

## 💾 저장된 파일 설명

### models_trained.pkl
- LR, RF, GB 3가지 모델 객체
- 각 모델의 학습 완료 상태

### metadata.json
- 피처 컬럼명
- 범주형 인코더 클래스
- 모델 성능 메트릭
- 데이터셋 통계

### scaler.pkl
- StandardScaler 객체 (LR용)
- 신규 데이터 전처리에 사용

### label_encoders.pkl
- Month 인코더
- VisitorType 인코더
- 범주형 변수 변환용

### data_clean.csv
- 정제 & 전처리된 데이터
- EDA 분석용

---

## 🔧 트러블슈팅

### 1. "No such file or directory" 에러

**원인:** 모델 파일이 없음  
**해결:**
1. 모델 학습 파이프라인 코드 재실행
2. 같은 폴더에 모든 pkl 파일이 있는지 확인

### 2. 앱 느려짐

**원인:** @st.cache_resource가 제대로 작동 안 함  
**해결:**
```bash
streamlit run streamlit_app.py --client.caching.enable_cache=true
```

### 3. 예측값이 이상함

**원인:** 입력 값 범위 벗어남  
**해결:** 슬라이더 범위 안내문 참조

---

## 📈 데이터 인사이트

**주요 발견:**

1. **PageValues가 가장 강력한 신호** (상관계수: 0.49)
   - PageValues = 0인 경우: 구매율 3.85%
   - PageValues > 0인 경우: 구매율 56.34%

2. **ExitRates & BounceRates가 이탈 지표** (상관계수: -0.21, -0.15)
   - 낮을수록 구매 확률 높음

3. **계절성 존재** (11월 구매율 25.4% vs 2월 1.63%)
   - 프로모션 시즌 효과 명확

4. **신규 방문자 > 재방문자** (24.91% vs 13.93%)
   - 광고/프로모션 효과 강함

---

## 💡 사용 팁

### Tip 1: 빠른 테스트
1. 페르소나 기능으로 5가지 시나리오 한번에 보기
2. 모든 패턴을 빠르게 이해

### Tip 2: 프레젠테이션용
1. EDA 대시보드 캡처
2. 모델 성능 비교 표 캡처
3. What-If 시뮬레이터 결과 캡처

### Tip 3: 데이터 분석가용
1. 채널 분석 → 마케팅 최적화
2. Feature Importance → 피처 선택
3. EDA → 가설 수립

---

## 🎓 학습 가치

이 프로젝트를 통해 배울 수 있는 것:

1. **End-to-End ML 프로젝트**
   - 데이터 전처리, 모델 학습, 평가, 배포

2. **Streamlit 앱 개발**
   - 인터랙티브 대시보드 구축
   - 실시간 예측 시스템

3. **실전 데이터 과학**
   - 불균형 데이터 처리
   - 다중 모델 비교
   - 비즈니스 인사이트 도출

4. **마케팅 분석**
   - 고객 이탈 분석
   - 채널 효율성 평가
   - 액션 추천 시스템

---

## 🚀 다음 단계

### 레벨 1: 기본 활용
- [ ] 앱 실행 후 10가지 기능 모두 테스트
- [ ] 각 페이지에서 인사이트 찾기

### 레벨 2: 심화 분석
- [ ] 실제 비즈니스 데이터 적용
- [ ] 채널별 최적화 전략 수립
- [ ] A/B 테스트 설계

### 레벨 3: 고도화
- [ ] SHAP 분석 추가 (모델 설명)
- [ ] 예측 모델 자동 재학습
- [ ] 데이터베이스 연동
- [ ] REST API로 배포

---

**준비 완료! Streamlit 앱을 시작하세요! 🚀**
