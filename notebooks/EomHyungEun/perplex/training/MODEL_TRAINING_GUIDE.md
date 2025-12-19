# 🎯 모델 학습 노트북 - 실행 가이드

## 📋 개요

이 노트북은 **여러 머신러닝 모델을 체계적으로 학습하고 비교**하는 완전한 파이프라인입니다.

**포함된 모델 (7가지):**
1. Logistic Regression
2. Decision Tree
3. Random Forest
4. Gradient Boosting
5. XGBoost (선택사항)
6. LightGBM (선택사항)
7. CatBoost (선택사항)

---

## 🚀 빠른 시작

### 1. 필수 라이브러리 설치

```bash
# 기본 라이브러리
pip install pandas numpy matplotlib seaborn scikit-learn

# 고급 모델 (선택사항)
pip install xgboost lightgbm catboost
```

### 2. 데이터 준비

노트북은 다음 경로에서 데이터를 로드합니다:
```
../../data/processed/train.csv
../../data/processed/test.csv
```

**필요한 경우 경로를 수정하세요:**
```python
train_df = pd.read_csv("YOUR_PATH/train.csv")
test_df = pd.read_csv("YOUR_PATH/test.csv")
```

### 3. 노트북 실행

**옵션 A: Jupyter Notebook**
```bash
jupyter notebook
# 또는
jupyter lab
```

**옵션 B: VS Code**
- `.py` 파일을 VS Code에서 열기
- `# %%` 마커가 있는 셀을 인터랙티브하게 실행

**옵션 C: Google Colab**
- 파일을 Colab에 업로드
- 각 셀을 순차적으로 실행

---

## 📊 노트북 구조 (16개 섹션)

| 섹션 | 내용 | 예상 시간 |
|------|------|-----------|
| 1. 라이브러리 Import | 필요한 모든 라이브러리 로드 | 10초 |
| 2. 데이터 로드 | train/test 데이터 읽기 | 5초 |
| 3. 데이터 전처리 | 타깃/피처 분리 | 5초 |
| 4. 범주형 인코딩 | LabelEncoder 적용 | 5초 |
| 5. 스케일링 | StandardScaler 적용 | 5초 |
| 6. 모델 정의 | 7가지 모델 설정 | 1초 |
| **7. 모델 학습 & 평가** | **핵심 - 모든 모델 학습** | **2-5분** |
| 8. 결과 비교 | 성능 비교표 생성 | 1초 |
| 9. 시각화 - 성능 비교 | 4가지 차트 (Acc, F1, AUC, Time) | 10초 |
| 10. Precision-Recall-F1 | 세부 메트릭 비교 | 5초 |
| 11. ROC Curve | 모든 모델 ROC 곡선 | 5초 |
| 12. Confusion Matrix | 최고 모델의 혼동 행렬 | 5초 |
| 13. Feature Importance | 트리 기반 모델 피처 중요도 | 10초 |
| 14. 최종 요약 | 메트릭별 최고 모델 | 1초 |
| 15. 모델 저장 | 최고 모델 pickle 저장 | 1초 |
| 16. 마무리 | 생성된 파일 목록 | 1초 |

**총 예상 시간: 약 3-6분**

---

## 📈 생성되는 결과물

### 📄 CSV 파일 (2개)
1. **model_comparison_results.csv** - 모든 모델 성능 비교표
2. **feature_importance_*.csv** - 각 모델별 피처 중요도 (트리 기반)

### 📊 이미지 파일 (5-10개)
1. **model_comparison_charts.png** - 4가지 메트릭 비교
2. **precision_recall_f1_comparison.png** - 정밀도/재현율/F1 비교
3. **roc_curves_comparison.png** - ROC 곡선 비교
4. **confusion_matrix_*.png** - 최고 모델 혼동 행렬
5. **feature_importance_*.png** - 각 트리 모델별 피처 중요도

### 💾 모델 파일 (1-2개)
1. **best_model_*.pkl** - 최고 성능 모델
2. **scaler_for_best_model.pkl** - 스케일러 (필요시)

---

## 🎯 핵심 섹션 설명

### 섹션 7: 모델 학습 & 평가

가장 중요한 섹션입니다. 다음을 수행합니다:

```python
for name, config in models.items():
    # 1. 데이터 선택 (스케일 여부)
    # 2. 모델 학습
    # 3. 예측 (train/test)
    # 4. 성능 평가 (Acc, Prec, Rec, F1, AUC)
    # 5. 결과 저장
```

**출력 예시:**
```
================================================================================
🔹 Random Forest 학습 중...
================================================================================

📊 Train 성능:
   Accuracy:  0.9856
   Precision: 0.9234
   Recall:    0.9567
   F1-Score:  0.9398
   ROC-AUC:   0.9945

📊 Test 성능:
   Accuracy:  0.8931
   Precision: 0.6397
   Recall:    0.7251
   F1-Score:  0.6798
   ROC-AUC:   0.9278

⏱️  학습 시간: 12.45초
✅ 완료
```

### 섹션 13: Feature Importance

트리 기반 모델만 가능합니다:
- Random Forest
- Gradient Boosting
- XGBoost
- LightGBM
- CatBoost
- Decision Tree

각 모델별로:
- Top 15 피처 시각화
- 전체 피처 중요도 CSV 저장

---

## 💡 사용 팁

### Tip 1: 빠른 테스트
모델 수를 줄여서 빠르게 테스트:
```python
# XGBoost, LightGBM, CatBoost만 주석 처리
# models에서 제거하면 해당 모델은 학습 안 됨
```

### Tip 2: 하이퍼파라미터 조정
각 모델의 파라미터를 섹션 6에서 수정:
```python
models['Random Forest'] = {
    'model': RandomForestClassifier(
        n_estimators=200,  # 기본 150 → 200으로 증가
        max_depth=20,      # 기본 15 → 20으로 증가
        # ...
    ),
    'use_scaled': False
}
```

### Tip 3: 메모리 절약
큰 데이터셋의 경우:
```python
# 모델 학습 후 즉시 저장하고 메모리에서 제거
del model
import gc
gc.collect()
```

### Tip 4: GPU 사용 (XGBoost, LightGBM)
```python
# XGBoost
models['XGBoost'] = {
    'model': xgb.XGBClassifier(
        tree_method='gpu_hist',  # GPU 사용
        # ...
    ),
    'use_scaled': False
}

# LightGBM
models['LightGBM'] = {
    'model': lgb.LGBMClassifier(
        device='gpu',  # GPU 사용
        # ...
    ),
    'use_scaled': False
}
```

---

## 🔧 트러블슈팅

### 1. "ModuleNotFoundError: No module named 'xgboost'"
**원인:** XGBoost가 설치되지 않음  
**해결:**
```bash
pip install xgboost
```

또는 해당 모델을 제외:
```python
# 섹션 6에서 XGBoost 부분 주석 처리 또는 삭제
```

### 2. 메모리 부족 에러
**원인:** 데이터셋이 너무 큼  
**해결:**
- 모델 수를 줄이기
- n_estimators를 줄이기 (예: 150 → 100)
- 샘플링으로 데이터 크기 줄이기

### 3. 학습이 너무 느림
**원인:** 복잡한 모델 + 큰 데이터  
**해결:**
- n_jobs=-1 사용 (병렬 처리)
- GPU 사용
- 간단한 모델 먼저 테스트

### 4. "KeyError: 'Revenue'"
**원인:** 타깃 컬럼명이 다름  
**해결:**
```python
# 실제 타깃 컬럼명으로 수정
y_train = train_df['YOUR_TARGET_COLUMN'].astype(int)
```

---

## 📊 성능 해석 가이드

### Accuracy (정확도)
- 전체 예측 중 맞춘 비율
- **불균형 데이터에서는 주의!**

### Precision (정밀도)
- 구매 예측 중 실제 구매 비율
- **거짓 긍정(False Positive)을 줄이고 싶을 때**

### Recall (재현율)
- 실제 구매 중 맞춘 비율
- **실제 구매자를 놓치지 않고 싶을 때**

### F1-Score
- Precision과 Recall의 조화 평균
- **균형잡힌 성능 평가**

### ROC-AUC
- 모든 threshold에서의 성능
- **0.5: 랜덤, 1.0: 완벽**
- **일반적으로 가장 중요한 메트릭**

---

## 🎓 다음 단계

### 레벨 1: 기본 실행
✅ 노트북 전체 실행  
✅ 결과 파일 확인  
✅ 최고 성능 모델 파악

### 레벨 2: 최적화
✅ 하이퍼파라미터 튜닝 (GridSearchCV, RandomizedSearchCV)  
✅ 교차 검증 (Cross-Validation)  
✅ 앙상블 기법 (Voting, Stacking)

### 레벨 3: 심화
✅ SHAP 분석 (모델 설명)  
✅ 불균형 처리 심화 (SMOTE, ADASYN)  
✅ 자동 ML (AutoML)

---

## 📞 추가 도움말

**모델별 특징:**

| 모델 | 장점 | 단점 | 추천 상황 |
|------|------|------|-----------|
| **Logistic Regression** | 빠름, 해석 쉬움 | 비선형 관계 약함 | 빠른 베이스라인 |
| **Decision Tree** | 해석 쉬움 | 과적합 쉬움 | 규칙 기반 설명 필요 |
| **Random Forest** | 안정적, 과적합 적음 | 느림, 메모리 많이 사용 | 균형잡힌 성능 |
| **Gradient Boosting** | 높은 성능 | 느림, 튜닝 어려움 | 최고 성능 필요 |
| **XGBoost** | 매우 높은 성능, 빠름 | 설치 필요 | 경진대회, 실전 |
| **LightGBM** | 빠르고 메모리 효율적 | 작은 데이터에 약함 | 큰 데이터셋 |
| **CatBoost** | 범주형 자동 처리 | 느림 | 범주형 변수 많을 때 |

---

**준비 완료! 노트북을 실행하세요! 🚀**
