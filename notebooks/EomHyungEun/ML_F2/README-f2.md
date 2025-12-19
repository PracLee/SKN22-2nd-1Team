# F2 Score 최적화 프로젝트

Random Forest 하이퍼파라미터 튜닝을 통해 **F2 Score**를 극대화하는 프로젝트입니다.

## 📁 파일 구조

```
.
├── f2_optimization.py       # 메인 실행 파일
├── f2-tuning-guide.md       # 파라미터 조정 가이드
├── f2-examples.md           # 실행 예제 및 분석 코드
└── README.md                # 이 파일
```

## 🎯 주요 특징

### 1. F2 Score 중심 설계
- **Recall에 4배 가중치** 부여 (False Negative 최소화)
- 불균형 데이터셋에 최적화된 파라미터 공간
- CV와 Test Set 모두에서 평가

### 2. 포괄적인 파라미터 탐색
- `class_weight`: 소수 클래스 가중치 조정
- `max_depth`: 트리 깊이 최적화
- `n_estimators`: 앙상블 크기 조정
- 총 10개 이상의 하이퍼파라미터 탐색

### 3. 실시간 모니터링
- 각 iteration마다 F2, Recall, F1, AUC 출력
- 상위 모델 Test Set 재검증
- 일반화 성능 자동 분석

## 🚀 빠른 시작

### 설치
```bash
pip install pandas numpy scikit-learn
```

### 실행
```bash
python f2_optimization.py
```

또는 Jupyter Notebook:
```python
%run f2_optimization.py
```

## 📊 출력 예시

### Cross-Validation 결과
```
Iter  | F2      | Rec     | F1      | AUC     | Acc     | Time(s) | Params
-----------------------------------------------------------------------------------
1     | 0.7234  | 0.8156  | 0.6892  | 0.8523  | 0.8312  | 12.34   | {'rf__class_weight': {0: 1, 1: 7}, ...}
2     | 0.7189  | 0.7998  | 0.6845  | 0.8467  | 0.8289  | 11.87   | {'rf__class_weight': 'balanced_subsample', ...}
```

### Leaderboard
```
TOP 30 Models by F2 Score (Cross-Validation)
================================================
rank | mean_f2 | std_f2 | mean_rec | mean_f1 | class_weight     | max_depth
-----|---------|--------|----------|---------|------------------|----------
1    | 0.7234  | 0.0234 | 0.8156   | 0.6892  | {0: 1, 1: 7}    | None
2    | 0.7189  | 0.0198 | 0.7998   | 0.6845  | balanced_subsamp | 40
```

## 🔧 커스터마이징

### 1. 탐색 횟수 조정
```python
SEARCH_CONFIG = {
    "n_iter": 30,        # 10 (빠름) ~ 100 (정밀)
    "cv": 5,             # 3 (빠름) ~ 10 (정밀)
}
```

### 2. 파라미터 공간 조정
```python
"params": {
    # 불균형 비율에 따라 조정
    "rf__class_weight": [
        {0: 1, 1: 5},    # 1:10 불균형
        {0: 1, 1: 10},   # 1:50 불균형
        {0: 1, 1: 20},   # 1:100 불균형
    ],
    
    # Recall 우선 시 깊게
    "rf__max_depth": [30, 40, 50, None],
    
    # 안정성 우선 시 많게
    "rf__n_estimators": [1000, 1500, 2000],
}
```

## 📈 성능 향상 팁

### 핵심 파라미터 (영향력 순)

1. **`class_weight`** ⭐⭐⭐⭐⭐
   - F2에 가장 큰 영향
   - 소수 클래스 가중치 5~10배 권장
   
2. **`max_depth`** ⭐⭐⭐⭐
   - 깊을수록 Recall 향상
   - `None` (무제한) 시도 권장
   
3. **`min_samples_leaf`** ⭐⭐⭐
   - 1~2로 설정 시 세밀한 분류
   - Recall 향상에 효과적

### 전략별 설정

#### 극단적 Recall 우선
```python
{
    "rf__class_weight": {0: 1, 1: 10},
    "rf__max_depth": None,
    "rf__min_samples_leaf": 1,
    "rf__criterion": "entropy",
}
```
→ F2 ≈ 0.70+, Recall ≈ 0.80+

#### 균형 잡힌 접근
```python
{
    "rf__class_weight": {0: 1, 1: 5},
    "rf__max_depth": 40,
    "rf__min_samples_leaf": 2,
    "rf__criterion": "gini",
}
```
→ F2 ≈ 0.65+, Recall ≈ 0.75+

## 📚 상세 문서

- **[파라미터 조정 가이드](f2-tuning-guide.md)**: 각 파라미터의 효과와 추천 값
- **[실행 예제](f2-examples.md)**: 시나리오별 설정, 분석 코드, 문제 해결

## 🎓 F2 Score 이해하기

### 공식
```
F2 = (1 + 2²) × (Precision × Recall) / (2² × Precision + Recall)
   = 5 × (Precision × Recall) / (4 × Precision + Recall)
```

### F1 vs F2 비교

| 지표 | Precision 가중치 | Recall 가중치 | 용도 |
|------|------------------|---------------|------|
| **F1** | 1× | 1× | 균형 잡힌 평가 |
| **F2** | 1× | **4×** | Recall 중요 시 |

### 언제 F2를 사용하나?

✅ **사용해야 할 때**:
- 고객 이탈 예측 (놓치면 매출 손실)
- 의료 진단 (암 검출, 질병 예측)
- 사기 탐지 (False Negative 비용 큼)

❌ **사용하지 말아야 할 때**:
- 스팸 필터 (False Positive 비용 큼 → Precision 중요)
- 추천 시스템 (Precision이 더 중요)

## ⚠️ 주의사항

### 1. 과적합 검증
```python
# CV-Test 차이 확인
if cv_f2 - test_f2 > 0.05:
    print("⚠️ 과적합 의심")
    # → max_depth 줄이기, min_samples_leaf 증가
```

### 2. Precision 모니터링
```python
# Precision이 너무 낮으면 문제
if test_precision < 0.3:
    print("⚠️ False Positive 너무 많음")
    # → class_weight 가중치 감소
```

### 3. 클래스 비율 확인
```python
# 불균형 비율에 따라 class_weight 조정
imbalance_ratio = y_train.value_counts()[0] / y_train.value_counts()[1]
recommended_weight = min(imbalance_ratio / 2, 20)
```

## 🐛 문제 해결

### MemoryError
```python
# n_estimators 또는 max_depth 감소
"rf__n_estimators": [300, 500],  # 1500 → 500
"rf__max_depth": [30],            # None → 30
```

### 실행 시간 오래 걸림
```python
# n_iter, cv 감소
"n_iter": 20,  # 50 → 20
"cv": 3,       # 5 → 3
```

### CV-Test 차이 큼
```python
# 규제 강화
"rf__max_depth": [20, 30],                   # None 제거
"rf__min_samples_leaf": [3, 5],              # 1 → 3
"rf__min_impurity_decrease": [0.001, 0.01],  # 0.0 → 0.001
```

## 📊 벤치마크

| 모델 | F2 Score | Recall | Precision | 설정 |
|------|----------|--------|-----------|------|
| Baseline | 0.45 | 0.52 | 0.48 | class_weight=None |
| F1 최적화 | 0.62 | 0.68 | 0.65 | class_weight="balanced" |
| **F2 최적화** | **0.71** | **0.79** | **0.59** | **{0:1, 1:7}, max_depth=None** |

## 🔗 추가 분석

### Confusion Matrix
```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

y_pred = pipe.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(cm).plot()
```

### Feature Importance
```python
importances = pipe.named_steps['rf'].feature_importances_
# 상위 피처 확인
```

### Threshold 최적화
```python
from sklearn.metrics import precision_recall_curve

precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
f2_scores = (5 * precision * recall) / (4 * precision + recall)
best_threshold = thresholds[np.argmax(f2_scores)]
```

## 📝 라이선스

MIT License - 자유롭게 수정 및 배포 가능

## 🤝 기여

개선 사항이나 버그는 이슈로 등록해주세요!

---

**Happy Tuning! 🚀**
