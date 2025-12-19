# F2 Score 최적화 실행 예제

## 빠른 시작

### 1. 기본 실행
```python
%run f2_optimization.py
```

### 2. 설정 커스터마이징
```python
# f2_optimization.py에서 SEARCH_CONFIG 수정

# 빠른 테스트 (5분 내)
SEARCH_CONFIG = {
    "n_iter": 10,
    "cv": 3,
    "params": {
        "rf__n_estimators": [300, 500],
        "rf__class_weight": ["balanced", {0: 1, 1: 5}],
        "rf__max_depth": [30, None],
    }
}

# 본격 탐색 (30분~1시간)
SEARCH_CONFIG = {
    "n_iter": 50,
    "cv": 5,
    # ... (기본 설정 사용)
}

# 최종 모델 (2~3시간)
SEARCH_CONFIG = {
    "n_iter": 100,
    "cv": 10,
    "params": {
        "rf__n_estimators": [1000, 1500, 2000],
        # ... 세밀한 파라미터 그리드
    }
}
```

---

## 결과 해석 가이드

### CV Leaderboard 예시
```
rank_test_f2 | mean_test_f2 | mean_test_rec | mean_test_f1 | param_rf__class_weight | param_rf__max_depth
-------------|--------------|---------------|--------------|------------------------|--------------------
1            | 0.7234       | 0.8156        | 0.6892       | {0: 1, 1: 7}          | None
2            | 0.7189       | 0.7998        | 0.6845       | balanced_subsample     | 40
3            | 0.7156       | 0.7856        | 0.6823       | {0: 1, 1: 5}          | 50
```

**해석**:
- 1위 모델: F2=0.7234, Recall=0.8156
  - 소수 클래스 7배 가중치 + 무제한 깊이
  - Recall 우선 전략 → False Negative 최소화
  - 단, 과적합 위험 → Test Set 검증 필수

---

### Test Leaderboard 예시
```
rank_test_f2 | cv_f2  | test_f2 | test_rec | test_f1 | test_prec | test_acc
-------------|--------|---------|----------|---------|-----------|----------
1            | 0.7234 | 0.7102  | 0.7934   | 0.6756  | 0.5892    | 0.8234
2            | 0.7189 | 0.7098  | 0.7856   | 0.6734  | 0.5967    | 0.8312
```

**분석**:
- CV-Test 차이: 0.7234 - 0.7102 = **0.0132** (작음 → 일반화 좋음)
- Test Recall: 0.7934 (높음 → False Negative 적음)
- Test Precision: 0.5892 (보통 → False Positive 있음)

**결론**: 
- F2 목표 달성 (Recall 우수)
- Precision 희생은 F2 전략상 정상
- 만약 Precision이 너무 낮으면 (< 0.3) class_weight 감소 고려

---

## 실전 시나리오별 설정

### 시나리오 1: 불균형 비율 1:10 (Revenue=1이 10%)
```python
"rf__class_weight": [
    {0: 1, 1: 3},    # 보수적
    {0: 1, 1: 5},    # 중간 (추천)
    {0: 1, 1: 7},    # 공격적
]
```

### 시나리오 2: 불균형 비율 1:50 (Revenue=1이 2%)
```python
"rf__class_weight": [
    {0: 1, 1: 10},   # 보수적
    {0: 1, 1: 20},   # 중간 (추천)
    {0: 1, 1: 50},   # 공격적
]
```

### 시나리오 3: 극단적 불균형 1:100+
```python
"rf__class_weight": [
    {0: 1, 1: 50},
    {0: 1, 1: 100},
]
# + SMOTE 오버샘플링 고려
```

---

## 체크리스트

### 실행 전
- [ ] 데이터 경로 확인 (`train.csv`, `test.csv`)
- [ ] 클래스 비율 확인 (`y_train.value_counts()`)
- [ ] 메모리 충분한지 확인 (n_estimators × max_depth 큼 → 메모리 증가)

### 실행 중
- [ ] F2 Score 추이 모니터링 (증가하는지 확인)
- [ ] Recall vs Precision 균형 확인
- [ ] 학습 시간 확인 (너무 오래 걸리면 n_iter 감소)

### 실행 후
- [ ] CV-Test 차이 확인 (< 0.05 권장)
- [ ] Test Recall 확인 (0.7+ 목표)
- [ ] Confusion Matrix 확인 (FN 개수)
- [ ] 특성 중요도 확인 (어떤 피처가 중요한지)

---

## 추가 분석 코드

### Confusion Matrix
```python
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# 최적 모델 학습
best_params = {...}  # 위 결과에서 복사
pipe.set_params(**best_params)
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# FN 개수 확인
fn_count = cm[1, 0]  # 실제 1인데 0으로 예측
print(f"False Negatives: {fn_count}")
```

### Feature Importance
```python
# 전처리 후 피처명 추출
feature_names = (
    num_cols + 
    list(pipe.named_steps['preprocess']
         .named_transformers_['cat']
         .named_steps['ohe']
         .get_feature_names_out(cat_cols))
)

# 중요도 추출
importances = pipe.named_steps['rf'].feature_importances_
importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': importances
}).sort_values('importance', ascending=False)

# 상위 20개 시각화
importance_df.head(20).plot(x='feature', y='importance', kind='barh', figsize=(10, 8))
plt.title('Top 20 Feature Importances')
plt.show()
```

### Threshold 최적화
```python
from sklearn.metrics import precision_recall_curve
import numpy as np

y_proba = pipe.predict_proba(X_test)[:, 1]

# F2 기준 최적 threshold 찾기
precision, recall, thresholds = precision_recall_curve(y_test, y_proba)

# F2 계산 (beta=2)
f2_scores = np.nan_to_num(
    (5 * precision * recall) / (4 * precision + recall)
)

best_idx = np.argmax(f2_scores)
best_threshold = thresholds[best_idx]
best_f2 = f2_scores[best_idx]

print(f"Optimal Threshold: {best_threshold:.4f}")
print(f"F2 Score at Optimal: {best_f2:.4f}")

# 최적 threshold로 재예측
y_pred_optimized = (y_proba >= best_threshold).astype(int)

# 비교
from sklearn.metrics import classification_report
print("\n=== 기본 Threshold (0.5) ===")
print(classification_report(y_test, y_pred))

print("\n=== 최적 Threshold ===")
print(classification_report(y_test, y_pred_optimized))
```

---

## 문제 해결

### Q1: "MemoryError" 발생
**해결책**:
```python
# n_estimators 감소
"rf__n_estimators": [300, 500]  # 1500 → 500

# 또는 n_jobs 감소
"n_jobs": 4  # -1 → 4
```

### Q2: 너무 오래 걸림
**해결책**:
```python
# n_iter 감소
"n_iter": 20  # 50 → 20

# cv 감소
"cv": 3  # 5 → 3

# 파라미터 공간 축소
"rf__n_estimators": [500],
"rf__max_depth": [30, None],
```

### Q3: CV-Test 차이가 큼 (과적합)
**해결책**:
```python
# max_depth 제한
"rf__max_depth": [20, 30]  # None 제거

# min_samples_leaf 증가
"rf__min_samples_leaf": [3, 5, 10]

# min_impurity_decrease 증가
"rf__min_impurity_decrease": [0.001, 0.01]
```

### Q4: Precision이 너무 낮음 (< 0.3)
**해결책**:
```python
# class_weight 감소
"rf__class_weight": [
    {0: 1, 1: 3},  # 10 → 3
    {0: 1, 1: 5},
]
```

---

## 최종 모델 저장

```python
import joblib

# 최적 모델 학습
best_params = {...}  # 결과에서 복사
pipe.set_params(**best_params)
pipe.fit(X_train, y_train)

# 저장
joblib.dump(pipe, 'best_f2_model.pkl')

# 로드
loaded_model = joblib.load('best_f2_model.pkl')
y_pred = loaded_model.predict(X_test)
```

---

## 성능 벤치마크

| 모델 | F2 Score | Recall | Precision | 비고 |
|------|----------|--------|-----------|------|
| Baseline (default RF) | 0.45 | 0.52 | 0.48 | class_weight=None |
| F1 최적화 | 0.62 | 0.68 | 0.65 | class_weight="balanced" |
| **F2 최적화** | **0.71** | **0.79** | **0.59** | **class_weight={0:1, 1:7}** |
| 극단 Recall | 0.68 | 0.85 | 0.42 | class_weight={0:1, 1:20} |

**권장**: F2 최적화 모델 (Recall과 Precision 균형)
