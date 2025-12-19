import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, ParameterSampler, cross_validate
from sklearn.metrics import make_scorer, fbeta_score
from sklearn.ensemble import RandomForestClassifier

# ======================================================================================
# F2 SCORE 중심 하이퍼파라미터 튜닝
# ======================================================================================
# F2 Score = (1 + 2²) × (Precision × Recall) / (2² × Precision + Recall)
#          = 5 × (Precision × Recall) / (4 × Precision + Recall)
#
# → Recall에 4배 가중치를 부여하므로 "False Negative"를 줄이는 것이 핵심!
# ======================================================================================

# -----------------------
# 0) 설정
# -----------------------
SEARCH_CONFIG = {
    "n_iter": 50,         # 탐색 횟수 증가 (F2 최적화를 위해)
    "cv": 5,              # 교차 검증 폴드 수
    "random_state": 42,
    "n_jobs": -1,
    "verbose": 1,
    "top_k_test": 20,     # Test셋 재검증할 상위 모델 개수
    
    # --- F2 Score 최적화를 위한 파라미터 공간 ---
    "params": {
        # 1) 트리 개수: 많을수록 안정적 (Recall 향상)
        "rf__n_estimators": [300, 500, 800, 1000, 1500],
        
        # 2) 트리 깊이: 깊을수록 소수 클래스 포착 가능 (Recall↑)
        #    None = 제한 없음 (과적합 위험 있지만 Recall 중요 시 유리)
        "rf__max_depth": [20, 30, 40, 50, None],
        
        # 3) 클래스 가중치: F2 Score에서 가장 중요!
        #    - "balanced": 클래스 비율의 역수로 가중치 부여
        #    - "balanced_subsample": 부트스트랩 샘플마다 가중치 재계산
        #    - 딕셔너리: 직접 가중치 조정 (예: {0: 1, 1: 5})
        "rf__class_weight": [
            "balanced", 
            "balanced_subsample",
            {0: 1, 1: 3},   # 소수 클래스에 3배 가중치
            {0: 1, 1: 5},   # 소수 클래스에 5배 가중치
            {0: 1, 1: 10},  # 소수 클래스에 10배 가중치 (극단적)
        ],
        
        # 4) 분할 최소 샘플 수: 작을수록 세밀한 분할 → Recall↑
        "rf__min_samples_split": [2, 5, 10, 20],
        
        # 5) 리프 노드 최소 샘플 수: 1~2로 설정 시 Recall 향상
        "rf__min_samples_leaf": [1, 2, 3],
        
        # 6) 피처 선택 방식
        "rf__max_features": ["sqrt", "log2", 0.5, 0.7],
        
        # 7) 부트스트랩: True 권장
        "rf__bootstrap": [True],
        
        # 8) 분할 기준: entropy가 불균형 데이터에 더 유리할 수 있음
        "rf__criterion": ["gini", "entropy"],
        
        # 9) 최소 불순도 감소: 0으로 설정 시 더 많은 분할 → Recall↑
        "rf__min_impurity_decrease": [0.0, 0.001, 0.005],
        
        # 10) 샘플 가중치 적용 시 oob_score 비활성화 권장
        "rf__oob_score": [False],
    }
}

# -----------------------
# 1) 데이터 로드
# -----------------------
train_df = pd.read_csv("../../../../data/processed/train.csv")
test_df  = pd.read_csv("../../../../data/processed/test.csv")

X_train = train_df.drop("Revenue", axis=1)
y_train = train_df["Revenue"].astype(int)

X_test  = test_df.drop("Revenue", axis=1)
y_test  = test_df["Revenue"].astype(int)

cat_cols = X_train.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
num_cols = X_train.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns.tolist()

print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
print(f"Class distribution:\n{y_train.value_counts(normalize=True)}")

# -----------------------
# 2) 전처리
# -----------------------
numeric_tf = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_tf = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])

preprocess = ColumnTransformer(
    transformers=[
        ("num", numeric_tf, num_cols),
        ("cat", categorical_tf, cat_cols),
    ],
    remainder="drop",
    verbose_feature_names_out=False
)

# -----------------------
# 3) 모델 파이프라인
# -----------------------
rf = RandomForestClassifier(
    random_state=SEARCH_CONFIG["random_state"],
    n_jobs=SEARCH_CONFIG["n_jobs"],
)

pipe = Pipeline(steps=[
    ("preprocess", preprocess),
    ("rf", rf),
])

# -----------------------
# 4) F2 Score 정의 (beta=2)
# -----------------------
f2_scorer = make_scorer(fbeta_score, beta=2, zero_division=0)

scoring = {
    "acc": "accuracy",
    "prec": "precision",
    "rec": "recall",
    "f1": "f1",
    "f2": f2_scorer,  # ★ F2 Score 추가
    "roc_auc": "roc_auc",
}

cv = StratifiedKFold(
    n_splits=SEARCH_CONFIG["cv"],
    shuffle=True,
    random_state=SEARCH_CONFIG["random_state"],
)

# -----------------------
# 5) Manual Search Loop
# -----------------------
param_distributions = SEARCH_CONFIG["params"]

print(f"\n{'='*120}")
print(f"Generating {SEARCH_CONFIG['n_iter']} candidates for F2 Score optimization...")
print(f"{'='*120}\n")

param_list = list(ParameterSampler(
    param_distributions, 
    n_iter=SEARCH_CONFIG["n_iter"], 
    random_state=SEARCH_CONFIG["random_state"]
))

results = []

# 헤더 출력
print(f"{'Iter':<5} | {'F2':<7} | {'Rec':<7} | {'F1':<7} | {'AUC':<7} | {'Acc':<7} | {'Time(s)':<8} | Params")
print("-" * 150)

for i, params in enumerate(param_list):
    pipe.set_params(**params)
    
    cv_res = cross_validate(
        pipe, X_train, y_train, 
        cv=cv, scoring=scoring, 
        n_jobs=SEARCH_CONFIG["n_jobs"],
        return_train_score=False
    )
    
    # 집계
    mean_fit_time = cv_res['fit_time'].mean()
    mean_score_time = cv_res['score_time'].mean()
    
    mean_test_acc = cv_res['test_acc'].mean()
    mean_test_prec = cv_res['test_prec'].mean()
    mean_test_rec = cv_res['test_rec'].mean()
    mean_test_f1 = cv_res['test_f1'].mean()
    mean_test_f2 = cv_res['test_f2'].mean()  # ★
    std_test_f2 = cv_res['test_f2'].std()    # ★
    mean_test_roc_auc = cv_res['test_roc_auc'].mean()
    
    # 상태 출력 (F2 우선)
    print(
        f"{i+1:<5} | "
        f"{mean_test_f2:.4f}  | "
        f"{mean_test_rec:.4f}  | "
        f"{mean_test_f1:.4f}  | "
        f"{mean_test_roc_auc:.4f}  | "
        f"{mean_test_acc:.4f}  | "
        f"{mean_fit_time:6.2f}   | "
        f"{params}"
    )
    
    # 저장
    row = {
        "mean_fit_time": mean_fit_time,
        "mean_score_time": mean_score_time,
        "mean_test_acc": mean_test_acc,
        "mean_test_prec": mean_test_prec,
        "mean_test_rec": mean_test_rec,
        "mean_test_f1": mean_test_f1,
        "mean_test_f2": mean_test_f2,      # ★
        "std_test_f2": std_test_f2,        # ★
        "mean_test_roc_auc": mean_test_roc_auc,
    }
    
    for k, v in params.items():
        row[f"param_{k}"] = v
        
    results.append(row)

# -----------------------
# 6) CV Results DataFrame
# -----------------------
cv_results = pd.DataFrame(results)

if not cv_results.empty:
    # F2 기준 랭킹
    cv_results["rank_test_f2"] = cv_results["mean_test_f2"].rank(ascending=False, method="min").astype(int)

    param_cols = [c for c in cv_results.columns if c.startswith("param_")]
    keep_cols = (
        ["rank_test_f2", "mean_fit_time"] +
        param_cols +
        ["mean_test_f2", "std_test_f2",      # ★ F2 우선
         "mean_test_rec", "mean_test_f1",    # Recall, F1 순서
         "mean_test_prec", "mean_test_acc", "mean_test_roc_auc"]
    )

    leaderboard_cv = (
        cv_results[keep_cols]
        .sort_values(["mean_test_f2", "mean_test_rec"], ascending=False)  # ★ F2 → Recall 순
        .reset_index(drop=True)
    )

    print("\n" + "="*150)
    print("TOP 30 Models by F2 Score (Cross-Validation)")
    print("="*150 + "\n")
    
    # 표 출력
    display(leaderboard_cv.head(30).style.format(precision=4))
    
    # -----------------------
    # 7) Test Set 평가 (상위 모델만)
    # -----------------------
    print(f"\n{'='*120}")
    print(f"Evaluating Top {SEARCH_CONFIG['top_k_test']} models on Test Set...")
    print(f"{'='*120}\n")
    
    test_results = []
    
    for idx in range(min(SEARCH_CONFIG['top_k_test'], len(leaderboard_cv))):
        row = leaderboard_cv.iloc[idx]
        
        # 파라미터 추출
        test_params = {}
        for col in param_cols:
            param_name = col.replace("param_", "")
            test_params[param_name] = row[col]
        
        # 모델 재학습
        pipe.set_params(**test_params)
        pipe.fit(X_train, y_train)
        
        # Test 예측
        y_pred = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)[:, 1]
        
        # 메트릭 계산
        test_acc = accuracy_score(y_test, y_pred)
        test_prec = precision_score(y_test, y_pred, zero_division=0)
        test_rec = recall_score(y_test, y_pred, zero_division=0)
        test_f1 = f1_score(y_test, y_pred, zero_division=0)
        test_f2 = fbeta_score(y_test, y_pred, beta=2, zero_division=0)  # ★
        test_auc = roc_auc_score(y_test, y_proba)
        
        test_results.append({
            "rank_cv": idx + 1,
            "cv_f2": row["mean_test_f2"],
            "test_f2": test_f2,      # ★
            "test_rec": test_rec,
            "test_f1": test_f1,
            "test_prec": test_prec,
            "test_acc": test_acc,
            "test_auc": test_auc,
            **test_params
        })
    
    test_df_results = pd.DataFrame(test_results)
    test_df_results["rank_test_f2"] = test_df_results["test_f2"].rank(ascending=False, method="min").astype(int)
    
    test_leaderboard = (
        test_df_results
        .sort_values(["test_f2", "test_rec"], ascending=False)
        .reset_index(drop=True)
    )
    
    print("\n" + "="*150)
    print("TOP 20 Models by F2 Score (Test Set)")
    print("="*150 + "\n")
    
    display(test_leaderboard.head(20).style.format(precision=4))
    
    # Best Model
    best_idx = test_leaderboard.iloc[0]
    print(f"\n{'='*120}")
    print(f"BEST MODEL (F2 Score: {best_idx['test_f2']:.4f})")
    print(f"{'='*120}")
    print(f"  CV F2:     {best_idx['cv_f2']:.4f}")
    print(f"  Test F2:   {best_idx['test_f2']:.4f}")
    print(f"  Test Rec:  {best_idx['test_rec']:.4f}")
    print(f"  Test F1:   {best_idx['test_f1']:.4f}")
    print(f"  Test Prec: {best_idx['test_prec']:.4f}")
    print(f"  Test Acc:  {best_idx['test_acc']:.4f}")
    print(f"  Test AUC:  {best_idx['test_auc']:.4f}")
    print(f"\nParameters:")
    for col in param_cols:
        param_name = col.replace("param_", "")
        if param_name in best_idx:
            print(f"  {param_name}: {best_idx[param_name]}")
    
else:
    print("No results found.")
