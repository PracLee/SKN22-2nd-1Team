
# ============================================================================
# Online Shoppers Intent Prediction - 모델 학습 & 비교
# Jupyter Notebook용
# ============================================================================

# %% [markdown]
# # 🛒 Online Shoppers Intent - 모델 학습 파이프라인
# 
# 이 노트북은 다양한 머신러닝 모델을 학습하고 비교합니다.
# 
# **목표:**
# - 여러 모델 (LR, RF, GB, XGB, LightGBM, CatBoost) 학습
# - 성능 비교 (Accuracy, Precision, Recall, F1, ROC-AUC)
# - 최적 모델 선택
# - Feature Importance 분석

# %% [markdown]
# ## 1. 라이브러리 Import

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, auc
)
import warnings
warnings.filterwarnings('ignore')

# 모델들
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# 고급 모델 (선택사항)
try:
    import xgboost as xgb
    HAS_XGB = True
except:
    HAS_XGB = False
    print("⚠️ XGBoost not installed")

try:
    import lightgbm as lgb
    HAS_LGB = True
except:
    HAS_LGB = False
    print("⚠️ LightGBM not installed")

try:
    import catboost as cb
    HAS_CB = True
except:
    HAS_CB = False
    print("⚠️ CatBoost not installed")

# 설정
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("✅ 라이브러리 로드 완료")

# %% [markdown]
# ## 2. 데이터 로드

# %%
# 데이터 로드
train_df = pd.read_csv("../../data/processed/train.csv")
test_df = pd.read_csv("../../data/processed/test.csv")

print("=" * 80)
print("데이터 로드 완료")
print("=" * 80)
print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")
print(f"\nTrain columns: {train_df.columns.tolist()}")

# 기본 정보
print(f"\n✅ Train set 구매율: {train_df['Revenue'].mean():.2%}")
print(f"✅ Test set 구매율: {test_df['Revenue'].mean():.2%}")

# %% [markdown]
# ## 3. 데이터 전처리

# %%
# 타깃과 피처 분리
X_train = train_df.drop('Revenue', axis=1)
y_train = train_df['Revenue'].astype(int)

X_test = test_df.drop('Revenue', axis=1)
y_test = test_df['Revenue'].astype(int)

print("=" * 80)
print("데이터 전처리")
print("=" * 80)
print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_test shape: {y_test.shape}")

# 데이터 타입 확인
print(f"\n📊 피처 타입:")
print(X_train.dtypes.value_counts())

# 범주형 변수 확인
categorical_cols = X_train.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
numerical_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()

print(f"\n✅ 범주형 변수 ({len(categorical_cols)}개): {categorical_cols}")
print(f"✅ 수치형 변수 ({len(numerical_cols)}개): {numerical_cols}")

# %% [markdown]
# ## 4. 범주형 변수 인코딩 (필요시)

# %%
# 범주형 변수가 있다면 인코딩
if len(categorical_cols) > 0:
    print("=" * 80)
    print("범주형 변수 인코딩")
    print("=" * 80)

    label_encoders = {}

    for col in categorical_cols:
        if col in X_train.columns:
            le = LabelEncoder()

            # Train 데이터로 fit
            X_train[col] = le.fit_transform(X_train[col].astype(str))

            # Test 데이터에 적용 (새로운 값이 있을 수 있으므로 처리)
            X_test[col] = X_test[col].astype(str).apply(
                lambda x: le.transform([x])[0] if x in le.classes_ else -1
            )

            label_encoders[col] = le
            print(f"✅ {col}: {len(le.classes_)} classes")

    print(f"\n✅ 총 {len(label_encoders)}개 변수 인코딩 완료")
else:
    print("✅ 범주형 변수 없음 - 인코딩 불필요")

# %% [markdown]
# ## 5. 스케일링 (옵션)

# %%
# 스케일링이 필요한 모델을 위해 스케일된 버전도 준비
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("✅ StandardScaler 적용 완료")
print(f"   - Scaled train shape: {X_train_scaled.shape}")
print(f"   - Scaled test shape: {X_test_scaled.shape}")

# %% [markdown]
# ## 6. 모델 정의 및 학습

# %%
# 모델 딕셔너리 정의
models = {}

# 1. Logistic Regression (스케일된 데이터 사용)
models['Logistic Regression'] = {
    'model': LogisticRegression(
        max_iter=2000,
        random_state=42,
        class_weight='balanced',
        solver='lbfgs'
    ),
    'use_scaled': True
}

# 2. Decision Tree
models['Decision Tree'] = {
    'model': DecisionTreeClassifier(
        max_depth=10,
        min_samples_split=20,
        random_state=42,
        class_weight='balanced'
    ),
    'use_scaled': False
}

# 3. Random Forest
models['Random Forest'] = {
    'model': RandomForestClassifier(
        n_estimators=150,
        max_depth=15,
        min_samples_split=10,
        random_state=42,
        class_weight='balanced',
        n_jobs=-1
    ),
    'use_scaled': False
}

# 4. Gradient Boosting
models['Gradient Boosting'] = {
    'model': GradientBoostingClassifier(
        n_estimators=150,
        learning_rate=0.1,
        max_depth=5,
        subsample=0.8,
        random_state=42
    ),
    'use_scaled': False
}

# 5. XGBoost (선택사항)
if HAS_XGB:
    models['XGBoost'] = {
        'model': xgb.XGBClassifier(
            n_estimators=150,
            learning_rate=0.1,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            scale_pos_weight=len(y_train[y_train==0]) / len(y_train[y_train==1]),
            use_label_encoder=False,
            eval_metric='logloss'
        ),
        'use_scaled': False
    }

# 6. LightGBM (선택사항)
if HAS_LGB:
    models['LightGBM'] = {
        'model': lgb.LGBMClassifier(
            n_estimators=150,
            learning_rate=0.1,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            class_weight='balanced',
            verbose=-1
        ),
        'use_scaled': False
    }

# 7. CatBoost (선택사항)
if HAS_CB:
    models['CatBoost'] = {
        'model': cb.CatBoostClassifier(
            iterations=150,
            learning_rate=0.1,
            depth=5,
            random_state=42,
            verbose=0,
            scale_pos_weight=len(y_train[y_train==0]) / len(y_train[y_train==1])
        ),
        'use_scaled': False
    }

print("=" * 80)
print(f"모델 정의 완료: {len(models)}개")
print("=" * 80)
for name in models.keys():
    print(f"  ✅ {name}")

# %% [markdown]
# ## 7. 모델 학습 & 평가

# %%
import time

results = []

print("\n" + "=" * 80)
print("모델 학습 시작")
print("=" * 80)

for name, config in models.items():
    print(f"\n{'='*80}")
    print(f"🔹 {name} 학습 중...")
    print("="*80)

    model = config['model']
    use_scaled = config['use_scaled']

    # 데이터 선택
    if use_scaled:
        X_tr = X_train_scaled
        X_te = X_test_scaled
    else:
        X_tr = X_train
        X_te = X_test

    # 학습 시간 측정
    start_time = time.time()

    try:
        # 학습
        model.fit(X_tr, y_train)

        # 예측
        y_pred_train = model.predict(X_tr)
        y_pred_test = model.predict(X_te)
        y_proba_train = model.predict_proba(X_tr)[:, 1]
        y_proba_test = model.predict_proba(X_te)[:, 1]

        # 학습 시간
        elapsed_time = time.time() - start_time

        # Train 성능
        train_acc = accuracy_score(y_train, y_pred_train)
        train_prec = precision_score(y_train, y_pred_train, zero_division=0)
        train_rec = recall_score(y_train, y_pred_train, zero_division=0)
        train_f1 = f1_score(y_train, y_pred_train, zero_division=0)
        train_auc = roc_auc_score(y_train, y_proba_train)

        # Test 성능
        test_acc = accuracy_score(y_test, y_pred_test)
        test_prec = precision_score(y_test, y_pred_test, zero_division=0)
        test_rec = recall_score(y_test, y_pred_test, zero_division=0)
        test_f1 = f1_score(y_test, y_pred_test, zero_division=0)
        test_auc = roc_auc_score(y_test, y_proba_test)

        # 결과 저장
        results.append({
            'Model': name,
            'Train_Acc': train_acc,
            'Test_Acc': test_acc,
            'Train_Prec': train_prec,
            'Test_Prec': test_prec,
            'Train_Rec': train_rec,
            'Test_Rec': test_rec,
            'Train_F1': train_f1,
            'Test_F1': test_f1,
            'Train_AUC': train_auc,
            'Test_AUC': test_auc,
            'Time(s)': elapsed_time
        })

        # 모델 저장 (딕셔너리에)
        config['trained_model'] = model
        config['y_pred_test'] = y_pred_test
        config['y_proba_test'] = y_proba_test

        # 출력
        print(f"\n📊 Train 성능:")
        print(f"   Accuracy:  {train_acc:.4f}")
        print(f"   Precision: {train_prec:.4f}")
        print(f"   Recall:    {train_rec:.4f}")
        print(f"   F1-Score:  {train_f1:.4f}")
        print(f"   ROC-AUC:   {train_auc:.4f}")

        print(f"\n📊 Test 성능:")
        print(f"   Accuracy:  {test_acc:.4f}")
        print(f"   Precision: {test_prec:.4f}")
        print(f"   Recall:    {test_rec:.4f}")
        print(f"   F1-Score:  {test_f1:.4f}")
        print(f"   ROC-AUC:   {test_auc:.4f}")

        print(f"\n⏱️  학습 시간: {elapsed_time:.2f}초")
        print("✅ 완료")

    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        continue

print("\n" + "=" * 80)
print("✨ 모든 모델 학습 완료!")
print("=" * 80)

# %% [markdown]
# ## 8. 결과 비교

# %%
# 결과 DataFrame 생성
results_df = pd.DataFrame(results)

# 포맷팅
for col in results_df.columns:
    if col not in ['Model', 'Time(s)']:
        results_df[col] = results_df[col].round(4)
    elif col == 'Time(s)':
        results_df[col] = results_df[col].round(2)

print("=" * 80)
print("모델 성능 비교표")
print("=" * 80)
print(results_df.to_string(index=False))

# CSV 저장
results_df.to_csv('model_comparison_results.csv', index=False, encoding='utf-8-sig')
print("\n✅ 결과 저장: model_comparison_results.csv")

# %% [markdown]
# ## 9. 시각화 - 성능 비교

# %%
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. Test Accuracy
ax1 = axes[0, 0]
results_df_sorted = results_df.sort_values('Test_Acc', ascending=True)
ax1.barh(results_df_sorted['Model'], results_df_sorted['Test_Acc'], color='steelblue')
ax1.set_xlabel('Accuracy', fontsize=12)
ax1.set_title('Test Accuracy 비교', fontsize=14, fontweight='bold')
ax1.set_xlim([0.8, 1.0])
for i, v in enumerate(results_df_sorted['Test_Acc']):
    ax1.text(v + 0.005, i, f'{v:.4f}', va='center', fontsize=10)

# 2. Test F1-Score
ax2 = axes[0, 1]
results_df_sorted = results_df.sort_values('Test_F1', ascending=True)
ax2.barh(results_df_sorted['Model'], results_df_sorted['Test_F1'], color='coral')
ax2.set_xlabel('F1-Score', fontsize=12)
ax2.set_title('Test F1-Score 비교', fontsize=14, fontweight='bold')
ax2.set_xlim([0.4, 0.8])
for i, v in enumerate(results_df_sorted['Test_F1']):
    ax2.text(v + 0.01, i, f'{v:.4f}', va='center', fontsize=10)

# 3. Test ROC-AUC
ax3 = axes[1, 0]
results_df_sorted = results_df.sort_values('Test_AUC', ascending=True)
ax3.barh(results_df_sorted['Model'], results_df_sorted['Test_AUC'], color='mediumseagreen')
ax3.set_xlabel('ROC-AUC', fontsize=12)
ax3.set_title('Test ROC-AUC 비교', fontsize=14, fontweight='bold')
ax3.set_xlim([0.85, 1.0])
for i, v in enumerate(results_df_sorted['Test_AUC']):
    ax3.text(v + 0.005, i, f'{v:.4f}', va='center', fontsize=10)

# 4. 학습 시간
ax4 = axes[1, 1]
results_df_sorted = results_df.sort_values('Time(s)', ascending=True)
ax4.barh(results_df_sorted['Model'], results_df_sorted['Time(s)'], color='mediumpurple')
ax4.set_xlabel('Time (seconds)', fontsize=12)
ax4.set_title('학습 시간 비교', fontsize=14, fontweight='bold')
for i, v in enumerate(results_df_sorted['Time(s)']):
    ax4.text(v + 0.1, i, f'{v:.2f}s', va='center', fontsize=10)

plt.tight_layout()
plt.savefig('model_comparison_charts.png', dpi=300, bbox_inches='tight')
plt.show()

print("✅ 차트 저장: model_comparison_charts.png")

# %% [markdown]
# ## 10. 정밀도-재현율-F1 비교

# %%
fig, ax = plt.subplots(figsize=(14, 8))

x = np.arange(len(results_df))
width = 0.25

bars1 = ax.bar(x - width, results_df['Test_Prec'], width, label='Precision', color='skyblue')
bars2 = ax.bar(x, results_df['Test_Rec'], width, label='Recall', color='lightcoral')
bars3 = ax.bar(x + width, results_df['Test_F1'], width, label='F1-Score', color='lightgreen')

ax.set_xlabel('Models', fontsize=12)
ax.set_ylabel('Score', fontsize=12)
ax.set_title('Test Set - Precision, Recall, F1-Score 비교', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(results_df['Model'], rotation=45, ha='right')
ax.legend(fontsize=11)
ax.set_ylim([0, 1])
ax.grid(axis='y', alpha=0.3)

# 값 표시
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('precision_recall_f1_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

print("✅ 차트 저장: precision_recall_f1_comparison.png")

# %% [markdown]
# ## 11. ROC Curve 비교

# %%
fig, ax = plt.subplots(figsize=(12, 10))

colors = plt.cm.tab10(np.linspace(0, 1, len(models)))

for i, (name, config) in enumerate(models.items()):
    if 'y_proba_test' in config:
        fpr, tpr, _ = roc_curve(y_test, config['y_proba_test'])
        roc_auc_val = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=colors[i], lw=2, 
                label=f'{name} (AUC = {roc_auc_val:.4f})')

ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random (AUC = 0.5000)')
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('ROC Curve 비교', fontsize=14, fontweight='bold')
ax.legend(loc='lower right', fontsize=10)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('roc_curves_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

print("✅ 차트 저장: roc_curves_comparison.png")

# %% [markdown]
# ## 12. Confusion Matrix (최고 성능 모델)

# %%
# 최고 성능 모델 찾기 (Test AUC 기준)
best_model_name = results_df.loc[results_df['Test_AUC'].idxmax(), 'Model']
best_model_config = models[best_model_name]

print("=" * 80)
print(f"최고 성능 모델 (Test ROC-AUC 기준): {best_model_name}")
print("=" * 80)

# Confusion Matrix
cm = confusion_matrix(y_test, best_model_config['y_pred_test'])

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True, 
            xticklabels=['No Purchase', 'Purchase'],
            yticklabels=['No Purchase', 'Purchase'],
            ax=ax)
ax.set_xlabel('Predicted', fontsize=12)
ax.set_ylabel('Actual', fontsize=12)
ax.set_title(f'Confusion Matrix - {best_model_name}', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig(f'confusion_matrix_{best_model_name.replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"✅ Confusion Matrix 저장: confusion_matrix_{best_model_name.replace(' ', '_')}.png")

# Classification Report
print(f"\n📊 {best_model_name} - Classification Report:")
print(classification_report(y_test, best_model_config['y_pred_test'], 
                          target_names=['No Purchase', 'Purchase']))

# %% [markdown]
# ## 13. Feature Importance (트리 기반 모델)

# %%
# Feature Importance 추출 가능한 모델들
tree_based_models = ['Random Forest', 'Gradient Boosting', 'XGBoost', 'LightGBM', 'CatBoost', 'Decision Tree']

for name in tree_based_models:
    if name in models and 'trained_model' in models[name]:
        model = models[name]['trained_model']

        # Feature Importance 추출
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            feature_names = X_train.columns

            # DataFrame 생성
            fi_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importances
            }).sort_values('Importance', ascending=False)

            # Top 15 피처
            top_n = 15
            fi_top = fi_df.head(top_n)

            # 시각화
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.barh(fi_top['Feature'], fi_top['Importance'], color='teal')
            ax.set_xlabel('Importance', fontsize=12)
            ax.set_title(f'{name} - Feature Importance (Top {top_n})', 
                        fontsize=14, fontweight='bold')
            ax.invert_yaxis()

            for i, v in enumerate(fi_top['Importance']):
                ax.text(v + 0.001, i, f'{v:.4f}', va='center', fontsize=9)

            plt.tight_layout()
            plt.savefig(f'feature_importance_{name.replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
            plt.show()

            print(f"✅ Feature Importance 차트 저장: feature_importance_{name.replace(' ', '_')}.png")

            # CSV 저장
            fi_df.to_csv(f'feature_importance_{name.replace(" ", "_")}.csv', 
                        index=False, encoding='utf-8-sig')
            print(f"✅ Feature Importance 데이터 저장: feature_importance_{name.replace(' ', '_')}.csv\n")

# %% [markdown]
# ## 14. 최종 결과 요약

# %%
print("\n" + "=" * 80)
print("최종 결과 요약")
print("=" * 80)

# 각 메트릭별 최고 모델
best_by_metric = {
    'Test Accuracy': results_df.loc[results_df['Test_Acc'].idxmax()],
    'Test Precision': results_df.loc[results_df['Test_Prec'].idxmax()],
    'Test Recall': results_df.loc[results_df['Test_Rec'].idxmax()],
    'Test F1-Score': results_df.loc[results_df['Test_F1'].idxmax()],
    'Test ROC-AUC': results_df.loc[results_df['Test_AUC'].idxmax()],
    'Fastest Training': results_df.loc[results_df['Time(s)'].idxmin()]
}

for metric, row in best_by_metric.items():
    print(f"\n🏆 {metric} 최고:")
    print(f"   모델: {row['Model']}")
    if 'Time' in metric:
        print(f"   값: {row['Time(s)']:.2f}초")
    else:
        metric_col = metric.replace(' ', '_')
        print(f"   값: {row[metric_col]:.4f}")

# 종합 평가
print("\n" + "=" * 80)
print("💡 추천 모델")
print("=" * 80)

best_overall = results_df.loc[results_df['Test_AUC'].idxmax()]
print(f"\n✨ 종합 최고 성능 (ROC-AUC 기준): {best_overall['Model']}")
print(f"   - Test Accuracy:  {best_overall['Test_Acc']:.4f}")
print(f"   - Test Precision: {best_overall['Test_Prec']:.4f}")
print(f"   - Test Recall:    {best_overall['Test_Rec']:.4f}")
print(f"   - Test F1-Score:  {best_overall['Test_F1']:.4f}")
print(f"   - Test ROC-AUC:   {best_overall['Test_AUC']:.4f}")
print(f"   - 학습 시간:      {best_overall['Time(s)']:.2f}초")

# %% [markdown]
# ## 15. 모델 저장 (선택사항)

# %%
import pickle

# 최고 성능 모델 저장
best_model_obj = models[best_overall['Model']]['trained_model']

with open(f'best_model_{best_overall["Model"].replace(" ", "_")}.pkl', 'wb') as f:
    pickle.dump(best_model_obj, f)

print(f"✅ 최고 성능 모델 저장: best_model_{best_overall['Model'].replace(' ', '_')}.pkl")

# 스케일러도 저장 (필요시)
if models[best_overall['Model']]['use_scaled']:
    with open('scaler_for_best_model.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    print("✅ 스케일러 저장: scaler_for_best_model.pkl")

# %% [markdown]
# ## 16. 마무리

# %%
print("\n" + "=" * 80)
print("✨ 모델 학습 파이프라인 완료!")
print("=" * 80)
print("\n생성된 파일:")
print("  📄 model_comparison_results.csv")
print("  📊 model_comparison_charts.png")
print("  📊 precision_recall_f1_comparison.png")
print("  📊 roc_curves_comparison.png")
print(f"  📊 confusion_matrix_{best_overall['Model'].replace(' ', '_')}.png")
print(f"  💾 best_model_{best_overall['Model'].replace(' ', '_')}.pkl")
print("  📊 feature_importance_*.png (트리 기반 모델)")
print("  📄 feature_importance_*.csv (트리 기반 모델)")
print("\n다음 단계:")
print("  1. 최고 성능 모델로 하이퍼파라미터 튜닝")
print("  2. 교차 검증으로 안정성 확인")
print("  3. 앙상블 기법 적용 고려")
print("  4. Streamlit 앱에 통합")
