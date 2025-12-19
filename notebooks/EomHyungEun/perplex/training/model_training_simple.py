
# ============================================================================
# 간단 버전 - 모델 학습 & 비교
# 주피터 노트북용 (핵심만)
# ============================================================================

# %% 1. 라이브러리
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
import warnings
warnings.filterwarnings('ignore')

# 선택사항
try:
    import xgboost as xgb
    HAS_XGB = True
except:
    HAS_XGB = False

try:
    import lightgbm as lgb
    HAS_LGB = True
except:
    HAS_LGB = False

print("✅ 라이브러리 로드 완료")

# %% 2. 데이터 로드
train_df = pd.read_csv("../../data/processed/train.csv")
test_df = pd.read_csv("../../data/processed/test.csv")

print(f"Train: {train_df.shape}, Test: {test_df.shape}")
print(f"구매율 - Train: {train_df['Revenue'].mean():.2%}, Test: {test_df['Revenue'].mean():.2%}")

# %% 3. 데이터 준비
X_train = train_df.drop('Revenue', axis=1)
y_train = train_df['Revenue'].astype(int)
X_test = test_df.drop('Revenue', axis=1)
y_test = test_df['Revenue'].astype(int)

# 범주형 인코딩 (필요시)
categorical_cols = X_train.select_dtypes(include=['object', 'category', 'bool']).columns
for col in categorical_cols:
    le = LabelEncoder()
    X_train[col] = le.fit_transform(X_train[col].astype(str))
    X_test[col] = X_test[col].astype(str).apply(lambda x: le.transform([x])[0] if x in le.classes_ else -1)

# 스케일링 (LR용)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("✅ 데이터 준비 완료")

# %% 4. 모델 정의
models = {}

models['LR'] = {'model': LogisticRegression(max_iter=2000, random_state=42, class_weight='balanced'), 'scaled': True}
models['RF'] = {'model': RandomForestClassifier(n_estimators=150, max_depth=15, random_state=42, class_weight='balanced', n_jobs=-1), 'scaled': False}
models['GB'] = {'model': GradientBoostingClassifier(n_estimators=150, learning_rate=0.1, max_depth=5, random_state=42), 'scaled': False}

if HAS_XGB:
    models['XGB'] = {'model': xgb.XGBClassifier(n_estimators=150, learning_rate=0.1, max_depth=5, random_state=42, eval_metric='logloss'), 'scaled': False}

if HAS_LGB:
    models['LGB'] = {'model': lgb.LGBMClassifier(n_estimators=150, learning_rate=0.1, max_depth=5, random_state=42, class_weight='balanced', verbose=-1), 'scaled': False}

print(f"✅ {len(models)}개 모델 정의")

# %% 5. 학습 & 평가
results = []

for name, config in models.items():
    print(f"\n{'='*60}")
    print(f"🔹 {name} 학습 중...")

    model = config['model']
    X_tr = X_train_scaled if config['scaled'] else X_train
    X_te = X_test_scaled if config['scaled'] else X_test

    # 학습
    model.fit(X_tr, y_train)

    # 예측
    y_pred = model.predict(X_te)
    y_proba = model.predict_proba(X_te)[:, 1]

    # 평가
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_proba)

    results.append({
        'Model': name,
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1-Score': f1,
        'ROC-AUC': auc
    })

    print(f"Accuracy: {acc:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")

    # 저장
    config['trained'] = model
    config['y_proba'] = y_proba

print(f"\n{'='*60}")
print("✅ 학습 완료!")

# %% 6. 결과 비교
results_df = pd.DataFrame(results)
print("\n" + "="*60)
print("📊 모델 성능 비교")
print("="*60)
print(results_df.to_string(index=False))

# 최고 모델
best = results_df.loc[results_df['ROC-AUC'].idxmax()]
print(f"\n🏆 최고 성능: {best['Model']} (AUC: {best['ROC-AUC']:.4f})")

# CSV 저장
results_df.to_csv('model_results.csv', index=False)
print("\n✅ 결과 저장: model_results.csv")

# %% 7. 시각화
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Accuracy
axes[0].barh(results_df['Model'], results_df['Accuracy'], color='steelblue')
axes[0].set_xlabel('Accuracy')
axes[0].set_title('Test Accuracy')
for i, v in enumerate(results_df['Accuracy']):
    axes[0].text(v + 0.01, i, f'{v:.4f}', va='center')

# F1-Score
axes[1].barh(results_df['Model'], results_df['F1-Score'], color='coral')
axes[1].set_xlabel('F1-Score')
axes[1].set_title('Test F1-Score')
for i, v in enumerate(results_df['F1-Score']):
    axes[1].text(v + 0.01, i, f'{v:.4f}', va='center')

# ROC-AUC
axes[2].barh(results_df['Model'], results_df['ROC-AUC'], color='mediumseagreen')
axes[2].set_xlabel('ROC-AUC')
axes[2].set_title('Test ROC-AUC')
for i, v in enumerate(results_df['ROC-AUC']):
    axes[2].text(v + 0.005, i, f'{v:.4f}', va='center')

plt.tight_layout()
plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

print("✅ 차트 저장: model_comparison.png")

# %% 8. ROC Curve
from sklearn.metrics import roc_curve, auc

plt.figure(figsize=(10, 8))

for name, config in models.items():
    if 'y_proba' in config:
        fpr, tpr, _ = roc_curve(y_test, config['y_proba'])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_auc:.4f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random (AUC = 0.5000)')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curve 비교', fontsize=14, fontweight='bold')
plt.legend(loc='lower right', fontsize=10)
plt.grid(alpha=0.3)
plt.savefig('roc_curves.png', dpi=300, bbox_inches='tight')
plt.show()

print("✅ ROC 차트 저장: roc_curves.png")

# %% 9. Feature Importance (트리 기반)
for name, config in models.items():
    if name in ['RF', 'GB', 'XGB', 'LGB'] and 'trained' in config:
        model = config['trained']

        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            fi_df = pd.DataFrame({
                'Feature': X_train.columns,
                'Importance': importances
            }).sort_values('Importance', ascending=False).head(15)

            plt.figure(figsize=(10, 6))
            plt.barh(fi_df['Feature'], fi_df['Importance'], color='teal')
            plt.xlabel('Importance')
            plt.title(f'{name} - Feature Importance (Top 15)', fontweight='bold')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.savefig(f'fi_{name}.png', dpi=300, bbox_inches='tight')
            plt.show()

            print(f"✅ {name} Feature Importance 저장")

# %% 10. 모델 저장
import pickle

best_model_name = best['Model']
best_model = models[best_model_name]['trained']

with open(f'best_model_{best_model_name}.pkl', 'wb') as f:
    pickle.dump(best_model, f)

print(f"✅ 최고 모델 저장: best_model_{best_model_name}.pkl")

# 스케일러도 저장 (필요시)
if models[best_model_name]['scaled']:
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    print("✅ 스케일러 저장: scaler.pkl")

# %% 11. Classification Report
print("\n" + "="*60)
print(f"📊 {best_model_name} - Classification Report")
print("="*60)
best_y_pred = best_model.predict(X_test_scaled if models[best_model_name]['scaled'] else X_test)
print(classification_report(y_test, best_y_pred, target_names=['No Purchase', 'Purchase']))

print("\n✨ 완료!")
