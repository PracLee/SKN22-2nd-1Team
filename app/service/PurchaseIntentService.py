import numpy as np
import pandas as pd

from adapters.purchase_intent_pr_auc_adapter import PurchaseIntentPRAUCModelAdapter



class PurchaseIntentService:
    def __init__(self, adapter: PurchaseIntentPRAUCModelAdapter):
        self.adapter = adapter
    
    # top_k_ratio: 상위 k%만 타깃팅(1)으로 표시하기 위한 비율 (예: 0.05 = 상위 5%)
    # 내부적으로 purchase_proba를 내림차순 정렬 후, 상위 k% 커트라인(threshold)을 quantile로 계산해 적용
    def score_top_k(self, features: pd.DataFrame, top_k_ratio: float = 0.05) -> pd.DataFrame:
        proba = self.adapter.predict_proba(features)

        # 상위 k% 컷
        thr = float(np.quantile(proba.values, 1.0 - top_k_ratio))
        pred = (proba >= thr).astype(int)

        out = features.copy()
        out["purchase_proba"] = proba
        out["purchase_pred"] = pred
        out["threshold_used"] = thr
        out["top_k_ratio"] = top_k_ratio
        return out
