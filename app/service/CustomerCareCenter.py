import numpy as np
import pandas as pd
# 상위 경로 설정 덕분에 'adapters'에서 바로 가져올 수 있습니다.
from adapters.purchase_intent_pr_auc_adapter import PurchaseIntentPRAUCModelAdapter

class PurchaseIntentService:
    def __init__(self, adapter):
        self.adapter = adapter
    
    def score_top_k(self, features, top_k_ratio=0.05):
        features = features.drop(columns=["Revenue"], errors="ignore")
        proba = self.adapter.predict_proba(features)
        thr = float(np.quantile(proba.values, 1.0 - top_k_ratio))
        pred = (proba >= thr).astype(int)
        
        out = features.copy()
        out["purchase_proba"] = proba
        out["purchase_pred"] = pred
        return out

    def classify_risk(self, purchase_proba: float) -> str:
        if purchase_proba < 0.2:
            return "HIGH_RISK"
        elif purchase_proba < 0.6:
            return "OPPORTUNITY"
        else:
            return "LIKELY_BUYER"

    def recommend_action(self, row: dict, purchase_proba: float) -> str:
        bounce = row.get("BounceRates", 0)
        exit_r = row.get("ExitRates", 0)
        page_val = row.get("PageValues", 0)
        duration = row.get("ProductRelated_Duration", 0)

        if purchase_proba < 0.4 and page_val > 0 and duration > 60:
            return "상품에 대한 관심은 높지만 구매로 이어지지 않고 있습니다. → 할인 쿠폰 제공 또는 리뷰 강조 배너 노출을 고려하세요."
        if bounce > 0.5 or exit_r > 0.5:
            return "초기 이탈 가능성이 높습니다. → 랜딩 페이지에 핵심 요약 정보 및 신뢰 요소 추가가 필요합니다."
        if purchase_proba >= 0.6:
            return "구매 가능성이 높습니다. 추가 개입 없이 구매 흐름을 유지하세요."
        return "현재는 관찰이 필요한 세션입니다."
    
     #PurchaseIntentService10