from __future__ import annotations
from dataclasses import dataclass
from typing import List, Literal, Optional

import pandas as pd

from adapters.purchase_model_adapter import (
    PurchaseModelAdapter,
    PurchaseModelAdapterConfig,
    ModelStrategy,
)


RiskBand = Literal["high", "medium", "low"]


@dataclass
class SessionPredictionResult:
    probability: float
    risk_band: RiskBand
    status_label: str
    compare_text: str
    reasons: List[str]
    average_text: str


class SessionProbabilityService:
    def __init__(
        self,
        adapter: Optional[PurchaseModelAdapter] = None,
        global_avg_purchase_prob: float = 0.15,
        default_strategy: ModelStrategy = "roc_auc",
    ):
        self.adapter = adapter or PurchaseModelAdapter(
            PurchaseModelAdapterConfig.from_default_layout()
        )
        self.global_avg_purchase_prob = global_avg_purchase_prob
        self.default_strategy = default_strategy

    def predict_session(
        self,
        session_df: pd.DataFrame,
        strategy: Optional[ModelStrategy] = None,
    ) -> SessionPredictionResult:
        strategy = strategy or self.default_strategy
        prob = self.adapter.predict_purchase_probability(session_df, strategy=strategy)

        risk_band, status_label = self._get_risk_band_and_label(prob)
        compare_text = self._build_compare_text(prob, self.global_avg_purchase_prob)

        row = session_df.iloc[0]
        reasons, avg_text = self._build_explanation(row, prob, self.global_avg_purchase_prob)

        return SessionPredictionResult(
            probability=prob,
            risk_band=risk_band,
            status_label=status_label,
            compare_text=compare_text,
            reasons=reasons,
            average_text=avg_text,
        )
        
    def _get_risk_band_and_label(self, prob: float):
        """
        확률 구간에 따라 라벨링
        - 0.60 이상: 높음
        - 0.30 ~ 0.60: 보통
        - 0.30 미만: 낮음
        """
        if prob >= 0.60:
            return "high", "구매 가능성 높음"
        elif prob >= 0.30:
            return "medium", "구매 가능성 보통"
        else:
            return "low", "구매 가능성 낮음"
    
    def _build_compare_text(self, prob: float, avg_prob: float) -> str:
        """
        평균 대비 차이를 '배'가 아닌 '퍼센트포인트' 단위로 표시
        예) 평균 15.0%, 현재 34.9% -> "19.9%p 높습니다."
        """
        if avg_prob <= 0:
            return "평균 값이 정의되어 있지 않아 비교가 어렵습니다."

        diff_pp = (prob - avg_prob) * 100  # percentage points
        avg_percent = avg_prob * 100

        if abs(diff_pp) < 0.05:
            # 거의 차이 없을 때
            return f"이 세션의 구매 확률은 평균({avg_percent:.1f}%)과 거의 같습니다."

        direction = "높습니다" if diff_pp > 0 else "낮습니다"
        diff_abs = abs(diff_pp)

        return (
            f"이 세션의 구매 확률은 평균({avg_percent:.1f}%)보다 "
            f"{diff_abs:.1f}%p {direction}."
        )

    def _build_explanation(
        self,
        row: pd.Series,
        prob: float,
        avg_prob: float,
    ):
        reasons: List[str] = []

        product_related = row.get("ProductRelated", None)
        if product_related is not None:
            if product_related >= 20:
                reasons.append("상품 페이지를 많이 조회하고 있어 관심도가 높습니다.")
            elif product_related <= 3:
                reasons.append("상품 페이지 조회 수가 적어 아직 탐색 단계일 가능성이 있습니다.")

        page_values = row.get("PageValues", None)
        if page_values is not None:
            if page_values >= 50:
                reasons.append("이미 장바구니/결제 단계 등 높은 가치 페이지에 도달했습니다.")
            elif page_values == 0:
                reasons.append("아직 구매 여정의 앞단에 있어 구체적인 구매 신호가 약합니다.")

        exit_rates = row.get("ExitRates", None)
        if exit_rates is not None:
            if exit_rates <= 0.2:
                reasons.append("세션 종료 비율이 낮아 이탈 위험이 비교적 적습니다.")
            elif exit_rates >= 0.5:
                reasons.append("세션 종료 비율이 높아 이탈 가능성이 큽니다.")

        visitor_type = row.get("VisitorType", None)
        if visitor_type == "Returning_Visitor":
            reasons.append("재방문 고객으로, 사이트 경험이 있어 구매 가능성이 더 높습니다.")
        elif visitor_type == "New_Visitor":
            reasons.append("신규 방문자로, 아직 사이트에 익숙하지 않아 구매까지 시간이 걸릴 수 있습니다.")

        weekend = row.get("Weekend", None)
        if isinstance(weekend, (bool, int)):
            if bool(weekend):
                reasons.append("주말 방문 세션으로, 여유 있는 쇼핑 가능성이 있습니다.")
            else:
                reasons.append("평일 방문 세션으로, 짧은 탐색 후 이탈할 수도 있습니다.")
                
        diff_pp = (prob - avg_prob) * 100
        if diff_pp >= 10:
            avg_text = f"이 세션의 구매 확률은 전체 평균보다 약 {diff_pp:.1f}%p 높습니다."
        elif diff_pp <= -10:
            avg_text = f"이 세션의 구매 확률은 전체 평균보다 약 {abs(diff_pp):.1f}%p 낮습니다."
        else:
            avg_text = "이 세션의 구매 확률은 전체 평균과 비슷한 수준입니다."

        return reasons, avg_text

    def get_training_data(self) -> pd.DataFrame:
        """
        학습에 사용된 원본 데이터를 로드하여 반환합니다.
        (EDA 및 시각화용)
        """
        # Adapter Config에서 root_dir을 가져옴
        root_dir = self.adapter.config.root_dir
        data_path = root_dir / "data" / "processed" / "train.csv"

        if not data_path.exists():
            raise FileNotFoundError(f"Training data not found at: {data_path}")

        df = pd.read_csv(data_path)
        if "row_id" in df.columns:
            df = df.drop(columns=["row_id"])
        
        return df
