from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

# from src.adapters.model_loader import JoblibArtifactLoader
from adapters.model_loader import JoblibArtifactLoader

class PurchaseIntentPRAUCModelAdapter:
    """
    PR-AUC 최적화로 선택된 모델 artifact를 로드해서
    서비스에서 사용할 수 있게 predict_proba 중심 API 제공
    """

    def __init__(self, artifact_path: str | Path):
        self._loader = JoblibArtifactLoader(artifact_path)

    @property
    def meta(self) -> Dict[str, Any]:
        return self._loader.load().meta

    def predict_proba(self, features: pd.DataFrame) -> pd.Series:
        pipe = self._loader.load().pipeline
        proba = pipe.predict_proba(features)[:, 1]
        return pd.Series(proba, index=features.index, name="purchase_proba")

    def predict(self, features: pd.DataFrame, threshold: float) -> pd.Series:
        # PR-AUC 모델은 threshold를 “정책”으로 서비스가 주는 걸 권장
        proba = self.predict_proba(features)
        return (proba >= float(threshold)).astype(int).rename("purchase_pred")
