from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Any, List

import joblib
import pandas as pd

ModelStrategy = Literal["roc_auc", "pr_auc"]


@dataclass
class PurchaseModelAdapterConfig:
    app_dir: Path
    roc_auc_model_path: Path
    pr_auc_model_path: Path

    @classmethod
    def from_default_layout(cls) -> "PurchaseModelAdapterConfig":
        """
        디렉토리 구조 가정:

        ROOT/
          └ app/
              ├ adapter/
              ├ service/
              ├ pages/
              └ artifacts/
                  ├ best_balancedrf_pipeline.joblib
                  └ best_pr_auc_balancedrf.joblib
        """
        adapter_dir = Path(__file__).resolve().parent   # app/adapter
        app_dir = adapter_dir.parent                    # app
        artifact_dir = app_dir / "artifacts"

        return cls(
            app_dir=app_dir,
            roc_auc_model_path=artifact_dir / "best_balancedrf_pipeline.joblib",
            pr_auc_model_path=artifact_dir / "best_pr_auc_balancedrf.joblib",
        )


def _extract_model(artifact: Any) -> Any:
    """
    joblib.load() 결과가 dict일 때, 안에서 진짜 모델 객체를 꺼내는 헬퍼.

    - artifact가 이미 predict_proba를 갖고 있으면 그대로 반환
    - dict이면 흔한 키들("model", "pipeline", "clf", "estimator", "classifier")을 우선 확인
    - 그래도 못 찾으면 values 전체를 돌면서 predict_proba 있는 애를 찾음
    """
    # 1) 이미 모델인 경우
    if hasattr(artifact, "predict_proba"):
        return artifact

    # 2) dict 안에 모델이 들어있는 경우
    if isinstance(artifact, dict):
        candidate_keys = ["model", "pipeline", "clf", "estimator", "classifier"]
        for key in candidate_keys:
            if key in artifact and hasattr(artifact[key], "predict_proba"):
                return artifact[key]

        # 키 이름을 모를 때: value들에서 찾아보기
        for value in artifact.values():
            if hasattr(value, "predict_proba"):
                return value

    raise TypeError(
        "Loaded artifact does not contain a model with 'predict_proba'. "
        f"type={type(artifact)}"
    )


class PurchaseModelAdapter:
    """
    DataFrame -> 구매 확률 어댑터
    """

    def __init__(self, config: Optional[PurchaseModelAdapterConfig] = None):
        self.config = config or PurchaseModelAdapterConfig.from_default_layout()
        self._roc_auc_model = None
        self._pr_auc_model = None

    # --------------------
    # 내부 로더
    # --------------------
    def _load_roc_auc_model(self):
        if self._roc_auc_model is None:
            artifact = joblib.load(self.config.roc_auc_model_path)
            self._roc_auc_model = _extract_model(artifact)
        return self._roc_auc_model

    def _load_pr_auc_model(self):
        if self._pr_auc_model is None:
            artifact = joblib.load(self.config.pr_auc_model_path)
            self._pr_auc_model = _extract_model(artifact)
        return self._pr_auc_model

    def _get_model(self, strategy: ModelStrategy):
        if strategy == "roc_auc":
            return self._load_roc_auc_model()
        elif strategy == "pr_auc":
            return self._load_pr_auc_model()
        else:
            raise ValueError(f"Unknown model strategy: {strategy}")

    # --------------------
    # Feature 정렬/채우기
    # --------------------
    def _align_features(self, df: pd.DataFrame, model: Any) -> pd.DataFrame:
        """
        - 모델이 학습에 사용한 feature_names_in_에 맞춰
          * 없는 컬럼은 기본값(0)으로 추가
          * 순서를 동일하게 맞춰줌
        """
        # feature_names_in_ 이 있으면 그걸 기준으로 맞춘다.
        feature_names: Optional[List[str]] = None

        if hasattr(model, "feature_names_in_"):
            feature_names = list(model.feature_names_in_)
        # Pipeline일 경우도 feature_names_in_을 대부분 갖고 있음
        # 없으면 그냥 원본 df 그대로 사용
        if feature_names is None:
            return df

        df = df.copy()

        # 없는 컬럼은 0으로 채워서 추가
        for col in feature_names:
            if col not in df.columns:
                df[col] = 0

        # (모델이 쓰지 않는) 추가 컬럼은 버린다.
        extra_cols = [c for c in df.columns if c not in feature_names]
        if extra_cols:
            df = df.drop(columns=extra_cols)

        # 최종 순서 맞추기
        df = df[feature_names]

        return df

    # --------------------
    # Public API
    # --------------------
    def predict_proba(
        self,
        session_df: pd.DataFrame,
        strategy: ModelStrategy = "roc_auc",
    ):
        """
        - session_df: 1개 이상 row를 가진 DataFrame
        - return: model.predict_proba(aligned_df)
        """
        model = self._get_model(strategy)
        aligned_df = self._align_features(session_df, model)
        return model.predict_proba(aligned_df)

    def predict_purchase_probability(
        self,
        session_df: pd.DataFrame,
        strategy: ModelStrategy = "roc_auc",
    ) -> float:
        """
        하나의 세션(row)에 대한 1(구매) 클래스 확률만 반환
        """
        proba = self.predict_proba(session_df, strategy=strategy)
        return float(proba[0][1])
