#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build (train + calibrate + save) BalancedRandomForest artifact.

Key idea (Option #2):
  - artifact["pipeline"]에 'calibrated model'을 넣어서 호출부 변경을 최소화한다.
  - 호출부는 기존처럼 artifact["pipeline"].predict_proba(X) 사용 가능.

Input:
  - data/processed/train.csv
  - (optional) data/processed/calib.csv  (없으면 train에서 split)
  - (optional) data/processed/test.csv   (있고 라벨 있으면 sanity check)

Output:
  - app/artifacts/best_pr_auc_balancedrf.joblib
"""

from __future__ import annotations

import argparse
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator

from imblearn.ensemble import BalancedRandomForestClassifier


# === 고정: 이미 찾은 best params ===
BEST_PARAMS = {
    "sampling_strategy": 0.5,
    "replacement": False,
    "n_estimators": 300,
    "min_samples_split": 5,
    "min_samples_leaf": 1,
    "max_features": 0.3,
    "max_depth": 8,
}

PAST_BEST_CV_PR_AUC = 0.7559602666233075


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build CALIBRATED artifact with fixed best PR-AUC params (BalancedRF)."
    )

    root = Path(__file__).resolve().parent.parent
    default_train = root / "data" / "processed" / "train.csv"
    default_calib = root / "data" / "processed" / "calib.csv"
    default_test = root / "data" / "processed" / "test.csv"
    default_out = root / "app" / "artifacts" / "best_pr_auc_balancedrf.joblib"

    p.add_argument(
        "--train", type=str, default=str(default_train), help="Path to train.csv"
    )
    p.add_argument(
        "--calib",
        type=str,
        default=str(default_calib),
        help="Path to calib.csv (optional). If not exists, split from train.",
    )
    p.add_argument(
        "--calib_size",
        type=float,
        default=0.2,
        help="If calib.csv not found, split fraction from train for calibration.",
    )
    p.add_argument(
        "--cal_method",
        type=str,
        default="sigmoid",
        choices=["sigmoid", "isotonic"],
        help="Calibration method. sigmoid is usually safer for smaller calib sets.",
    )
    p.add_argument(
        "--test",
        type=str,
        default=str(default_test),
        help="Path to test.csv (optional for eval)",
    )
    p.add_argument("--target", type=str, default="Revenue", help="Target column name")
    p.add_argument(
        "--out", type=str, default=str(default_out), help="Output joblib path"
    )

    p.add_argument("--random_state", type=int, default=42, help="Random seed")
    p.add_argument(
        "--no_eval", action="store_true", help="Skip evaluation even if test has label"
    )

    p.add_argument(
        "--compress",
        type=str,
        default="xz:3",
        help="joblib compress option, e.g. 'xz:3', 'zlib:3', or 'none'",
    )
    return p.parse_args()


def parse_compress_arg(s: str):
    s = (s or "").strip().lower()
    if s in ("none", "0", "false", "off"):
        return 0
    if ":" in s:
        method, lvl = s.split(":", 1)
        return (method, int(lvl))
    if s.isdigit():
        return int(s)
    return ("xz", 3)


def best_fbeta_threshold(y_true: np.ndarray, proba: np.ndarray, beta: float = 2.0):
    thresholds = np.linspace(0.0, 1.0, 2001)
    best = {"thr": 0.5, "f": -1.0, "precision": 0.0, "recall": 0.0}
    for thr in thresholds:
        pred = (proba >= thr).astype(int)
        p, r, f, _ = precision_recall_fscore_support(
            y_true, pred, average="binary", beta=beta, zero_division=0
        )
        if f > best["f"]:
            best = {
                "thr": float(thr),
                "f": float(f),
                "precision": float(p),
                "recall": float(r),
            }
    return best


def main() -> None:
    args = parse_args()

    train_path = Path(args.train)
    calib_path = Path(args.calib)
    test_path = Path(args.test)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not train_path.exists():
        raise FileNotFoundError(f"Train file not found: {train_path}")

    # ---- Load train
    train_df = pd.read_csv(train_path)
    if args.target not in train_df.columns:
        raise ValueError(f"Target '{args.target}' not found in train columns.")

    X_all = train_df.drop(columns=[args.target])
    y_all = train_df[args.target].astype(int).to_numpy()

    # ---- Prepare calibration data
    if calib_path.exists():
        calib_df = pd.read_csv(calib_path)
        if args.target not in calib_df.columns:
            raise ValueError(f"Target '{args.target}' not found in calib columns.")
        X_fit = X_all
        y_fit = y_all
        X_cal = calib_df.drop(columns=[args.target])
        y_cal = calib_df[args.target].astype(int).to_numpy()
        calib_origin = f"file:{calib_path}"
    else:
        X_fit, X_cal, y_fit, y_cal = train_test_split(
            X_all,
            y_all,
            test_size=args.calib_size,
            random_state=args.random_state,
            stratify=y_all,
        )
        calib_origin = f"split_from_train:test_size={args.calib_size}"

    # ---- Column typing (fit set 기준)
    num_cols = X_fit.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = X_fit.select_dtypes(
        include=["object", "bool", "category"]
    ).columns.tolist()

    preprocess = ColumnTransformer(
        transformers=[
            ("num", RobustScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ],
        remainder="drop",
    )

    brf = BalancedRandomForestClassifier(
        **BEST_PARAMS,
        n_jobs=-1,
        random_state=args.random_state,
    )

    base_pipeline = Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("model", brf),
        ]
    )

    # ---- 1) Fit base model ONLY on fit set
    base_pipeline.fit(X_fit, y_fit)

    # ---- 2) Fit calibrator on calibration set (no leakage)
    calibrator = CalibratedClassifierCV(
        estimator=FrozenEstimator(base_pipeline),
        method=args.cal_method,
        cv=None,
    )
    calibrator.fit(X_cal, y_cal)

    # ---- Calibration metrics (reference)
    base_cal_proba = base_pipeline.predict_proba(X_cal)[:, 1]
    cal_cal_proba = calibrator.predict_proba(X_cal)[:, 1]

    calib_metrics = {
        "base_pr_auc": float(average_precision_score(y_cal, base_cal_proba)),
        "base_roc_auc": float(roc_auc_score(y_cal, base_cal_proba)),
        "cal_pr_auc": float(average_precision_score(y_cal, cal_cal_proba)),
        "cal_roc_auc": float(roc_auc_score(y_cal, cal_cal_proba)),
    }

    # ---- (권장) threshold 재최적화 (calibrated proba 기준)
    best_thr_f2 = best_fbeta_threshold(y_cal, cal_cal_proba, beta=2.0)

    print("\n=== Using fixed best params ===")
    print(json.dumps({f"model__{k}": v for k, v in BEST_PARAMS.items()}, indent=2))
    print("\n=== CALIB metrics (base vs calibrated) ===")
    print(json.dumps(calib_metrics, indent=2))
    print("\n=== CALIB best F2 threshold (on calibrated proba) ===")
    print(json.dumps(best_thr_f2, indent=2))

    # ---- Optional eval on test (calibrated)
    test_metrics = {}
    if (not args.no_eval) and test_path.exists():
        test_df = pd.read_csv(test_path)
        if args.target in test_df.columns:
            X_test = test_df.drop(columns=[args.target])
            y_test = test_df[args.target].astype(int).to_numpy()

            test_proba = calibrator.predict_proba(X_test)[:, 1]
            test_metrics = {
                "test_pr_auc": float(average_precision_score(y_test, test_proba)),
                "test_roc_auc": float(roc_auc_score(y_test, test_proba)),
            }
            print("\n=== TEST (calibrated reference) ===")
            print(json.dumps(test_metrics, indent=2))
        else:
            print("\n(test.csv에 target 컬럼이 없어 평가를 스킵합니다.)")

    # ---- Save artifact:
    # 핵심: artifact["pipeline"]에 calibrator를 넣는다 (호출부 변경 최소)
    artifact = {
        "pipeline": calibrator,  # (예측용) calibrated 모델
        "base_pipeline": base_pipeline,  # (디버깅/전처리 접근용) Pipeline -> named_steps 있음
        "best_params": {f"model__{k}": v for k, v in BEST_PARAMS.items()},
        "cv_best_pr_auc": float(PAST_BEST_CV_PR_AUC),
        "calibration": {
            "method": args.cal_method,
            "origin": calib_origin,
            "calib_metrics": calib_metrics,
        },
        # threshold도 calibrated 기준으로 저장해두는 것을 강력 추천
        "best_threshold_f2_on_calib": best_thr_f2,
        "test_metrics": test_metrics,
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "target_col": args.target,
        "meta": {
            "builder": "build_best_pr_auc_balancedrf_calibrated_option2.py",
            "scoring_origin": "average_precision (PR-AUC)",
            "note": "artifact['pipeline'] is CalibratedClassifierCV(FrozenEstimator(base_pipeline))",
            "random_state": args.random_state,
        },
    }

    compress_opt = parse_compress_arg(args.compress)
    joblib.dump(artifact, out_path, compress=compress_opt)
    print(f"\nSaved artifact to: {out_path.resolve()}")
    print(f"joblib.compress = {compress_opt}")


if __name__ == "__main__":
    main()
