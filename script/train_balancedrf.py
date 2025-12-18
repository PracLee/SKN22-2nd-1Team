#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Train BalancedRandomForest model on processed train.csv and save artifact to artifacts/.

Saves a joblib containing:
- pipeline: preprocess (RobustScaler + OneHotEncoder) + BalancedRandomForestClassifier
- best_threshold: float
- best_params: dict
- column lists (num_cols, cat_cols)
"""

from __future__ import annotations

import argparse
from pathlib import Path
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    f1_score, roc_auc_score, average_precision_score,
    confusion_matrix, classification_report
)

from imblearn.ensemble import BalancedRandomForestClassifier


DEFAULT_BEST_PARAMS = {
    "n_estimators": 1200,
    "max_depth": None,
    "min_samples_leaf": 8,
    "min_samples_split": 2,
    "max_features": 0.5,
    "sampling_strategy": "auto",
    "replacement": False,
}

DEFAULT_THRESHOLD = 0.7412374222999228
DEFAULT_RANDOM_STATE = 42


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train and save BalancedRF pipeline artifact.")
    p.add_argument("--train", type=str, default="data/processed/train.csv", help="Path to train.csv")
    p.add_argument("--test", type=str, default="data/processed/test.csv", help="Path to test.csv (optional for eval)")
    p.add_argument("--target", type=str, default="Revenue", help="Target column name")
    p.add_argument("--out", type=str, default="artifacts/best_balancedrf_pipeline.joblib", help="Output joblib path")
    p.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD, help="Decision threshold to save with model")
    p.add_argument("--random_state", type=int, default=DEFAULT_RANDOM_STATE, help="Random seed")
    p.add_argument("--no_eval", action="store_true", help="Skip evaluation even if test has label")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    train_path = Path(args.train)
    test_path = Path(args.test)
    out_path = Path(args.out)

    if not train_path.exists():
        raise FileNotFoundError(f"Train file not found: {train_path}")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load train
    train_df = pd.read_csv(train_path)
    if args.target not in train_df.columns:
        raise ValueError(f"Target '{args.target}' not found in train columns: {list(train_df.columns)}")

    X_train = train_df.drop(columns=[args.target])
    y_train = train_df[args.target].astype(int)

    # Auto column typing
    num_cols = X_train.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = X_train.select_dtypes(include=["object", "bool", "category"]).columns.tolist()

    preprocess = ColumnTransformer(
        transformers=[
            ("num", RobustScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ],
        remainder="drop",
    )

    # Model with best params you found
    brf = BalancedRandomForestClassifier(
        **DEFAULT_BEST_PARAMS,
        n_jobs=-1,
        random_state=args.random_state,
    )

    pipeline = Pipeline(steps=[
        ("preprocess", preprocess),
        ("model", brf),
    ])

    # Fit on full train
    pipeline.fit(X_train, y_train)

    # Optional evaluation (only if test exists and has label)
    if (not args.no_eval) and test_path.exists():
        test_df = pd.read_csv(test_path)
        if args.target in test_df.columns:
            X_test = test_df.drop(columns=[args.target])
            y_test = test_df[args.target].astype(int)

            proba = pipeline.predict_proba(X_test)[:, 1]
            pred = (proba >= args.threshold).astype(int)

            print("\n=== TEST Evaluation (using saved threshold) ===")
            print(f"Threshold: {args.threshold:.6f}")
            print("ROC-AUC :", roc_auc_score(y_test, proba))
            print("PR-AUC  :", average_precision_score(y_test, proba))
            print("F1      :", f1_score(y_test, pred))
            print("Confusion matrix:\n", confusion_matrix(y_test, pred))
            print("\nReport:\n", classification_report(y_test, pred, digits=4))
        else:
            print("\n(test.csv에 target 컬럼이 없어 평가를 스킵합니다.)")

    artifact = {
        "pipeline": pipeline,
        "best_threshold": float(args.threshold),
        "best_params": DEFAULT_BEST_PARAMS,
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "target_col": args.target,
    }

    joblib.dump(artifact, out_path)
    print(f"\nSaved artifact to: {out_path.resolve()}")


if __name__ == "__main__":
    main()
