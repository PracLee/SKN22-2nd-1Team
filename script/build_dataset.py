#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

# script/build_dataset.py -> parent=script -> parent.parent=ROOT
ROOT = Path(__file__).resolve().parent.parent
RAW_PATH = ROOT / "data" / "raw" / "online_shoppers_intention.csv"
OUT_DIR = ROOT / "data" / "processed"

TARGET = "Revenue"
TEST_SIZE = 0.2          # 전체에서 test로 뺄 비율
CALIB_SIZE_IN_TRAIN = 0.2  # train 내부에서 calib로 뺄 비율 (즉, 전체 대비 0.8*0.2=0.16)
SEED = 42


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(RAW_PATH)

    # Revenue가 True/False일 수 있어 0/1로 통일
    if df[TARGET].dtype == bool:
        df[TARGET] = df[TARGET].astype(int)

    # 원본에 고유 ID가 없으니 재현성/추적용 row_id 부여
    df = df.reset_index(drop=True)
    df.insert(0, "row_id", df.index)

    # 1) 전체 -> train_full / test
    train_full_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=SEED,
        stratify=df[TARGET],
        shuffle=True,
    )

    # 2) train_full -> train / calib
    train_df, calib_df = train_test_split(
        train_full_df,
        test_size=CALIB_SIZE_IN_TRAIN,
        random_state=SEED,
        stratify=train_full_df[TARGET],
        shuffle=True,
    )

    train_df.to_csv(OUT_DIR / "train.csv", index=False)
    calib_df.to_csv(OUT_DIR / "calib.csv", index=False)
    test_df.to_csv(OUT_DIR / "test.csv", index=False)

    meta = {
        "seed": SEED,
        "target": TARGET,

        # split ratios
        "test_size_of_total": TEST_SIZE,
        "calib_size_within_train": CALIB_SIZE_IN_TRAIN,
        "calib_size_of_total": float((1.0 - TEST_SIZE) * CALIB_SIZE_IN_TRAIN),
        "train_size_of_total": float((1.0 - TEST_SIZE) * (1.0 - CALIB_SIZE_IN_TRAIN)),

        # counts
        "n_total": int(len(df)),
        "n_train": int(len(train_df)),
        "n_calib": int(len(calib_df)),
        "n_test": int(len(test_df)),

        # positive rates
        "train_positive_rate": float(train_df[TARGET].mean()),
        "calib_positive_rate": float(calib_df[TARGET].mean()),
        "test_positive_rate": float(test_df[TARGET].mean()),
    }

    (OUT_DIR / "split_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )


if __name__ == "__main__":
    main()
