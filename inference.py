"""Score the gold feature store with the latest model from the model bank.

Runs in pure pandas — the pickled artefact carries its own imputer, target-encoding
maps and scaler, so inference reproduces the training-time transformations exactly
without needing a Spark session.

Usage:
    python inference.py                                            # score every month
    python inference.py --snapshotdate 2024-09-01                  # one month
    python inference.py --modelname credit_model_2024_12_01.pkl    # pick a model
"""

import os
import glob
import argparse
import pickle

import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# Preprocessing helpers (must mirror train.py exactly)
# -----------------------------------------------------------------------------
def derive_categoricals(df):
    """
    Re-create the categorical columns used at training time from the raw
    Payment_Behaviour / Occupation fields kept in the Gold table.
    """
    pb = df["Payment_Behaviour"].fillna("").astype(str)
    df = df.copy()
    # Spend level bucket (Low / High / Unknown)
    df["Spend_Level"] = np.where(pb.str.contains("Low_spent"), "Low",
                         np.where(pb.str.contains("High_spent"), "High", "Unknown"))
    # Payment value bucket (Small / Medium / Large / Unknown)
    df["Payment_Value"] = np.where(pb.str.contains("Small_value"), "Small",
                          np.where(pb.str.contains("Medium_value"), "Medium",
                          np.where(pb.str.contains("Large_value"), "Large", "Unknown")))
    df["Occupation"] = df["Occupation"].fillna("Unknown").astype(str)
    return df


def apply_target_encoding(X_cat, te_maps):
    """
    Apply the train-only target-encoding maps. Unseen categories fall back to
    the global (train) mean stored inside each map -> no leakage, no NaN.
    """
    out = {}
    for c, (mapping, global_mean) in te_maps.items():
        out[c + "_TE"] = X_cat[c].map(mapping).fillna(global_mean).astype(float).values
    return pd.DataFrame(out, index=X_cat.index)


# -----------------------------------------------------------------------------
# Model bank access
# -----------------------------------------------------------------------------
def load_model_artefact(model_bank_dir, model_name=None):
    """
    Load a model artefact (.pkl) from the model bank. If no explicit name is
    given, pick the most recently modified artefact (the "best/latest" model).
    """
    if model_name:
        model_path = os.path.join(model_bank_dir, model_name)
    else:
        candidates = glob.glob(os.path.join(model_bank_dir, "*.pkl"))
        if not candidates:
            raise FileNotFoundError(f"No model artefacts in {model_bank_dir}. Run train.py first.")
        model_path = max(candidates, key=os.path.getmtime)

    print(f"Model artefact: {model_path}")
    with open(model_path, "rb") as f:
        artefact = pickle.load(f)
    return artefact, os.path.basename(model_path)


# -----------------------------------------------------------------------------
# Gold feature store access
# -----------------------------------------------------------------------------
def load_gold_features(gold_label_dir):
    """
    Read every Gold label-store partition and de-duplicate to one row per loan
    (Gold writes one cumulative file per month, so loans repeat across files).
    """
    parts = glob.glob(os.path.join(gold_label_dir, "*.parquet"))
    if not parts:
        raise FileNotFoundError(f"No gold partitions in {gold_label_dir}. Run the data pipeline first.")
    df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    df = df.drop_duplicates("loan_id")
    df["snapshot_date"] = pd.to_datetime(df["snapshot_date"]).dt.date
    return df


def main(snapshot_date_arg=None, model_name=None):

    model_bank_dir = "model_bank/"
    gold_label_dir = "datamart/gold/label_store/"
    pred_out_dir = "datamart/gold/model_predictions/"
    os.makedirs(pred_out_dir, exist_ok=True)

    # 1. Retrieve the best/latest model and its preprocessing objects
    artefact, model_file = load_model_artefact(model_bank_dir, model_name)
    model = artefact["model"]
    feature_names = artefact["feature_names"]
    model_version = artefact["model_version"]
    pp = artefact["preprocessing_transformers"]
    imputer = pp["imputer"]
    te_maps = pp["target_encoding"]
    numeric_cols = pp["numeric_cols"]
    categorical_cols = pp["categorical_cols"]
    scaler = pp["stdscaler"]
    print(f"Version {model_version}, {len(feature_names)} features")

    # 2. Load the Gold feature store (one row per loan)
    df = load_gold_features(gold_label_dir)

    # 3. Optionally restrict to a single application month
    if snapshot_date_arg:
        target = pd.to_datetime(snapshot_date_arg).date()
        df = df[df["snapshot_date"] == target].copy()
        print(f"Scoring application month {target}")
    else:
        print("Scoring all application months")
    print(f"Rows to score: {len(df)}")

    if df.empty:
        print("No rows to score.")
        return

    # 4. Rebuild features EXACTLY as in training
    df = derive_categoricals(df)
    #   4a. Numeric block -> median impute (fitted on train)
    num_block = pd.DataFrame(imputer.transform(df[numeric_cols]), columns=numeric_cols, index=df.index)
    #   4b. Categorical block -> train-only target encoding
    te_block = apply_target_encoding(df[categorical_cols], te_maps)
    #   4c. Assemble in the exact training feature order, then scale
    X = pd.concat([num_block, te_block], axis=1)[feature_names]
    X_scaled = scaler.transform(X)

    # 5. Predict default probability and a 0.5-threshold class label
    proba = model.predict_proba(X_scaled)[:, 1]
    pred_label = (proba >= 0.5).astype(int)

    predictions = pd.DataFrame({
        "loan_id": df["loan_id"].values,
        "Customer_ID": df["Customer_ID"].values,
        "snapshot_date": df["snapshot_date"].values,
        "label": df["label"].values,             # actual label kept for monitoring
        "model_version": model_version,
        "model_predict_proba": proba,
        "model_predict_label": pred_label,
    })

    # 6. Persist predictions as a Gold table, one parquet file per month
    n_written = 0
    for snap, group in predictions.groupby("snapshot_date"):
        suffix = str(snap).replace("-", "_")
        out_path = os.path.join(pred_out_dir, f"gold_predictions_{suffix}.parquet")
        group.to_parquet(out_path, index=False)
        n_written += 1
    print(f"Wrote {n_written} monthly partition(s) to {pred_out_dir}")
    print(f"Predicted bad rate {round(pred_label.mean(), 3)} vs actual {round(df['label'].mean(), 3)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Credit Model Inference / Scoring Runner")
    parser.add_argument("--snapshotdate", type=str, required=False,
                        help="Application month YYYY-MM-DD to score. Omit to score all months.")
    parser.add_argument("--modelname", type=str, required=False,
                        help="Model artefact filename in model_bank/. Omit to use the latest.")
    args = parser.parse_args()
    main(snapshot_date_arg=args.snapshotdate, model_name=args.modelname)
