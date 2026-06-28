import os
import glob
import argparse
import pickle
import pprint
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

import pyspark
import pyspark.sql.functions as F
from pyspark.sql.functions import col

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer 
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import roc_auc_score


# -----------------------------------------------------------------------------
# TensorBoard training logging
# XGBoost is not a deep-learning model, so we stream the per-boosting-round
# evaluation metrics (AUC / logloss on train & dev) to TensorBoard event files
# via a training callback. TensorBoard (served at http://localhost:6006) then
# refreshes these curves live as the model trains.
# -----------------------------------------------------------------------------
class TensorBoardCallback(xgb.callback.TrainingCallback):
    def __init__(self, writer, name_map):
        self.writer = writer
        self.name_map = name_map  # e.g. {"validation_0": "train", "validation_1": "dev"}

    def after_iteration(self, model, epoch, evals_log):
        # Called after every boosting round -> write the latest metric values.
        for data_name, metric_dict in evals_log.items():
            tag = self.name_map.get(data_name, data_name)
            for metric_name, values in metric_dict.items():
                self.writer.add_scalar(f"{metric_name}/{tag}", values[-1], epoch)
        return False  # returning False means: never stop early


def create_dynamic_config(model_train_date_str, train_test_period_months=12, oot_period_months=2, train_test_ratio=0.8):
    """
    Generate chronological training, testing, and Out-of-Time (OOT) anchors
    based on the input target snapshot train date.
    """
    config = {}
    config["model_train_date_str"] = model_train_date_str
    config["train_test_period_months"] = train_test_period_months
    config["oot_period_months"] = oot_period_months
    config["train_test_ratio"] = train_test_ratio

    # Timeline calculations
    config["model_train_date"] = datetime.strptime(model_train_date_str, "%Y-%m-%d")
    config["oot_end_date"] = config['model_train_date'] - timedelta(days=1)
    config["oot_start_date"] = config['model_train_date'] - relativedelta(months=oot_period_months)
    config["train_test_end_date"] = config["oot_start_date"] - timedelta(days=1)
    config["train_test_start_date"] = config["oot_start_date"] - relativedelta(months=train_test_period_months)

    return config


#Target encoding
def derive_categoricals(df):
    pb = df["Payment_Behaviour"].fillna("").astype(str)
    df = df.copy()
    df["Spend_Level"] = np.where(pb.str.contains("Low_spent"), "Low",
                         np.where(pb.str.contains("High_spent"), "High", "Unknown"))
    df["Payment_Value"] = np.where(pb.str.contains("Small_value"), "Small",
                          np.where(pb.str.contains("Medium_value"), "Medium",
                          np.where(pb.str.contains("Large_value"), "Large", "Unknown")))
    df["Occupation"] = df["Occupation"].fillna("Unknown").astype(str)
    return df


def fit_target_encoding(X_cat, y, cols, smoothing=20):
    global_mean = float(y.mean())
    maps = {}
    for c in cols:
        stats = pd.DataFrame({"cat": X_cat[c].values, "y": y.values}).groupby("cat")["y"].agg(["mean", "count"])
        smooth = (stats["mean"] * stats["count"] + global_mean * smoothing) / (stats["count"] + smoothing)
        maps[c] = (smooth.to_dict(), global_mean)
    return maps


def apply_target_encoding(X_cat, maps):
    out = {}
    for c, (mp, gm) in maps.items():
        out[c + "_TE"] = X_cat[c].map(mp).fillna(gm).astype(float).values
    return pd.DataFrame(out, index=X_cat.index)


def main(snapshot_date_arg, tune=True, hp=None):
    print("\n=====================================================================")
    print("      INITIATING PRODUCTION MODEL TRAINING PIPELINE ARCHITECTURE     ")
    print("=====================================================================\n")

    # 1. Initialize heavy local-optimized Spark Session
    spark = pyspark.sql.SparkSession.builder \
        .appName("Credit-Model-Training") \
        .master("local[*]") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    # 2. Build configuration setup
    config = create_dynamic_config(model_train_date_str=snapshot_date_arg)
    print("[SYSTEM INFO] Training configuration timeline anchors finalized:")
    pprint.pprint(config)

    # 3. Connect and extract from Gold Label Store
    label_folder_path = "datamart/gold/label_store/"
    label_files = glob.glob(os.path.join(label_folder_path, '*.parquet'))

    if not label_files:
        raise FileNotFoundError(f"[CRITICAL] No gold label parquet shards discovered at {label_folder_path}")

    label_store_sdf = spark.read.parquet(*label_files)

    
    label_store_sdf = label_store_sdf.dropDuplicates(["loan_id"])
    print(f"[INFO] Gold rows after de-duplication (1 row/loan): {label_store_sdf.count()}")

    # Apply dynamic historical timeline filter constraints
    labels_sdf = label_store_sdf.filter(
        (col("snapshot_date") >= config["train_test_start_date"]) &
        (col("snapshot_date") <= config["oot_end_date"])
    )
    print(f"\n[INFO] Extracted Gold Datasets snapshot records count: {labels_sdf.count()}")

    # 4. Migrate to Pandas
    print("[INFO] Loading analytical storage framework into Pandas ecosystem...")
    data_pdf = labels_sdf.toPandas()
    spark.stop()

    if data_pdf.empty:
        print("[WARNING] Compiled master dataframe is empty. Terminating process.")
        return

    # 5. Time-based cohorts (leakage-safe split)
    print("[INFO] Stratifying datasets into specialized chronological cohorts...")
    data_pdf['snapshot_date'] = pd.to_datetime(data_pdf['snapshot_date']).dt.date

    oot_start_d = config["oot_start_date"].date()
    oot_end_d = config["oot_end_date"].date()
    tt_start_d = config["train_test_start_date"].date()
    tt_end_d = config["train_test_end_date"].date()

    oot_pdf = data_pdf[(data_pdf['snapshot_date'] >= oot_start_d) & (data_pdf['snapshot_date'] <= oot_end_d)].copy()
    train_test_pdf = data_pdf[(data_pdf['snapshot_date'] >= tt_start_d) & (data_pdf['snapshot_date'] <= tt_end_d)].copy()


    oot_pdf = derive_categoricals(oot_pdf)
    train_test_pdf = derive_categoricals(train_test_pdf)

    # Numeric features
    numeric_cols = [
        "Debt_to_Income_Ratio", "EMI_to_Salary_Ratio", "Savings_Propensity",
        "Credit_History_Age_Years", "is_Credit_Age_gt_15", "Num_Bank_Accounts",
        "Num_Credit_Card", "Num_of_Loan", "Interest_Rate", "Monthly_Inhand_Salary",
        "Annual_Income", "Outstanding_Debt", "Credit_Utilization_Ratio",
        "Total_EMI_per_month", "Amount_invested_monthly", "Monthly_Balance",
        "Is_Age_gt_45", "Age",
        "fe_1", "fe_2", "fe_3", "fe_4", "fe_5", "fe_6", "fe_7", "fe_8", "fe_9", "fe_10",
        "fe_11", "fe_12", "fe_13", "fe_14", "fe_15", "fe_16", "fe_17", "fe_18", "fe_19", "fe_20"
    ]
    categorical_cols = ["Occupation", "Spend_Level", "Payment_Value"]
    te_cols = [c + "_TE" for c in categorical_cols]
    feature_cols = numeric_cols + te_cols   

    carry_cols = numeric_cols + categorical_cols

    # Train / Dev split (stratify)
    Xtt_df, Xdev_df, y_train, y_test = train_test_split(
        train_test_pdf[carry_cols],
        train_test_pdf["label"],
        test_size=1.0 - config["train_test_ratio"],
        random_state=88,
        shuffle=True,
        stratify=train_test_pdf["label"]
    )
    Xoot_df = oot_pdf[carry_cols].copy()
    y_oot = oot_pdf["label"]

    print(f"   -> Numeric features: {len(numeric_cols)} | Target-encoded: {len(te_cols)} | Total: {len(feature_cols)}")
    print(f"   -> Train Size: {Xtt_df.shape[0]} | Bad Rate: {round(y_train.mean(), 3)}")
    print(f"   -> Dev   Size: {Xdev_df.shape[0]} | Bad Rate: {round(y_test.mean(), 3)}")
    print(f"   -> OOT   Size: {Xoot_df.shape[0]} | Bad Rate: {round(y_oot.mean(), 3)}\n")

    # 6a. Impute numeric features:
    imputer = SimpleImputer(strategy="median")
    num_train = pd.DataFrame(imputer.fit_transform(Xtt_df[numeric_cols]), columns=numeric_cols, index=Xtt_df.index)
    num_dev   = pd.DataFrame(imputer.transform(Xdev_df[numeric_cols]),   columns=numeric_cols, index=Xdev_df.index)
    num_oot   = pd.DataFrame(imputer.transform(Xoot_df[numeric_cols]),   columns=numeric_cols, index=Xoot_df.index)

    # 6b. Target encoding categorical features
    te_maps = fit_target_encoding(Xtt_df[categorical_cols], y_train, categorical_cols, smoothing=20)
    te_train = apply_target_encoding(Xtt_df[categorical_cols], te_maps)
    te_dev   = apply_target_encoding(Xdev_df[categorical_cols], te_maps)
    te_oot   = apply_target_encoding(Xoot_df[categorical_cols], te_maps)

    # 6c. Assemble final feature matrices
    X_train_raw = pd.concat([num_train, te_train], axis=1)[feature_cols]
    X_test_raw  = pd.concat([num_dev,   te_dev],   axis=1)[feature_cols]
    X_oot       = pd.concat([num_oot,   te_oot],   axis=1)[feature_cols]

    # 6d. Standard Scaler
    print("[INFO] Fitting Standard Scaler preprocessing pipelines...")
    scaler = StandardScaler()
    transformer_stdscaler = scaler.fit(X_train_raw)
    X_train_processed = transformer_stdscaler.transform(X_train_raw)
    X_test_processed = transformer_stdscaler.transform(X_test_raw)
    X_oot_processed = transformer_stdscaler.transform(X_oot)

    # 7. Model fitting: hyperparameter SEARCH (tune=True) or FIXED params (tune=False)
    pos = int((y_train == 1).sum())
    neg = int((y_train == 0).sum())
    scale_pos_weight = (neg / pos) if pos > 0 else 1.0
    print(f"[INFO] Class balance -> pos={pos}, neg={neg}, scale_pos_weight={round(scale_pos_weight,3)}")

    if tune:
        # ---- tune=True: search a grid with RandomizedSearchCV ----
        print("[INFO] tune=True -> RandomizedSearchCV hyperparameter tuning...")
        xgb_clf = xgb.XGBClassifier(
            eval_metric='logloss', random_state=88, scale_pos_weight=scale_pos_weight
        )
        param_dist = {
            'n_estimators': [25, 50],
            'max_depth': [2, 3],
            'learning_rate': [0.01, 0.1],
            'subsample': [0.6, 0.8],
            'colsample_bytree': [0.6, 0.8],
            'gamma': [0, 0.1],
            'min_child_weight': [1, 3, 5],
            'reg_alpha': [0, 0.1, 1],
            'reg_lambda': [1, 1.5, 2]
        }
        random_search = RandomizedSearchCV(
            estimator=xgb_clf, param_distributions=param_dist, scoring='roc_auc',
            n_iter=100, cv=3, verbose=0, random_state=42, n_jobs=-1
        )
        print(">>> Fitting CV candidate permutations...")
        random_search.fit(X_train_processed, y_train)
        best_params = random_search.best_params_
        # best_estimator_ is already refit on the full training set (clean to pickle)
        best_model = random_search.best_estimator_
    else:
        # ---- tune=False: train ONE model with the fixed hyperparameters that
        # were supplied from the Airflow params / CLI (--n_estimators, ...) ----
        hp = hp or {}
        best_params = {
            'n_estimators': int(hp.get('n_estimators', 50)),
            'max_depth': int(hp.get('max_depth', 3)),
            'learning_rate': float(hp.get('learning_rate', 0.1)),
            'subsample': float(hp.get('subsample', 0.8)),
            'colsample_bytree': float(hp.get('colsample_bytree', 0.8)),
        }
        print(f"[INFO] tune=False -> training single model with fixed params: {best_params}")
        best_model = xgb.XGBClassifier(
            **best_params, eval_metric='logloss', random_state=88, scale_pos_weight=scale_pos_weight
        )
        best_model.fit(X_train_processed, y_train)

    print(f"[SUCCESS] Hyperparameters in use: {best_params}")

    # ---- TensorBoard: train a SEPARATE instrumented copy ONLY to stream the
    # per-boosting-round curves. This copy is never persisted, so the saved
    # artefact never carries a callback / SummaryWriter (which would otherwise
    # break unpickling) nor a list-valued eval_metric. ----
    model_version = "credit_model_" + config["model_train_date_str"].replace('-', '_')
    tb_log_dir = os.path.join("tensorboard_logs", model_version)
    writer = None
    try:
        from tensorboardX import SummaryWriter
        os.makedirs(tb_log_dir, exist_ok=True)
        writer = SummaryWriter(logdir=tb_log_dir)
        print(f"[INFO] TensorBoard logs -> {tb_log_dir} (view at http://localhost:6006)")
    except Exception as e:
        print(f"[WARN] TensorBoard logging disabled: {e}")

    if writer:
        tb_callbacks = [TensorBoardCallback(writer, {"validation_0": "train", "validation_1": "dev"})]
        tb_model = xgb.XGBClassifier(
            **best_params,
            eval_metric=["auc", "logloss"],
            random_state=88,
            scale_pos_weight=scale_pos_weight,
            callbacks=tb_callbacks,
        )
        tb_model.fit(
            X_train_processed, y_train,
            eval_set=[(X_train_processed, y_train), (X_test_processed, y_test)],
            verbose=False,
        )

    # 8. Validation metrics & Gini
    train_auc = roc_auc_score(y_train, best_model.predict_proba(X_train_processed)[:, 1])
    test_auc = roc_auc_score(y_test, best_model.predict_proba(X_test_processed)[:, 1])
    oot_auc = roc_auc_score(y_oot, best_model.predict_proba(X_oot_processed)[:, 1])

    print("\n--- Model Validation Diagnostic Summary ---")
    print(f"   -> Train AUC: {round(train_auc, 4)} | Gini: {round(2 * train_auc - 1, 3)}")
    print(f"   -> Dev   AUC: {round(test_auc, 4)}  | Gini: {round(2 * test_auc - 1, 3)}")
    print(f"   -> OOT   AUC: {round(oot_auc, 4)}  | Gini: {round(2 * oot_auc - 1, 3)}\n")

    # ---- TensorBoard: log final summary scalars + hyperparameters, then close ----
    if writer:
        writer.add_scalar("final/auc_train", train_auc, 0)
        writer.add_scalar("final/auc_dev", test_auc, 0)
        writer.add_scalar("final/auc_oot", oot_auc, 0)
        writer.add_scalar("final/gini_oot", 2 * oot_auc - 1, 0)
        try:
            hp_log = {k: (v if isinstance(v, (int, float, str, bool)) else str(v)) for k, v in best_params.items()}
            writer.add_hparams(hp_log, {"hparam/auc_dev": test_auc, "hparam/auc_oot": oot_auc})
        except Exception as e:
            print(f"[WARN] add_hparams skipped: {e}")
        writer.close()

    # 9. Serialize artefact (model_version already defined above for TensorBoard)
    model_artefact = {
        'model': best_model,
        'model_version': model_version,
        'feature_names': feature_cols,
        'preprocessing_transformers': {
            'imputer': imputer,
            'target_encoding': te_maps,
            'numeric_cols': numeric_cols,
            'categorical_cols': categorical_cols,
            'stdscaler': transformer_stdscaler
        },
        'data_dates': config,
        'data_stats': {
            'X_train': X_train_raw.shape[0], 'X_test': X_test_raw.shape[0], 'X_oot': X_oot.shape[0],
            'y_train': round(y_train.mean(), 3), 'y_test': round(y_test.mean(), 3), 'y_oot': round(y_oot.mean(), 3),
            'scale_pos_weight': round(scale_pos_weight, 3)
        },
        'results': {
            'auc_train': train_auc, 'auc_test': test_auc, 'auc_oot': oot_auc,
            'gini_train': round(2 * train_auc - 1, 3), 'gini_test': round(2 * test_auc - 1, 3), 'gini_oot': round(2 * oot_auc - 1, 3)
        },
        'hp_params': best_params
    }

    # 10. Persist to model bank
    model_bank_dir = "model_bank/"
    if not os.path.exists(model_bank_dir):
        os.makedirs(model_bank_dir)

    output_file_path = os.path.join(model_bank_dir, f"{model_version}.pkl")
    with open(output_file_path, 'wb') as file:
        pickle.dump(model_artefact, file)

    print(f"[SUCCESS] Core model artefact secured and exported to: {output_file_path}")
    print("\n=====================================================================")
    print("             PIPELINE SYSTEM SHUTDOWN COMPLETED CLEANLY              ")
    print("=====================================================================\n")


def _str2bool(v):
    """Parse a CLI/Airflow string like 'True'/'false'/'1' into a real bool."""
    return str(v).strip().lower() in ("true", "1", "yes", "y")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Production Credit Scoring Model Training Platform")
    parser.add_argument("--snapshot", type=str, required=True,
                        help="Training snapshot date YYYY-MM-DD (Train/Dev/OOT derived backwards from here)")
    # --- Hyperparameter controls (driven by Airflow params) ---
    parser.add_argument("--tune", type=str, default="true",
                        help="true = RandomizedSearchCV; false = use the fixed hyperparameters below")
    parser.add_argument("--n_estimators", type=int, default=50)
    parser.add_argument("--max_depth", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=0.1)
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--colsample_bytree", type=float, default=0.8)
    args = parser.parse_args()

    fixed_hp = {
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth,
        "learning_rate": args.learning_rate,
        "subsample": args.subsample,
        "colsample_bytree": args.colsample_bytree,
    }
    main(snapshot_date_arg=args.snapshot, tune=_str2bool(args.tune), hp=fixed_hp)
