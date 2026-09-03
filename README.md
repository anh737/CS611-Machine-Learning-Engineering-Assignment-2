# Credit Default Risk — Orchestrated ML Pipeline

An end-to-end credit-default prediction pipeline on Apache Airflow: a PySpark
medallion datamart, XGBoost training, batch inference and drift monitoring,
packaged with Docker. Nothing needs to be installed on the host besides Docker.

## Problem

Predict whether a loan will go bad, defined as **30 or more days past due within
the first 6 months on book**.

Data: 12,500 customers, 137,500 monthly loan snapshots, January 2023 to
November 2025.

## Pipeline

```
                        ┌──────────────────────────────────────┐
   data/ (raw CSVs) ──► │  data_pipeline_bronze_silver_gold    │  PySpark
                        │  raw → typed → cleaned → labelled    │
                        └──────────────────┬───────────────────┘
                                           ▼
                        ┌──────────────────────────────────────┐
                        │  train_model                         │  XGBoost
                        │  time-based split, tuned, → .pkl     │  → TensorBoard
                        └──────────────────┬───────────────────┘
                                           ▼
                        ┌──────────────────────────────────────┐
                        │  model_inference                     │  pandas
                        │  score gold → gold predictions table │
                        └──────────────────┬───────────────────┘
                                           ▼
                        ┌──────────────────────────────────────┐
                        │  model_monitoring                    │  AUC / Gini / PSI
                        │  per-month metrics + chart + verdict │
                        └──────────────────────────────────────┘
```

| Stage | Script | What it does |
|---|---|---|
| Data pipeline | `test.py` + `utils/` | Bronze ingest → silver clean → gold label and feature store. Runs when `run_data_pipeline=true`, or automatically when the gold store is empty. |
| Train | `train.py` | Chronological train/dev/OOT split, median imputation, target encoding, scaling, XGBoost with optional `RandomizedSearchCV`. Streams per-round curves to TensorBoard and pickles the artefact. |
| Inference | `inference.py` | Loads the latest artefact with its preprocessing objects, scores the gold store, writes a gold predictions table. |
| Monitoring | `monitoring.py` | Per-month AUC and Gini, PSI against a baseline month, a chart, and a drift verdict. |

## Quickstart

```bash
docker-compose build
docker-compose up
```

1. Open Airflow at **http://localhost:8080** (login `airflow` / `airflow`).
2. Un-pause the **`ml_pipeline`** DAG.
3. **▶ → Trigger DAG w/ config**, adjust parameters if needed, **Trigger**.

The first run builds the datamart automatically.

### Parameters

| Param | Default | Meaning |
|---|---|---|
| `run_data_pipeline` | `false` | Rebuild the bronze/silver/gold datamart. Also forced when the gold store is empty. |
| `train_snapshot` | `2024-12-01` | Training snapshot; train/dev/OOT windows are derived backwards from it. |
| `tune_hyperparameters` | `true` | `true` runs `RandomizedSearchCV`; `false` trains once with the fixed values below. |
| `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree` | 50 / 3 / 0.1 / 0.8 / 0.8 | Used only when tuning is off. |

### Outputs

| Path | Contents |
|---|---|
| `model_bank/credit_model_*.pkl` | model, imputer, target-encoding maps, scaler, training config |
| `datamart/gold/model_predictions/` | scored predictions, one parquet per month |
| `datamart/gold/model_monitoring/gold_model_monitoring.{parquet,csv}` | AUC, Gini and PSI by month |
| `datamart/gold/model_monitoring/model_monitoring_plot.png` | three-panel monitoring chart |

TensorBoard runs at **http://localhost:6006** once `train_model` has executed.

```bash
docker-compose down       # stop, keep data
docker-compose down -v    # also wipe the Airflow metadata DB
```

Ports: 8080 (Airflow), 6006 (TensorBoard), 5432 (Postgres, internal).

## Gold layer: label and feature anchoring

The label looks forward: the worst days-past-due a loan reaches in months 1–6
on book. The features are anchored at month 0, the application date.

```
   month 0            month 1 ─────────────────► month 6
   ┌──────────┐       ┌─────────────────────────────────┐
   │ FEATURES │       │  LABEL: worst DPD in this window │
   │ anchored │       └─────────────────────────────────┘
   │   here   │
   └──────────┘
```

An earlier version keyed the gold row on the latest snapshot in the 1–6 month
window. That had two effects: financials and attributes only exist at month 0,
so they joined as null; and clickstream joined at month 6, which included six
months of behaviour after the application. Anchoring at month 0 fixes both.
Only loans observed for the full window receive a label, and the gold row is
identical across partitions, so de-duplication in training is by `loan_id`.

`data_leakage_check.ipynb` contains the audit.

## Design decisions

- **Bronze is stored raw**, so cleaning rules can be revised and months replayed
  without re-ingesting from source.
- **Target encoding is fitted on train only**, with smoothing; unseen categories
  fall back to the training global mean. The maps are stored in the model
  artefact.
- **Inference runs without Spark.** The artefact carries its own imputer,
  encoding maps and scaler, so scoring is pure pandas.
- **Class imbalance is handled with `scale_pos_weight`** rather than resampling.
- **PSI thresholds:** below 0.10 stable, 0.10–0.25 investigate, above 0.25
  refresh the model. `monitoring.py` prints the verdict alongside the number.
- **Training curves go to TensorBoard** via an XGBoost callback on a separate
  instrumented copy of the model, so the persisted artefact carries no callback
  or `SummaryWriter`.

## Repository layout

```
dags/ml_pipeline_dag.py                   Airflow DAG: data → train → inference → monitor
test.py                                   bronze → silver → gold backfill (the DAG's first task)
train.py                                  training, TensorBoard, model bank
inference.py                              batch scoring → gold predictions
monitoring.py                             AUC / Gini / PSI → gold monitoring table + chart
utils/
  ├── data_processing_bronze_table.py     raw ingest
  ├── data_processing_silver_table.py     cleaning and schema enforcement
  └── data_processing_gold_table.py       labels and feature engineering
data/                                     raw source CSVs (pipeline input)
data_leakage_check.ipynb                  leakage audit
data_processing_main.ipynb                pipeline walkthrough
test.ipynb                                exploratory data cleaning
Dockerfile, docker-compose.yaml           Airflow + JDK 17 + Python ML deps
```

Generated at runtime and git-ignored: `datamart/`, `model_bank/`,
`tensorboard_logs/`, `logs/`.

## Known limitations

- `test.py` is the data-pipeline entrypoint, not a test suite.
- No automated tests.
- About 28% of customers have no clickstream rows, so `fe_1`–`fe_20` are null
  for them.
- PSI is baselined on the earliest scored month rather than the training-period
  score distribution.
- The raw CSVs (~36 MB) are committed to the repository.
- Inference picks the newest artefact in `model_bank/` by modification time;
  there is no promotion gate.

## Stack

Apache Airflow 2.9 · PySpark 3.5 · XGBoost · scikit-learn · pandas · TensorBoard ·
Postgres · Docker Compose

## License

MIT — see [LICENSE](LICENSE).
